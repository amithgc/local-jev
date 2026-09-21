"""Run the 100 comparison cases against several systems and write one HTML report.

    local-jev serve &                                        # the local server must be running
    python evals/compare/run_compare.py                      # Jev + every model the local server can load
    python evals/compare/run_compare.py --systems jev nli-deberta-large llm-qwen3-4b
    python evals/compare/run_compare.py --systems nli-deberta-large llm-qwen3.5-4b      # no hosted API involved

A system is either ``jev`` (the hosted API; the key is read from $TYPESAFE_API_KEY or secrets.txt) or the
name of a local model, as listed by ``local-jev models``. Local models are all called through the same
local server, selected with the request's ``model`` field, so the comparison exercises the real API.

Raw responses are cached per system under evals/compare/results/ (git-ignored): re-running only re-renders
the report, ``--refresh`` calls again. Cached Jev responses exist for this side-by-side report only. Nothing
under training/ or data/ reads them, and they must not be used as training labels (see "What is
deliberately not done" in docs/training.md).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import re
import statistics
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]


def read_key(path: pathlib.Path) -> str | None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    if path.exists():
        found = re.search(r"apikey_[A-Za-z0-9_]+", path.read_text())
        return found.group(0) if found else None
    return None


def call(base_url: str, model: str, case: dict, key: str | None, timeout: float = 30.0) -> dict:
    body = json.dumps({"state": case["state"], "model": model, "questions": {"q": case["question"]}}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key or 'local'}"}
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone", body, headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        return {"ms": round((time.perf_counter() - started) * 1000, 1), "model": payload.get("model"),
                "answer": payload["answers"]["q"], "usage": payload.get("usage")}
    except urllib.error.HTTPError as err:
        return {"ms": round((time.perf_counter() - started) * 1000, 1), "error": f"HTTP {err.code}: {err.read().decode()[:200]}"}
    except Exception as err:                                       # a failed case should not sink the run
        return {"ms": round((time.perf_counter() - started) * 1000, 1), "error": f"{type(err).__name__}: {err}"}


def run_system(name, base_url, model, cases, key, workers, cache_path, refresh):
    cache = {} if refresh or not cache_path.exists() else json.loads(cache_path.read_text())
    todo = [c for c in cases if c["id"] not in cache or "error" in cache[c["id"]]]
    if todo:
        call(base_url, model, todo[0], key)                          # warm-up, so the first case is not timed cold
        print(f"  {name}: calling {len(todo)} cases ({workers} at a time)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for case, result in zip(todo, pool.map(lambda c: call(base_url, model, c, key), todo)):
                cache[case["id"]] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1))
    else:
        print(f"  {name}: using {len(cache)} cached responses ({cache_path.name}); --refresh to call again")
    return cache


def verdict(case: dict, result: dict) -> dict:
    """Reduce a raw answer to: what it said, how sure it was, and whether that matches the gold answer."""
    if "error" in result:
        return {"error": result["error"], "ms": result["ms"]}
    answer, question, gold = result["answer"], case["question"], case["gold"]
    kind = question["type"]
    if kind == "noul":
        p = float(answer["noul"])
        said, sure, right = p >= 0.5, max(p, 1 - p), (p >= 0.5) == gold
        label, probs = ("Yes" if said else "No"), {"Yes": p, "No": 1 - p}
    elif kind == "choice":
        said, sure, right = answer["choice"], float(answer.get("confidence", 0)), answer["choice"] == gold
        label, probs = said, answer.get("probabilities", {})
    else:
        score = float(answer["score"])
        said = min(max(int(round(score)), 0), len(question["criteria"]) - 1)
        sure, right = float(answer.get("confidence", 0)), said == gold
        label = f"{score + 1:.2f} of {len(question['criteria'])}"
        probs = {str(int(k) + 1): v for k, v in answer.get("probabilities", {}).items()}
    return {"said": said, "label": label, "sure": round(sure, 4), "right": bool(right), "probs": probs, "ms": result["ms"],
            "err": abs(float(answer["score"]) - gold) if kind == "score" else None}


def gold_label(case):
    q, gold = case["question"], case["gold"]
    if q["type"] == "noul":
        return "Yes" if gold else "No"
    if q["type"] == "score":
        return f"{gold + 1} of {len(q['criteria'])}: {q['criteria'][gold]}"
    return gold


def summarise(rows, system, reference=None):
    good = [(r, r["answers"][system]) for r in rows if "error" not in r["answers"][system]]
    out = {"n": len(good), "errors": len(rows) - len(good)}
    if not good:
        return out
    verdicts = [v for _, v in good]
    out["accuracy"] = sum(v["right"] for v in verdicts) / len(verdicts)
    out["median_ms"] = statistics.median(v["ms"] for v in verdicts)
    out["p90_ms"] = sorted(v["ms"] for v in verdicts)[max(int(len(verdicts) * 0.9) - 1, 0)]
    sure = [v for v in verdicts if v["sure"] >= 0.8]
    unsure = [v for v in verdicts if v["sure"] < 0.8]
    out["confident_share"] = len(sure) / len(verdicts)
    out["confident_accuracy"] = (sum(v["right"] for v in sure) / len(sure)) if sure else None
    out["unsure_accuracy"] = (sum(v["right"] for v in unsure) / len(unsure)) if unsure else None
    errs = [v["err"] for v in verdicts if v.get("err") is not None]
    out["score_mae"] = statistics.mean(errs) if errs else None
    if reference and reference != system:
        both = [(v, r["answers"][reference]) for r, v in good if "error" not in r["answers"][reference]]
        out["agreement"] = (sum(v["said"] == ref["said"] for v, ref in both) / len(both)) if both else None
    return out


def markdown(data) -> str:
    """The summary as a Markdown table, for pasting into documentation."""
    pct = lambda x: "-" if x is None else f"{x:.0%}"                                            # noqa: E731
    kinds = sorted(data["breakdown"]["type"])
    names = {"choice": "Category", "noul": "Yes/No", "score": "Score"}
    lines = ["| System | Backend | Accuracy | " + " | ".join(names[k] for k in kinds) + " | Says it is sure | Accuracy when sure | Median latency |",
             "|---|---|---|" + "---|" * len(kinds) + "---|---|---|"]
    for system in data["systems"]:
        s = data["summary"][system["id"]]
        per_kind = " | ".join(pct(data["breakdown"]["type"][k][system["id"]].get("accuracy")) for k in kinds)
        lines.append(f"| {system['label']} | {system['kind']} | **{pct(s.get('accuracy'))}** | {per_kind} | {pct(s.get('confident_share'))} | "
                     f"{pct(s.get('confident_accuracy'))} | {s.get('median_ms', 0):.0f} ms |")
    return "\n".join(lines) + "\n"


def local_models(base_url):
    with urllib.request.urlopen(base_url.rstrip("/") + "/api/meta", timeout=10) as response:
        return json.load(response)["models"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", help="'jev' and/or local model names; default: jev plus every local model that can be loaded")
    ap.add_argument("--local-url", default="http://127.0.0.1:8765")
    ap.add_argument("--jev-url", default="https://api.typesafe.ai")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--key-file", default=str(ROOT / "secrets.txt"))
    ap.add_argument("--workers", type=int, default=4, help="parallel requests to the hosted API (local requests run one at a time so latency is clean)")
    ap.add_argument("--refresh", nargs="*", metavar="SYSTEM", help="call these systems again instead of using cached responses; with no names, all of them")
    ap.add_argument("--out", default=str(HERE / "report.html"))
    ap.add_argument("--markdown", help="also write the summary as a Markdown table to this path (for docs)")
    args = ap.parse_args()

    cases = json.loads((HERE / "cases.json").read_text())
    known = {m["name"]: m for m in local_models(args.local_url)}
    names = args.systems or (["jev"] + [n for n, m in known.items() if not m.get("unavailable")])
    refresh = set(names) if args.refresh == [] else set(args.refresh or [])

    systems, raw = [], {}
    for name in names:
        cache = HERE / "results" / f"{name}.json"
        if name == "jev":
            key = read_key(pathlib.Path(args.key_file))
            if not key:
                raise SystemExit("No Jev API key: set TYPESAFE_API_KEY or put it in secrets.txt, or leave 'jev' out of --systems.")
            raw[name] = run_system(name, args.jev_url, args.jev_model, cases, key, args.workers, cache, name in refresh)
            systems.append({"id": name, "label": "Jev", "kind": "hosted API", "where": "over the internet"})
        else:
            if name not in known:
                raise SystemExit(f"The local server has no model {name!r}. It offers: {', '.join(known)}")
            if known[name].get("unavailable"):
                raise SystemExit(f"{name}: {known[name]['unavailable']}")
            raw[name] = run_system(name, args.local_url, name, cases, None, 1, cache, name in refresh)
            systems.append({"id": name, "label": name, "kind": f"{known[name]['backend']} backend", "where": "on this machine",
                            "description": known[name].get("description", ""), "license": known[name].get("license", ""),
                            "evaluation": known[name].get("evaluation")})
    for system in systems:
        system["model"] = next((r.get("model") for r in raw[system["id"]].values() if r.get("model")), system["id"])

    rows = []
    for case in cases:
        rows.append({"id": case["id"], "domain": case["domain"], "type": case["question"]["type"], "state": case["state"],
                     "instructions": case["question"].get("instructions"), "gold": gold_label(case),
                     "answers": {name: verdict(case, raw[name][case["id"]]) for name in names}})

    reference = "jev" if "jev" in names else None
    axes = {"type": sorted({r["type"] for r in rows}), "domain": sorted({r["domain"] for r in rows})}
    data = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "systems": systems, "reference": reference, "rows": rows,
            "summary": {name: summarise(rows, name, reference) for name in names},
            "breakdown": {axis: {value: {name: summarise([r for r in rows if r[axis] == value], name) for name in names}
                                 for value in values} for axis, values in axes.items()}}
    template = (HERE / "report_template.html").read_text()
    pathlib.Path(args.out).write_text(template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))

    if args.markdown:
        pathlib.Path(args.markdown).write_text(markdown(data))
        print(f"  markdown: {args.markdown}")

    print()
    for system in systems:
        s = data["summary"][system["id"]]
        if s.get("n"):
            print(f"  {system['label']:20s} {system['kind']:16s} accuracy {s['accuracy']:5.0%}   median {s['median_ms']:5.0f} ms   errors {s['errors']}")
    print(f"\n  report: {args.out}")


if __name__ == "__main__":
    main()
