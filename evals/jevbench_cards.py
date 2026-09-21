"""Record JevBench public-set results in the model cards, where the UI's model picker reads them.

    # 1. run JevBench's public items against a running local-jev server, one model at a time
    #    (see "Independent benchmark" in docs/training.md for the exact commands)
    # 2. then:
    python evals/jevbench_cards.py --runs <folder with one <model>.jsonl per model> --tasks <public.jsonl>

Accuracy is the share of the benchmark's public decisions whose top answer is correct; failed requests count as
wrong. Latency is measured by the harness around each HTTP request, one decision per request, so it includes
the local API. Built-in cards are updated in place; other models get an override in models/cards/.
"""
import argparse
import datetime
import json
import pathlib
import platform
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from local_jev.models import BUILTIN_CARDS_DIR, USER_CARDS_DIR, discover  # noqa: E402

JEV_PUBLIC = 0.866        # Jev 1.13.0 on the same 231 public items, from JevBench v1.2's per-task results


def hardware() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def summarise(path: pathlib.Path, tiers: dict) -> dict:
    rows = [json.loads(line) for line in path.open()]
    by = {t: [0, 0] for t in ("easy", "standard", "hard")}
    for r in rows:
        tier = tiers[r["task_id"]]
        by[tier][1] += 1
        by[tier][0] += bool(r.get("correct"))
    total = sum(v[1] for v in by.values())
    latency = sorted(1000 * r["latency_s"] for r in rows if r.get("ok") and r.get("latency_s"))
    return {"benchmark": "JevBench public", "items": total,
            "accuracy": round(sum(v[0] for v in by.values()) / total, 4),
            "by_tier": {t: round(c / n, 4) for t, (c, n) in by.items() if n},
            "p50_ms": round(statistics.median(latency)) if latency else None,
            "p95_ms": round(latency[max(int(len(latency) * 0.95) - 1, 0)]) if latency else None,
            "failed": sum(not r.get("ok") for r in rows), "hardware": hardware(),
            "measured": datetime.date.today().isoformat(), "reference": {"jev-1.13.0": JEV_PUBLIC}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tasks", required=True, help="the public.jsonl the runs used (easy + original + hard)")
    args = ap.parse_args()
    tiers = {}
    for line in open(args.tasks):
        tid = json.loads(line)["id"]
        tiers[tid] = "easy" if tid.startswith("easy") else "hard" if tid.startswith("hard") else "standard"
    cards = discover()
    for path in sorted(pathlib.Path(args.runs).glob("*.jsonl")):
        name = path.stem
        if name not in cards:
            continue
        evaluation = summarise(path, tiers)
        if evaluation["items"] != len(tiers):
            print(f"  skip {name}: {evaluation['items']}/{len(tiers)} items (incomplete run)")
            continue
        target = BUILTIN_CARDS_DIR / f"{name}.json"
        if not target.exists():
            target = USER_CARDS_DIR / f"{name}.json"
        card = json.loads(target.read_text()) if target.exists() else {"name": name}
        card["evaluation"] = evaluation
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(card, indent=2) + "\n")
        print(f"  {name:22s} {evaluation['accuracy']:.1%}  p50 {evaluation['p50_ms']} ms  -> {target}")


if __name__ == "__main__":
    main()
