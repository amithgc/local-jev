"""Evaluate a model on every task under data/eval/.

    python evals/run_eval.py --model nli-deberta-large --n 100

Reports accuracy and calibration per task, then rolled up by primitive and by whether a
fine-tuned model saw the task in training ("held-in": unseen rows of trained tasks;
"held-out": tasks never trained on). For the zero-shot built-in models every task is unseen.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.dirname(__file__))
import local_jev  # noqa: E402,F401
from local_jev.backends import Item  # noqa: E402
from local_jev.engine import Engine, calibrate  # noqa: E402
from local_jev.questions import parse_question  # noqa: E402
from metrics import reliability, summarise  # noqa: E402

HOLDOUT = {"ag_news", "banking77", "emotion", "sms_spam", "phishing", "clickbait", "rte", "boolq", "yelp"}


def load(path, n):
    with open(path) as handle:
        return [json.loads(line) for line in handle][:n]


def collect(backend, rows, temperature=None, max_state_tokens=420, batch=16):
    """Calibrated probability vectors for wire-format examples, in order. Works with any backend."""
    temperature = temperature or {}
    probs = []
    for start in range(0, len(rows), batch):
        items = [Item(parse_question(r["question"]), r["state"]) for r in rows[start:start + batch]]
        for item, scored in zip(items, backend.score(items, truncate=True, max_state_tokens=max_state_tokens)):
            probs.append(calibrate(scored.probs, temperature.get(item.question.type, 1.0)))
    return probs


def make_backend(args):
    """Returns (backend, display name, backend kind, temperatures)."""
    engine = Engine(default=args.model)
    card = engine.models[engine.default_model]
    return engine.backend(card.name), args.name or card.name, card.backend, card.temperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nli-deberta-large", help="any name from `local-jev models`")
    ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--out", default="evals/reports")
    args = ap.parse_args()

    backend, name, mode, temperature = make_backend(args)
    report = {"model": name, "backend": mode, "n_per_task": args.n, "tasks": {}, "created": time.strftime("%Y-%m-%d %H:%M")}
    pooled = {}
    started = time.time()
    for path in sorted(glob.glob("data/eval/*.jsonl")):
        task = os.path.basename(path)[:-6]
        if args.tasks and task not in args.tasks:
            continue
        rows = load(path, args.n)
        probs = collect(backend, rows, temperature)
        for kind in sorted({r["question"]["type"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["question"]["type"] == kind]
            p, g = [probs[i] for i in idx], [rows[i]["gold"] for i in idx]
            report["tasks"][f"{task}:{kind}"] = {**summarise(p, g, kind), "holdout": task in HOLDOUT}
            group = pooled.setdefault(("held-out" if task in HOLDOUT else "held-in", kind), ([], []))
            group[0].extend(p); group[1].extend(g)
        top = max(report["tasks"].items(), key=lambda kv: kv[0].startswith(task + ":") and kv[1]["n"])[1]
        print(f"  {task:18s} acc {top['accuracy']:.3f} (chance {top['chance']:.2f})  ece {top['ece']:.3f}  "
              f"[{time.time() - started:.0f}s]", flush=True)

    report["summary"] = {}
    for (split, kind), (p, g) in sorted(pooled.items()):
        entry = summarise(p, g, kind)
        conf = np.array([float(np.max(x)) for x in p])
        hit = np.array([int(np.argmax(x)) == y for x, y in zip(p, g)])
        entry["reliability"] = reliability(conf, hit)
        report["summary"][f"{split}:{kind}"] = entry
    # Macro average over tasks, so big tasks do not drown small ones.
    for split in ("held-in", "held-out"):
        accs = [t["accuracy"] for t in report["tasks"].values() if t["holdout"] == (split == "held-out")]
        chance = [t["chance"] for t in report["tasks"].values() if t["holdout"] == (split == "held-out")]
        eces = [t["ece"] for t in report["tasks"].values() if t["holdout"] == (split == "held-out")]
        if accs:
            report["summary"][f"{split}:macro"] = {"tasks": len(accs), "accuracy": float(np.mean(accs)),
                                                   "chance": float(np.mean(chance)), "ece": float(np.mean(eces))}
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{name}.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"\n  {name}  ({time.time() - started:.0f}s)")
    for key, entry in report["summary"].items():
        extra = f"  mae {entry['mae']:.3f}" if "mae" in entry else ""
        nll = f"  nll {entry['nll']:.3f}" if "nll" in entry else ""
        print(f"  {key:18s} acc {entry['accuracy']:.3f}  chance {entry['chance']:.3f}  ece {entry['ece']:.3f}{nll}{extra}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
