"""Fit one temperature per primitive on held-back rows, and record it in the model's card.

    python training/calibrate.py --model llm-qwen3-4b
    python training/calibrate.py --model nli-deberta-large --kinds score      # refit only some primitives

Works for any model in the registry. It writes models/cards/<name>.json, which overrides the built-in
card of the same name; restart the server to pick it up.

Temperature scaling cannot change which answer wins; it only makes the reported probabilities honest, so
that "0.8" is right about 80% of the time. Instruction-tuned LLMs in particular are badly overconfident
read raw (almost every answer at 1.00), which makes confidence-gated routing useless until this is done.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))
from run_eval import collect  # noqa: E402


def nll_at(logp_rows, gold, t):
    total = 0.0
    for logp, g in zip(logp_rows, gold):
        z = logp / t
        z = z - z.max()
        total -= z[g] - np.log(np.exp(z).sum())
    return total / len(gold)


def level_error_at(logp_rows, gold, t):
    """Mean distance between the expected level and the true level, at temperature t."""
    total = 0.0
    for logp, g in zip(logp_rows, gold):
        z = logp / t
        p = np.exp(z - z.max())
        p /= p.sum()
        total += abs(float(np.dot(np.arange(len(p)), p)) - g)
    return total / len(gold)


def fit(logp_rows, gold, kind="choice"):
    """Pick the temperature. Returns (T, objective before, objective after, objective name).

    Choices and yes/no answers are judged on their probabilities, so they minimise negative
    log-likelihood. A score is different: what callers read is the *expected level*, and a temperature
    that makes the probabilities honest on noisy data also drags that expectation toward the middle of
    the scale ("complete waste of money" came out as 1.5 of 5). So scores minimise the error of the
    expected level instead.
    """
    grid = np.exp(np.linspace(np.log(0.3), np.log(15.0), 160))   # raw LLM logits can need T well above 5
    objective = level_error_at if kind == "score" else nll_at
    if kind == "score":
        grid = grid[grid >= 1.0]          # never sharpen a score: that buys a sliver of accuracy and destroys its confidence
    losses = [objective(logp_rows, gold, t) for t in grid]
    return (float(grid[int(np.argmin(losses))]), float(objective(logp_rows, gold, 1.0)), float(min(losses)),
            "level_error" if kind == "score" else "nll")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a model from the registry (see `local-jev models`)")
    ap.add_argument("--calib", default="data/calib.jsonl")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--kinds", nargs="*", default=["choice", "score", "noul"], help="refit only these primitives, keeping the card's other temperatures")
    args = ap.parse_args()

    from local_jev.engine import Engine
    from local_jev.models import BUILTIN_CARDS_DIR, USER_CARDS_DIR
    backend = Engine(default=args.model).backend(args.model)

    rows = [r for r in (json.loads(line) for line in open(args.calib)) if r["question"]["type"] in args.kinds][: args.n]
    probs = collect(backend, rows)                                   # uncalibrated: no temperatures passed
    temperature, detail = {}, {}
    for kind in ("choice", "score", "noul"):
        idx = [i for i, r in enumerate(rows) if r["question"]["type"] == kind]
        if len(idx) < 50:
            continue
        logp = [np.log(np.clip(probs[i], 1e-9, 1)) for i in idx]
        t, before, after, objective = fit(logp, [rows[i]["gold"] for i in idx], kind)
        temperature[kind] = round(t, 3)
        detail[kind] = {"n": len(idx), "objective": objective, "before": round(before, 4), "after": round(after, 4)}
        print(f"  {kind:7s} T = {t:.3f}   {objective} {before:.4f} -> {after:.4f}   (n={len(idx)})")

    source = next((d / f"{args.model}.json" for d in (USER_CARDS_DIR, BUILTIN_CARDS_DIR) if (d / f"{args.model}.json").exists()), None)
    if source is None:
        raise SystemExit(f"{args.model} has no card file to update")
    card = json.load(open(source))
    previous = card.get("calibration") or {}
    card["calibration"] = {"temperature": {**(previous.get("temperature") or {}), **temperature},
                           "fit": {**(previous.get("fit") or {}), **detail}}
    USER_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(card, open(USER_CARDS_DIR / f"{args.model}.json", "w"), indent=1)
    print(f"  wrote models/cards/{args.model}.json (overrides the built-in card; restart the server to pick it up)")


if __name__ == "__main__":
    main()
