"""Build training / calibration / evaluation sets in wire format.

    python training/build_data.py --per-task 1500

Writes data/train.jsonl (held-in tasks), data/calib.jsonl (held-in tasks, unseen
rows) and data/eval/<task>.jsonl (unseen rows of every task, held-out tasks included).
"""
import argparse
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from tasks import TASKS, generate  # noqa: E402


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=1500)
    ap.add_argument("--eval-per-task", type=int, default=300)
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--email-per-email", type=int, default=16, help="examples drawn from each hand-written triage email or comment; 0 to leave those families out")
    ap.add_argument("--seed", type=int, default=11, help="training sample seed; change it to draw different source rows")
    args = ap.parse_args()
    import datasets
    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bars()

    train, calib = [], []
    for task in TASKS:
        if args.only and task.name not in args.only:
            continue
        try:
            evals = generate(task, "eval", args.eval_per_task + (0 if task.holdout else 150), seed=7)
            if task.holdout:
                write(f"{args.out}/eval/{task.name}.jsonl", evals)
                print(f"  holdout  {task.name:18s} eval {len(evals)}", flush=True)
                continue
            write(f"{args.out}/eval/{task.name}.jsonl", evals[: args.eval_per_task])
            calib += evals[args.eval_per_task:]
            rows = generate(task, "train", args.per_task, seed=args.seed)
            train += rows
            print(f"  train    {task.name:18s} train {len(rows)}  eval {min(len(evals), args.eval_per_task)}", flush=True)
        except Exception as err:                          # a dataset going missing should not sink the build
            print(f"  FAILED   {task.name:18s} {type(err).__name__}: {str(err)[:120]}", flush=True)
    if not args.only and args.email_per_email > 0:
        # Hand-written families for the everyday questions public datasets do not cover.
        from comment_triage import examples as comment_examples
        from email_triage import examples as email_examples
        for name, make in (("email_triage", email_examples), ("comment_triage", comment_examples)):
            rows = make(args.email_per_email, args.seed)
            train += rows
            print(f"  train    {name:18s} train {len(rows)}  (hand-written, see training/{name}.py)", flush=True)
    random.Random(0).shuffle(train)
    random.Random(0).shuffle(calib)
    if not args.only:
        write(f"{args.out}/train.jsonl", train)
        write(f"{args.out}/calib.jsonl", calib)
    print(f"  total    train {len(train)}  calib {len(calib)}")


if __name__ == "__main__":
    main()
