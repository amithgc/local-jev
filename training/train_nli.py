"""Fine-tune the entailment (nli) backend on this project's gold-labelled, wire-format data.

    python training/train_nli.py --data data/train.jsonl --val data/calib.jsonl --out models/hf/nli-deberta-large-ft1

Every example is rendered through the backend's own ``hypotheses()``, so training and inference
cannot drift apart. The loss mirrors how the backend turns model outputs into answers:

  choice / score   the options of one question compete: softmax over their entailment log-odds,
                   cross-entropy against the correct option (the gold option plus sampled rivals;
                   for scores, neighbouring levels are preferred rivals because they are the hard ones)
  yes / no         binary cross-entropy on the sigmoid of the single hypothesis's log-odds

The base model is already a strong zero-shot judge (95% on the comparison set untuned). Fine-tuning can
erode exactly that generality, so the defaults are gentle: a low learning rate, one pass over a subset,
and a validation read-out split by primitive. Judge the result on the held-out tasks
(``evals/run_eval.py --model <name>``) and the comparator before adopting it.

The result is a Hugging Face model directory plus a card under models/cards/, so the server lists it.
Uses CUDA if present (bf16 autocast), else Apple MPS, else the CPU.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from local_jev.backends._hf import load_pretrained, pick_device, require_torch  # noqa: E402
from local_jev.backends.nli import hypotheses  # noqa: E402
from local_jev.questions import as_text, parse_question  # noqa: E402


def load_examples(path, limit, max_options, rng):
    """-> list of (premise, [hypotheses], index of the correct one | None, yes/no truth | None, type)"""
    rows = [json.loads(line) for line in open(path)]
    rng.shuffle(rows)
    out = []
    for row in rows:
        if limit and len(out) >= limit:
            break
        try:
            q = parse_question(row["question"])
        except Exception:
            continue
        hyps, gold, premise = hypotheses(q), row["gold"], as_text(row["state"])
        if q.type == "noul":
            out.append((premise, hyps, None, gold == 0, "noul"))
            continue
        rivals = [i for i in range(len(hyps)) if i != gold]
        if q.type == "score":                                   # neighbours first: they are the distinction that matters
            rivals.sort(key=lambda i: (abs(i - gold), rng.random()))
        else:
            rng.shuffle(rivals)
        keep = [gold] + rivals[: max_options - 1]
        rng.shuffle(keep)
        out.append((premise, [hyps[i] for i in keep], keep.index(gold), None, q.type))
    return out


def batches(examples, pair_budget, rng):
    """Group examples so each batch holds about ``pair_budget`` (premise, hypothesis) pairs of similar length."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    out, current, pairs = [], [], 0
    for i in order:
        n = len(examples[i][1])
        if current and pairs + n > pair_budget:
            out.append(current)
            current, pairs = [], 0
        current.append(i)
        pairs += n
    if current:
        out.append(current)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--val", default="data/calib.jsonl")
    ap.add_argument("--base", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    ap.add_argument("--out", default="models/hf/nli-deberta-large-ft1")
    ap.add_argument("--name", help="model name for the card; defaults to the last part of --out")
    ap.add_argument("--examples", type=int, default=40000, help="training examples to use (one pass); 0 = all of them")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--pair-budget", type=int, default=32, help="(premise, hypothesis) pairs per step; lower it if you run out of memory")
    ap.add_argument("--max-options", type=int, default=6, help="options per choice/score example: the correct one plus sampled rivals")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--val-examples", type=int, default=1200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    args = ap.parse_args()

    torch = require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
    device = pick_device(torch, args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log = open(args.out.rstrip("/") + ".log", "a")

    def emit(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    # Master weights must be float32. The published checkpoint is stored in float16 and transformers keeps that
    # dtype on load; AdamW's epsilon (1e-8) underflows to zero in float16, so the very first update divides by
    # zero and every weight becomes NaN. Speed comes from bf16 autocast on CUDA instead.
    model = load_pretrained(AutoModelForSequenceClassification, args.base, torch.float32).to(device)
    assert all(p.dtype == torch.float32 for p in model.parameters()), "training needs float32 master weights"
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    entail = labels.get("entailment", 0)
    against = [i for name, i in labels.items() if name != "entailment"]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    train = load_examples(args.data, args.examples or None, args.max_options, rng)
    val = load_examples(args.val, args.val_examples, args.max_options, random.Random(1))
    kinds = {k: sum(e[4] == k for e in train) for k in ("choice", "noul", "score")}
    emit(f"device: {device}   base: {args.base}")
    emit(f"data: {len(train)} train examples {kinds}, {len(val)} val examples")

    def log_odds(group):
        """Entailment log-odds for every (premise, hypothesis) pair of a group of examples, flattened."""
        premises = [e[0] for e in group for _ in e[1]]
        hyps = [h for e in group for h in e[1]]
        enc = tokenizer(premises, hyps, return_tensors="pt", padding=True, truncation="only_first",
                        max_length=args.max_length).to(device)
        with autocast:
            logits = model(**enc).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp[:, entail] - torch.logsumexp(logp[:, against], dim=-1)

    def loss_and_hits(group):
        z, cursor, losses, hits = log_odds(group), 0, [], []
        for premise, hyps, gold, truth, kind in group:
            mine = z[cursor: cursor + len(hyps)]
            cursor += len(hyps)
            if kind == "noul":
                target = torch.tensor(1.0 if truth else 0.0, device=z.device)
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(mine[0], target))
                hits.append((kind, bool((mine[0] > 0).item()) == truth))
            else:
                losses.append(torch.nn.functional.cross_entropy(mine[None, :], torch.tensor([gold], device=z.device)))
                hits.append((kind, int(mine.argmax().item()) == gold))
        return torch.stack(losses).mean(), hits

    def validate():
        model.eval()
        total, n, by = 0.0, 0, {}
        with torch.no_grad():
            for idx in batches(val, args.pair_budget * 2, random.Random(2)):
                loss, hits = loss_and_hits([val[i] for i in idx])
                total += float(loss) * len(idx)
                n += len(idx)
                for kind, ok in hits:
                    by.setdefault(kind, []).append(ok)
        model.train()
        return total / n, {k: sum(v) / len(v) for k, v in sorted(by.items())}

    plan = []
    for _ in range(math.ceil(args.epochs)):
        plan += batches(train, args.pair_budget, rng)
    plan = plan[: int(len(plan) * args.epochs / math.ceil(args.epochs))]
    if args.max_steps:
        plan = plan[: args.max_steps]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, len(plan) // 20), len(plan))
    emit(f"schedule: {len(plan)} steps, peak lr {args.lr:g}, about {args.pair_budget} pairs per step, max {args.max_length} tokens")

    vloss, vacc = validate()
    emit(f"step 0/{len(plan)}  val loss {vloss:.4f}  val acc {json.dumps({k: round(v, 3) for k, v in vacc.items()})}  (untuned)")
    model.train()
    started, running = time.time(), []
    for step, idx in enumerate(plan, 1):
        loss, _ = loss_and_hits([train[i] for i in idx])
        if not torch.isfinite(loss):
            raise SystemExit(f"loss became {float(loss)} at step {step}; stopping before the weights are ruined")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        running.append(loss.item())
        if step % 25 == 0 or len(plan) <= 12:
            rate = (time.time() - started) / step
            emit(f"step {step}/{len(plan)}  loss {sum(running[-25:]) / len(running[-25:]):.4f}  {rate:.2f}s/step  eta {rate * (len(plan) - step) / 60:.0f}m")
        if step % args.eval_every == 0 or step == len(plan):
            vloss, vacc = validate()
            emit(f"step {step}/{len(plan)}  val loss {vloss:.4f}  val acc {json.dumps({k: round(v, 3) for k, v in vacc.items()})}")
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)

    name = args.name or os.path.basename(args.out.rstrip("/"))
    from local_jev.models import USER_CARDS_DIR
    USER_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    card = {"name": name, "backend": "nli", "priority": 75, "hf_model": os.path.abspath(args.out), "context_tokens": 512,
            "description": f"{args.base} fine-tuned by training/train_nli.py on {len(train)} examples of this project's gold-labelled data.",
            "release_date": time.strftime("%Y-%m-%d"), "license": "MIT base model (MoritzLaurer/deberta-v3-large-zeroshot-v2.0)"}
    json.dump(card, open(USER_CARDS_DIR / f"{name}.json", "w"), indent=1)
    emit(f"done: {args.out}  +  models/cards/{name}.json   next: python training/calibrate.py --model {name}")


if __name__ == "__main__":
    main()
