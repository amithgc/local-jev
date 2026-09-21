# Comparison results

Every number local-jev reports, in one place. Other documents link here rather than repeat these tables.

**Measured:** 2026-09-21, on an Apple M4 Max (36 GB) with PyTorch on the Mac GPU (MPS), models used zero-shot as published, with the calibration shipped in their cards. Local models were always called through local-jev's own `/v1/systemone`, so every latency includes the local HTTP API.

## Contents

1. [JevBench public items](#1-jevbench-public-items): the headline
2. [The 100 hand-written cases](#2-the-100-hand-written-cases): a regression check
3. [Prompt layout experiment](#3-prompt-layout-experiment)
4. [The removed ensemble](#4-the-removed-ensemble)
5. [Performance](#5-performance)
6. [Reproducing](#6-reproducing)
7. [Caveats](#7-caveats)

## 1. JevBench public items

[JevBench](https://github.com/fstandhartinger/jevbench) is an independent benchmark for Jev-class systems by Benchmark Heaven (MIT). Its 231 public items were run (48 easy, 72 standard, 111 hard); the other 303 items are held out by the benchmark and were not run. One decision per request, through JevBench's own `typesafe` adapter. Accuracy is the share of the 231 decisions whose top answer is correct; failed requests count as wrong (there were none). Latency is measured by the harness around each HTTP request.

| System | Accuracy | Easy (48) | Standard (72) | Hard (111) | Median | p95 |
|---|---|---|---|---|---|---|
| Jev 1.13.0, hosted (published result) | **86.6%** | 48 | 71 | 81 | | |
| SemIf, Qwen3.5-4B (published result) | **81.0%** | 48 | 71 | 68 | | |
| `llm-qwen3.5-4b` (the default) | **80.5%** | 48 | 69 | 69 | 651 ms | 6,860 ms |
| `llm-qwen3-4b` | **70.1%** | 48 | 59 | 55 | 177 ms | 2,210 ms |
| `llm-qwen3.5-2b` | **66.2%** | 48 | 56 | 49 | 261 ms | 2,658 ms |
| `llm-qwen2.5-1.5b` | **58.9%** | 48 | 49 | 39 | 74 ms | 783 ms |
| `nli-deberta-large` | **54.1%** | 45 | 41 | 39 | 76 ms | 710 ms |

- The default model is two items short of Jev on the easy and standard tiers and twelve short on the hard tier, where the questions are policy application and multi-step judgments.
- The two smallest models answer in about 75 ms but miss most questions beyond clear-cut classification.
- For reference, another team's fine-tuned DeBERTa scores 52.4% on the same items.
- Fine-tuning the entailment model on this project's 40,000 training examples lifted it from 73.9% to 82.5% on nine held-out public tasks, but only from 54.1% to 55.8% here, within noise ([details](training.md#results-so-far)). It is not a built-in model.

These results are stored in each built-in model card's `evaluation` field, where the portal's model picker and Models page read them.

## 2. The 100 hand-written cases

`evals/compare/cases.json`: 100 cases written for this project across 8 domains (support tickets, email, product reviews, comments, news headlines, assistant intents, PII screening, claim checking), 43 category, 40 yes/no and 17 score questions, each with a known answer. One question per request; Jev over the internet.

| System | Backend | Accuracy | Category | Yes/No | Score | Says it is sure | Accuracy when sure | Median latency |
|---|---|---|---|---|---|---|---|---|
| Jev | hosted API | **100%** | 100% | 100% | 100% | 100% | 100% | 842 ms |
| llm-qwen3.5-4b | llm backend | **100%** | 100% | 100% | 100% | 93% | 100% | 478 ms |
| llm-qwen3-4b | llm backend | **96%** | 100% | 95% | 88% | 94% | 98% | 162 ms |
| llm-qwen3.5-2b | llm backend | **91%** | 100% | 88% | 76% | 68% | 99% | 200 ms |
| llm-qwen2.5-1.5b | llm backend | **87%** | 100% | 82% | 65% | 62% | 98% | 68 ms |
| nli-deberta-large | nli backend | **95%** | 98% | 98% | 82% | 46% | 100% | 46 ms |

"Says it is sure" is the share of cases answered with a confidence of at least 0.8, and "accuracy when sure" the accuracy on those. This column is where calibration shows: every local model is right 98% to 100% of the time when it says it is sure.

**Use this set as a regression check, not a ranking.** It is too easy. Every system scores 87% or more here, while on JevBench the same systems range from 54.1% to 86.6%. The entailment model scores 95% here and 54.1% on JevBench. The personal addresses in the cases were moved to the reserved `example.com` domain before publication, and a re-run gave identical results.

## 3. Prompt layout experiment

The `llm` backend can lay a question out as plain text or as one JSON object (SemIf's layout); each card's `"prompt"` option chooses. JevBench public items:

| Model | `plain` | `json` | Shipped |
|---|---|---|---|
| Qwen3.5-4B | 73.2% (standard 58, hard 63) | **80.5%** (standard 69, hard 69) | `json` |
| Qwen3.5-2B | 64.1% | **66.2%** | `json` |
| Qwen3-4B | **70.1%** | 67.5% | `plain` |

With the plain layout Qwen3.5 was over-cautious, answering "no" where a policy in the state clearly permitted the action, and "other" where a request was clear. The JSON layout reproduces SemIf's own result for the same model (81.0%).

Precision: on Qwen3-4B, `bfloat16` scored 70.6% against 70.1% for the default `float16`, no meaningful difference.

## 4. The removed ensemble

`ensemble-nli-qwen3` pooled `nli-deberta-large` and `llm-qwen3-4b` (a weighted geometric mean of their calibrated probabilities).

| | 100 hand-written cases | JevBench public items |
|---|---|---|
| `ensemble-nli-qwen3` | **98%**, the best local result at the time | **68.8%** |
| `llm-qwen3-4b` alone | 96% | 70.1% |
| `nli-deberta-large` alone | 95% | 54.1% |

On the independent benchmark the ensemble was less accurate than its stronger member alone, and slower, so its built-in card was removed. The `ensemble` backend remains for your own cards. This result is also the clearest evidence that the hand-written set had been misleading.

## 5. Performance

**Shared-prefix scoring** (the `llm` backend reads a state once and reuses it for every question about it; `share_prefix` in the card). Sample inbox, short emails, seven questions each:

| Model | Without | With | Speed-up |
|---|---|---|---|
| `llm-qwen3-4b` | 1,494 ms per email | 727 ms per email | 2.1x |
| `llm-qwen3.5-4b` | | | 1.8x |

Exactness: on 84 real decisions the answers were identical with and without sharing, largest probability difference 0.008 (fp16 noise), checked on Qwen3-4B and on the hybrid Qwen3.5-4B. The gain grows with document length.

**Warm-up.** Each model, as it loads, answers three throwaway questions, on both the single and the shared-prefix path, so the first real request does not pay for first-use costs:

| Model | First request after start, without warm-up | With warm-up | Added to start-up |
|---|---|---|---|
| `nli-deberta-large` | 224 ms | 51 ms | 0.26 s |
| `llm-qwen3-4b` | 295 ms | 173 ms | 0.53 s |

`local-jev serve --no-warmup` skips it. The same warm-up runs for any model loaded later, on demand.

**Other measurements**

- Profiling the `llm` backend showed its time is spent in GPU work, not in Python or tokenisation.
- `nli-deberta-large` batch size 16 to 32 (in its card): 422 to 399 ms for 7 questions on one email.
- Earlier single-request figures: a 3-question request (3-option choice, 3-level score, yes/no, short state) took 64 ms on `nli-deberta-large` and 156 ms on `llm-qwen2.5-1.5b`.
- Qwen3.5-4B is about 3.5 times slower than Qwen3-4B per request on this Mac (651 against 177 ms median on JevBench): it is a hybrid model, and its linear-attention layers have no fast path in PyTorch on Apple GPUs. An MLX runtime would address this; local-jev does not have one.

**The portal at 50,000 items** (`--ui` only). Answers store their verdict (choice key, probability of yes, or expected score) in plain columns, so aggregating and filtering never decode JSON, and item bodies are decoded only for the page shown:

| Portal endpoint | Before | After |
|---|---|---|
| Item list | 567 ms | 28 ms |
| Search | 622 ms | 66 ms |
| Filtered list | 564 ms | 229 ms |
| Results summary | 554 ms | 253 ms |
| "Latest 1,000" view | 374 ms | 19 ms |

Output was byte-identical before and after, on a 50,000-item fixture and on a live portal (2 projects x 3 models). `/api/meta` answers in 1 to 4 ms.

## 6. Reproducing

**JevBench** (details in [training.md](training.md#independent-benchmark)):

```sh
git clone https://github.com/fstandhartinger/jevbench && cd jevbench
#   install the harness as its README describes
cat datasets/public/{easy,original,hard}.jsonl > public.jsonl

# with `local-jev serve` running on the default port; one run per model
python -m jevbench.cli run --tasks public.jsonl --adapter typesafe \
    --endpoint http://127.0.0.1:8765 --key-env '' --model llm-qwen3.5-4b \
    --results runs/llm-qwen3.5-4b.jsonl \
    --cap-usd 100000 --price-in-per-m 0 --price-out-per-m 0

# in local-jev: summarise the runs into the model cards
python evals/jevbench_cards.py --runs <path to jevbench>/runs --tasks <path to jevbench>/public.jsonl
```

The price and cap flags matter: by default the harness prices each request as if it were a hosted API and stops at $15 of estimated spend, which a long local run reaches.

**The 100 hand-written cases:**

```sh
local-jev serve &
python evals/compare/run_compare.py --systems llm-qwen3.5-4b llm-qwen3-4b llm-qwen3.5-2b llm-qwen2.5-1.5b nli-deberta-large
```

Add `jev` to `--systems` to include the hosted API; its key is read from `TYPESAFE_API_KEY`. Responses are cached under `evals/compare/results/` and the report is written to `evals/compare/report.html`; both are git-ignored. The report opens with the JevBench results read from the cards.

## 7. Caveats

- **Public items only, one machine.** JevBench's 303 held-out items were not run, and these results have not been submitted to or checked by Benchmark Heaven.
- **Jev's JevBench row is its published result**, not a local measurement; SemIf's likewise.
- **Jev's responses to the hand-written cases are not published in this repository.** They were obtained with the repository owner's key, cached for the report only, and are git-ignored. They are never used for training or calibration ([why](training.md#what-is-deliberately-not-done)).
- **Chosen on the test set.** Each Qwen card's prompt layout was chosen by measuring both layouts on the same 231 public items, so the `json` cards' results are mildly optimistic, in the same way the ensemble's 98% on the hand-written cases was.
- **Small, clean hand-written cases flatter everything.** On the messier sample inbox the entailment model over-triggers some yes/no questions (real sponsorship offers score 0.96 to 1.00, false alarms 0.5 to 0.7, so a threshold of 0.9 separates them), and Qwen3-4B rates plain notifications as a strong "sponsor fit".
- **Latency depends on hardware.** CUDA has not been measured. Measure on your own data and hardware before relying on any number here.
