# How local-jev got here

local-jev today runs on an entailment model and small instruction-tuned language models. It did not start there. It started as an attempt to build the whole thing on one very small model, and the reasons that attempt was abandoned are worth keeping, because most of them are not specific to that model.

This page is the only place the old design is described. Nothing on it is needed to use or change the current code.

## The starting point: Needle 3

Needle 3 is a 121M-parameter on-device model from Cactus Compute, built for one job: read a list of tool descriptions, read a request, and emit a JSON tool call. It ships with a native engine under 1 MB and runs in tens of milliseconds on a phone. A System One server needs fast, closed-set decisions rather than prose, so a tiny model that already picks one item from a described list looked like a natural fit.

## Why its own engine could not be used

Two things ruled out the prebuilt engine on the first day.

- **It exposes no logits.** Its interface returns one call and one confidence number. A System One answer is a probability for *every* option, and there was no way to ask for that.
- **Its repair step rewrites answers.** After decoding, the engine "repairs" arguments against the request text; one rule moves an enum value to the option the request names. On a support message containing the phrase "I'm losing sales", the model's own reasoning chose `technical`, and the engine then returned `sales` with confidence 0.9955. That rule is right for device control and wrong for classification.

So the project loaded the full-precision checkpoint and ran the model directly, reading next-token logits over the candidate answers. That part worked, and the idea survives: every backend in local-jev still reads probabilities off a model instead of letting it generate.

## Three ways of asking, and the lesson

Untuned, the model was close to guessing, so it had to be fine-tuned, and what it could learn turned out to depend entirely on how a question was posed.

| Design | What the model was asked to do | What was measured |
|---|---|---|
| **Labels** | One tool with options lettered `A`, `B`, `C`, each letter explained in a description; emit the letter. | After two runs and about 20,000 training examples, yes/no questions were being learned, but category and score questions were **exactly at chance**. Calibration improved while accuracy did not: the model had learned a uniform prior and nothing else. |
| **Pairwise** | One yes/no per option: "is this candidate the right one?" | A diagnostic ended at a loss of 0.6935, which is ln 2. A coin flip. It also exposed why earlier yes/no results had looked good: sentiment depends on the text alone, so the model had been answering without reading the question. |
| **Tools** | Every option becomes its own tool, named after the option; emit the tool's name. | Above chance with no training at all, and it learned quickly. Scores only worked once each level's tool was named after the level's own words rather than `level_3`. |

The difference between the first design and the third is one hop of reasoning. With letters, the model has to decide the input is a technical problem and then remember that *technical* was called `B`. With tools, it writes the concept's own name.

> A small model can emit the name of a concept, but it cannot bind an arbitrary symbol to it.

This is a lesson about model **capacity**, not about letters. local-jev's `llm` backend letters its options and reads the logits over `A`, `B`, `C`, exactly the design that failed here, and it works zero-shot, because a 1.5B+ instruction-tuned model can do that binding. The general form of the lesson is: pose the question in the form the model you have is actually able to answer, and when a diagnostic succeeds, check that the model could not have succeeded without reading the part of the prompt you care about.

## What two fine-tuned adapters achieved

Both used the tools design, LoRA on a frozen base, and this project's gold-labelled data (public datasets plus two small hand-written families; never the output of another model).

| | `local-jev-1` | `local-jev-2` |
|---|---|---|
| Training | Apple M4 Max CPU, 71k examples, one epoch plus a short corrective pass | RTX 3090 Ti, 258k examples, three epochs |
| Trained task families, unseen rows (macro accuracy) | 68% | 81% |
| Task families never trained on (macro accuracy) | 55% | 60% |
| 100-case comparator | 56% | 67% |

Fine-tuning clearly worked: from chance to useful on the tasks it saw, with well-calibrated confidence. What it did not do is generalise far. Accuracy on unseen kinds of question, which is the only case that matters to someone bringing their own categories, moved five points for 3.6 times the data.

## The decision

The comparator put every candidate on the same 100 hand-written cases:

| System | Accuracy |
|---|---|
| `local-jev-1`, fine-tuned 121M | 56% |
| `local-jev-2`, fine-tuned 121M, more data, GPU | 67% |
| DeBERTa-v3-large entailment model, 435M, **zero-shot** | 95% |
| Qwen3-4B-Instruct, **zero-shot** | 96% |
| those two pooled | 98% |
| the hosted Jev API | 100% |

More training gave Needle steady but diminishing gains. The large step came from a larger base model that already knew how to judge a sentence against a text, or how to answer a multiple-choice question, with no training from this project at all. The repository owner judged the Needle models not effective enough to keep, and the Needle backend, its compiler and scorer, its training script and its adapters were removed.

What carried over is everything that was not about Needle: the backend interface, question validation, calibration, the answer maths, the API, the UI, the data pipeline, the evaluation harness and the comparator. The training data now feeds an optional fine-tune of the entailment model instead ([training.md](training.md)). The current design is described in [architecture.md](architecture.md).

Needle 3 is a good model at the job it was built for. Judging arbitrary text against arbitrary descriptions, zero-shot, at 121M parameters, was not that job.

## 2026-09-21: measuring against an independent benchmark

Until this point every model decision in this project, including the one above, rested on the 100 hand-written comparator cases. On 2026-09-21 the built-in models were run on [JevBench](https://github.com/fstandhartinger/jevbench), an independent benchmark for Jev-class systems by Benchmark Heaven: its 231 public items, through local-jev's own API with the benchmark's `typesafe` adapter.

**Our own test set had been far too easy.** The entailment model went from 95% on the hand-written cases to 54.1% on JevBench. The pooled entailment-plus-Qwen3-4B ensemble, the best local result on the hand-written cases at 98%, scored 68.8%: below Qwen3-4B on its own (70.1%), and slower. Its built-in card was removed; the ensemble backend stays for anyone who wants to pool models of comparable strength. Another team's fine-tuned DeBERTa scores 52.4% on the same items, in line with our untuned one. The hand-written cases remain, as a regression check. The ranking in the table above (small model far behind, larger models ahead) still holds; the absolute numbers in it were optimistic.

**The prompt turned out to be part of the model.** Qwen3.5-4B scored 73.2% with local-jev's plain prompt and 80.5% with the JSON layout SemIf uses, reproducing SemIf's own 81.0% for the same model. With the plain layout it was over-cautious, answering "no" where a policy clearly permitted the action and "other" where a request was clear. Qwen3-4B went the other way (70.1% plain, 67.5% JSON), and Qwen3.5-2B gained two points from JSON. Since then each card names its layout, and a new model is measured both ways.

**Qwen3.5-4B became the default.** At 80.5% it is within two items of Jev (86.6%, published) on the easy and standard tiers and behind it on the hard tier (69 against 81 of 111). It is also the slowest built-in model on a Mac, because its hybrid linear-attention layers have no fast path in PyTorch's Apple-GPU backend; an MLX runtime is the obvious next step and does not exist yet.

The lesson is the same one the Needle work taught in a different form: measure on a benchmark you did not write. The layout choice for each Qwen card was itself made on JevBench's public items, so the new numbers carry a mild version of the same optimism, and the benchmark's 303 held-out items have not been run.
