# Related work

Several open projects appeared in the week after TypeSafe released Jev (15 September 2026), each rebuilding the "System One" pattern in the open: a state and typed questions in, a probability for every allowed answer out, no text generation. This page describes the ones we looked at, how each works, and how local-jev differs. It is a snapshot **as of 2026-09-21**; these projects change daily, so check their pages for current numbers. Figures quoted below are the projects' own unless stated otherwise, and were not re-measured by us.

None of these projects, nor local-jev, is affiliated with TypeSafe.

## At a glance

| Project | Model | How answers are read | Trained for decisions | Jev wire format (`/v1/systemone`) | Licence (code) |
|---|---|---|---|---|---|
| [decider](https://github.com/Mapika/decider) | Qwen3.5 2B / 0.8B / 35B-A3B | a learned head at one answer slot per question, over option label tokens | yes: ~95 public datasets + a local 27B teacher, then calibration-aware RL (2B v10) | yes; TypeSafe's SDK works against it | Apache-2.0 |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev) | Qwen3.5-4B, frozen (also 0.6B, 2B in the browser) | logits over lettered options at the final position | no | no (a research harness and CLI) | MIT |
| [Bespoke Nimble](https://github.com/bespokelabsai/nimble) ([weights](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)) | Qwen3.5-9B + LoRA | scores the allowed answer tokens directly | yes: 2,676 contrastively curated examples; states it did not distil from Jev | yes, via an SGLang-based server | see repository |
| [OpenJev (AlexWortega)](https://huggingface.co/AlexWortega/openjev) | Qwen3.5-4B (also 0.8B, 35B-A3B) | three-way NLI cross-encoder (entailment / neutral / contradiction) | yes: NLI, faithfulness and adversarial-NLI data | no (a transformers model) | MIT |
| [DiffusionGemma structured mode](https://github.com/vllm-project/vllm/pull/57250), served by [razorback16/openjev](https://github.com/razorback16/openjev) | DiffusionGemma 26B-A4B | one "read-only" denoising step over a seeded canvas; logprobs at each answer slot | no | the openjev server implements it | Apache-2.0 (openjev) |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Qwen3-0.6B + decision heads | shared scoring head over candidate paths; set attention for Choice | yes: expert gameplay in four games | no (`/api/evaluate`) | MIT |
| [Laya](https://github.com/NandhaKishorM/laya) | ModernBERT-large 421M; mmBERT-base 322M | encoder with typed decision heads, one forward pass | yes: RL against proper scoring rules (RLCD) | no (Python SDK) | Apache-2.0 |
| **local-jev** (this project) | Qwen3.5-4B (default), Qwen3-4B-Instruct, Qwen3.5-2B, Qwen2.5-1.5B-Instruct, all zero-shot; DeBERTa-v3-large 435M zero-shot | logits over lettered options (plain or JSON layout, per model); entailment log-odds per answer | no (one fine-tune of the entailment model was evaluated and not shipped) | yes; verified with the unmodified official SDK | MIT |

## The projects

### JevBench (the benchmark)

[JevBench](https://benchmarkheaven.com/jev-models) ([harness, MIT](https://github.com/fstandhartinger/jevbench)) is an independent benchmark for Jev-class systems by Benchmark Heaven. Version 1.2 has 534 decisions in four tiers (easy 72, standard 96, judge 146, hard 220); 231 are public, the rest are held out. The JevBench Score is the geometric mean of four axes weighted equally: Intelligence (weighted accuracy, hard tier 30%), Calibration (top-label ECE plus fidelity to exact gold distributions on the hard tier), Speed and Cost. At the time of writing its ranking is led by Jev 1.13.0 (75.4), SemIf (74.7), djev (74.3), openJev Verdict 1.4 (72.5) and Laya (70.1); frontier LLMs score highest on Intelligence but lower overall because of cost. Its `typesafe` adapter talks to any server that implements Jev's `/v1/systemone`, so a wire-compatible system such as local-jev can be run against the public items directly.

local-jev ran its built-in models on the 231 public items this way (2026-09-21, Apple M4 Max, one decision per request): `llm-qwen3.5-4b` 80.5%, `llm-qwen3-4b` 70.1%, `llm-qwen3.5-2b` 66.2%, `llm-qwen2.5-1.5b` 58.9%, `nli-deberta-large` 54.1%, against 86.6% published for Jev 1.13.0 and 81.0% for SemIf on the same items. These are accuracy figures on the public items only, not JevBench Scores, and have not been submitted to or verified by Benchmark Heaven. Method and caveats: [training.md](training.md#independent-benchmark).

### decider

A trained decision model by Mark Marosi. A request is rendered as text with one answer slot per question; a head reads the hidden state at each slot and projects it onto one label token per option (up to 255), so all questions come out of one pass. Score levels can each be judged in isolation. Training uses about 95 public datasets plus data written by a local Qwen3.5-27B teacher; the 2B v10 adds reinforcement learning with a proper-scoring reward. It ships an HTTP server in TypeSafe's wire format, a schema cache that precomputes fixed question sets, CUDA graphs and FP8. It reports results on its own 94-task set, on JevBench's public items and on Bespoke's public suite, where its 35B model reports a macro accuracy of 0.774 against 0.760 it quotes for Jev 1.13.0.

**How local-jev differs:** local-jev has not trained a decision model; it uses off-the-shelf models behind a backend interface, runs on a laptop (CPU or Apple GPU), and adds a project UI. decider is a trained model with a production-oriented CUDA serving path.

### SemIf (formerly OpenJev, TheoLeeCJ)

An open research baseline: a frozen Qwen3.5-4B is prompted with the state, a criterion and 2-16 lettered options, and one forward pass yields a softmax over the option letters. It includes prefix reuse for many questions over one state, an MLX backend for Apple silicon and a WebGPU browser demo. Evaluation covers 144 authored decisions, WANLI and a 102-row subset of TypeSafe's public evaluation cases, with committed raw outputs. It states explicitly that its probabilities are conditional on the offered options and not calibrated confidence.

**How local-jev differs:** local-jev's `llm` backend uses the same letter-logit technique, and has since adopted two of SemIf's ideas, credited in its code: SemIf's JSON prompt layout, which it uses for its Qwen3.5 models (on JevBench's public items local-jev's Qwen3.5-4B scores 80.5% with it, against SemIf's published 81.0% for the same model), and reading a state once and reusing that prefix across questions. local-jev adds the Jev HTTP API and SDK compatibility, per-primitive calibration, Score and Noul primitives, other backends and a UI. SemIf goes further on Apple-silicon speed (MLX; local-jev runs the same model in PyTorch, where it is several times slower on a Mac), in-browser inference and evaluation transparency.

### Bespoke Nimble (Bespoke Labs)

A LoRA adapter on Qwen3.5-9B that scores the allowed answer tokens directly, trained on 2,676 examples made by *contrastive curation*: change one fact in a context so the correct answer flips, and label with code-applied rules. The repository states it did not distil from Jev. Its most reusable contribution is a **public benchmark suite**: 13 human-labelled subsets (3,880 records) covering routing, multilingual routing, yes/no over a passage, answerability, paraphrase, entailment, moderation, guardrails, rubric rating and summary quality, rebuildable byte-for-byte, with Jev 1.13.0 measured on every subset. Prompts are limited to 2,048 tokens.

**How local-jev differs:** local-jev uses zero-shot models and has no curated training set of this kind; Nimble's suite is a benchmark local-jev can be measured on.

### OpenJev (AlexWortega)

A Qwen3.5-based cross-encoder with a three-way head (contradiction / entailment / neutral), trained on NLI, faithfulness, instruction-following and adversarial-NLI data, with 0.8B, 4B and 35B-A3B variants and image input on v2. It reports, for example, MNLI 0.91, ANLI round 3 0.63 and WANLI 0.77. It is a model rather than a server.

**How local-jev differs:** this is the same family of approach as local-jev's `nli` backend (answers posed as hypotheses to an entailment model) with a larger, newer backbone. Its model card lists a standard transformers interface.

### DiffusionGemma structured mode (vLLM) and razorback16/openjev

An open vLLM pull request by Matt Mastracci adds a Jev-like mode for DiffusionGemma, a 26B diffusion language model: the canvas is seeded with a template and answer placeholders, and a single read-only denoising step returns log-probabilities at every answer slot. [razorback16/openjev](https://github.com/razorback16/openjev) wraps this as a Jev-compatible decision server. The pull request reports about 8.7 requests per second single-stream and 54 at 32-way concurrency.

**How local-jev differs:** a different model class (diffusion) that needs a large GPU; local-jev targets smaller autoregressive and encoder models on commodity hardware.

### NanoJev

A 0.6B Qwen3 backbone with decision heads (sigmoid for booleans, a set-attention softmax over 2-255 candidates for Choice, an expected value over levels for Score), trained end to end on expert gameplay for ViZDoom, a maze and Snake, with a side-by-side browser replay against Jev and the untuned base model. Its compatibility audit documents where it does and does not yet match Jev's request format.

**How local-jev differs:** NanoJev focuses on decision-making for control tasks; local-jev on text classification-style questions behind the Jev API.

### Laya

An encoder-based system (ModernBERT-large 421M for English, mmBERT-base 322M for 100+ languages, plus a fine-tuned checkpoint) with typed decision heads, trained with reinforcement learning against strictly proper scoring rules, and a router that picks a checkpoint per request by script and language. It fits one temperature per question type and option count, ships workflow presets (routing, guardrails, moderation, triage) and a fine-tuning notebook, and reports 33 ms per question on a T4. It documents limits including a per-question option-token budget that makes large option sets (for example 77 intents) hard.

**How local-jev differs:** local-jev supports up to 255 options and structured state through its backends, is English-first, and ships no trained model (its one entailment fine-tune did not improve on JevBench); Laya is multilingual and trained.

## Where local-jev sits

What local-jev offers that is less common in this list:

- **A drop-in server for the published API**, checked with TypeSafe's unmodified Python SDK, including validation errors and the request-id header.
- **Backends as configuration.** A model card chooses between an entailment model, a small LLM or an ensemble, per server or per request; adding a model is a JSON file.
- **Calibration per primitive shipped with every model**, with Score temperatures fitted to the error of the expected level rather than likelihood.
- **An optional project portal** (`local-jev serve --ui`) for testing a question on one input and for running questions over a pile of items, with filtering, CSV export and a guide to choosing a model.
- **Runs on a laptop.** No CUDA-specific code paths; CPU and Apple GPU work.
- **Measured on an independent benchmark**, with results recorded in each model card and shown in the UI: 54.1% to 80.5% on JevBench's public items, depending on the model.

What it does not have yet: a model trained for the decision format, results on JevBench's held-out items or on Bespoke's public suite, a fast runtime for Qwen3.5 on Apple silicon (such as MLX), and schema caching. Its best model trails Jev mostly on JevBench's hard tier (69 against 81 of 111 public items).
