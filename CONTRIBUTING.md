# Contributing

Thanks for looking. local-jev is a small project with a narrow goal: a local server that behaves like a System One API and is honest about how well it does so. Contributions that serve that goal are welcome.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest          # 36 tests need no model weights and download none; 3 more are skipped without a live server
LOCAL_JEV_URL=http://127.0.0.1:8765 .venv/bin/python -m pytest     # all 39, with `local-jev serve` running
```

`models/`, `data/`, `.venv*/` and `secrets.txt` are git-ignored. Never commit weights, generated datasets or credentials.

PyTorch and `transformers` are part of the base install. Model weights download from Hugging Face the first time a model is used (0.87 GB for the entailment model, 3.09 to 9.32 GB for the LLMs). For training on an NVIDIA GPU, see [training/README-gpu.md](training/README-gpu.md).

## What makes a useful contribution

- **New training and evaluation tasks.** A fine-tuned model generalises in proportion to the variety of tasks it sees, and every model is better understood the more kinds of task it is measured on. A task is one entry in [training/tasks.py](training/tasks.py): a public dataset with human labels, a function from row to state, a function from row to label, and several paraphrased instructions. Prefer tasks unlike the ones already there (different domain, different state shape, a genuine ordered scale). Check the dataset's licence allows it.
- **Evaluation sets that look like real use.** Small, hand-checked sets of questions a user would actually ask, with states a user would actually send.
- **Failure reports with a reproduction.** The state, the question, the answer you got and the answer you expected. These become evaluation cases.
- **New backends and new built-in models** (see below), with JevBench numbers.
- **Faster runtimes.** Qwen3.5 is several times slower than its size suggests on Apple GPUs in PyTorch, because its linear-attention layers have no fast path there. An MLX runtime for the `llm` backend would be very welcome.
- **Performance work** on any backend, measured before and after.
- **Ports and packaging**: Linux and CUDA notes, a container, Windows.

## Adding a backend

A backend answers one question: *for this `Question` and this state, how likely is each answer?* Everything else is shared. The full recipe is in [docs/architecture.md](docs/architecture.md#adding-a-backend); in short:

1. Subclass `Backend` in `src/local_jev/backends/<kind>.py` and implement `score(items, truncate, max_state_tokens)`, returning one probability per `question.answers` entry, with no temperature applied, and hold `self.lock` (the process-wide `DEVICE_LOCK`) around every model call: two models on one Apple GPU crash the process without it. Do not re-validate questions (that is `questions.py`) and do not apply temperature (that is the engine).
2. Register it in `load_backend()` and `BACKENDS` in `backends/__init__.py`, with lazy imports. If it needs a dependency beyond PyTorch and `transformers`, make it an optional extra in `pyproject.toml` and have `ModelCard.unavailable()` say what to install.
3. Keep the question-to-prompt logic in pure functions and test those without a model, as `tests/test_backends.py` does for `hypotheses()` and `options_of()`. Tests must not download weights or need a GPU.
4. Calibrate it (`training/calibrate.py --model <name>`) and report numbers: JevBench's public items through the API, recorded with `evals/jevbench_cards.py` ([how](docs/training.md#independent-benchmark)), then `evals/run_eval.py --model <name>` and a comparator run as regression checks.

## Adding a built-in model card

A card under `src/local_jev/cards/` ships with the package, so the bar is higher than for a card in your own `models/cards/`:

- **State the licence in the card** (`"license": "Apache-2.0 (org/repo)"`), and only add models whose licence permits this use and redistribution of a pointer to them. Cards without a licence will not be merged.
- The model must be publicly downloadable without credentials, and the card's `description` must say what it is, its size, and that it is used zero-shot if it is.
- Include `context_tokens`, a `priority` that places it sensibly among the existing models by its JevBench accuracy (it decides which model the server starts on), a `calibration` block fitted with `training/calibrate.py`, and `batch_size` if the default is too large for a 16 GB machine.
- Include its JevBench result as an `evaluation` block written by `evals/jevbench_cards.py`, and a `size` block (parameters, download measured on disk, memory estimated from the weights). The UI's model picker and Models page read both.
- Write `summary`, `use_when` and `avoid_when` in plain language, grounded in the measured numbers; the Models page shows them to people choosing a model.
- For an `llm` card, measure both prompt layouts (`"prompt": "plain"` and `"json"`) and ship the better one; the difference was seven points for Qwen3.5-4B and went the other way for Qwen3-4B.
- Say in the pull request which hardware the numbers come from.

## Ground rules

1. **Jev's output is never a training signal.** Not as training data, not as soft targets, not as a calibration target, not pasted into a test or a dataset. See [docs/training.md](docs/training.md#what-is-deliberately-not-done). The comparator may call the hosted API for a side-by-side report when a contributor supplies their own key; its cached responses stay out of the repository (`evals/compare/results/` is git-ignored) and out of every training, calibration and data-generation path. Pull requests that break this will be closed. Labels generated by other language models need discussion first; open an issue.
2. **Training and inference share one rendering.** A backend's question-to-prompt functions (`hypotheses()`, `options_of()`) are the single source of truth; a trainer must render its examples through them, as `train_nli.py` does. If you change the rendering, say so in the pull request: fine-tuned models and fitted temperatures for that backend are invalidated.
3. **Keep the wire format exact.** The `/v1` endpoints must stay compatible with the official `typesafe-sdk`. Extra fields need a strong reason.
4. **Measure, then claim.** Quality claims come with numbers from an independent benchmark (JevBench's public items) and, where a model was trained, from the evaluation harness on held-out tasks. The hand-written comparator cases are a regression check; they proved far too easy to rank models. Speed claims come with the hardware, device and dtype. Do not report a number for a model that has not been run.
5. **Offline by default.** Nothing in the serving path may make a network call other than the one-time weight download, and the server must never start a multi-gigabyte download on its own.
6. **Model code stays in the backends.** PyTorch and `transformers` are imported only under `src/local_jev/backends/` and in `training/`, lazily. The engine, API, UI, store and evaluation code must not know which backend is running.

## Code style

Match the surrounding code: small modules, docstrings that explain *why*, comments only where the reason is not obvious from the code. Python 3.10+ syntax. No new dependencies in the serving path without discussion; the UI has no build step and should keep none.

## Documentation

Diagrams live in `docs/assets/` as hand-written SVG with a PNG rendered beside each. Edit the SVG, then run `docs/assets/build.sh` (needs `uv` and cairo). Two rendering quirks to know about: arrow characters in text do not render, so arrows are drawn as paths with markers; and runs of spaces collapse, so align with `x` coordinates rather than spaces.

## Licence

The project's licence has not been chosen yet. By contributing you agree that your contribution may be distributed under whatever open-source licence the repository owner selects.
