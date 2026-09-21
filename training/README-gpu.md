# Fine-tuning the entailment model on an NVIDIA GPU

`training/train_nli.py` fine-tunes the `nli` backend's model (DeBERTa-v3-large zero-shot, 435M) on this project's gold-labelled data. It is ordinary PyTorch, so it needs none of the JAX/XLA memory settings an earlier version of this guide described.

**Status: not yet shown to help; no fine-tuned model has been evaluated.** The untuned model scores 54.1% on JevBench's public items ([results](../docs/results.md)), strong on clear-cut classification and weak on hard questions. Full fine-tuning can *erode* the zero-shot generality it does have, so treat a run as an experiment: judge the result on the held-out tasks and on JevBench before using it (step 4).

Requirements: Linux or WSL2, a CUDA 12 driver, Python 3.10+, [uv](https://docs.astral.sh/uv/). A 24 GB card is comfortable; 12 GB should work with a smaller `--pair-budget`.

## 0. Windows: WSL2

1. Update the **Windows** NVIDIA driver. Do not install a Linux NVIDIA driver inside WSL.
2. In an administrator PowerShell: `wsl --install -d Ubuntu`, reboot, open Ubuntu. `nvidia-smi` inside Ubuntu should list the card.
3. `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new shell.

## 1. Set up

```sh
git clone <this repo> ~/local-jev && cd ~/local-jev
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"          # dev: the dataset tools build_data.py needs
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"    # expect: True NVIDIA GeForce RTX 3090 Ti
```

If that prints `False`, stop and fix the CUDA install: the script would fall back to the CPU and take days.

## 2. Data

If you still have `data/train.jsonl` from an earlier build, reuse it: the format has not changed. Otherwise:

```sh
.venv/bin/python training/build_data.py --per-task 5000 --eval-per-task 300 --email-per-email 40
```

## 3. Smoke test, then the run

```sh
# two minutes: does it train, and does the loss stay finite?
.venv/bin/python training/train_nli.py --out /tmp/nli-smoke --name nli-smoke --examples 400 --val-examples 200 --max-steps 40 --eval-every 20
rm -f models/cards/nli-smoke.json
```

Look for `device: cuda`, a loss that is a number (not `nan`) and falls, and a step time well under a second.

```sh
nohup .venv/bin/python training/train_nli.py --out models/hf/nli-deberta-large-ft1 \
    --examples 40000 --epochs 1 --lr 5e-6 --pair-budget 32 > nli-ft1.log 2>&1 &
tail -f nli-ft1.log
```

The defaults are deliberately gentle (one pass over 40,000 examples at a low learning rate). Validation accuracy is printed per primitive every 500 steps, against the untuned model's numbers at step 0; the model is saved at each of those points. If `choice` or `noul` validation accuracy *falls* below its step-0 value, stop: that is the generality eroding. Out of memory: lower `--pair-budget` to 16.

## 4. Judge it before using it

Copy `models/hf/nli-deberta-large-ft1/` (about 1.7 GB) and `models/cards/nli-deberta-large-ft1.json` to the machine that serves local-jev, fix the `hf_model` path in the card if the folder moved, then calibrate it, run JevBench's public items against it ([how](../docs/training.md#independent-benchmark)), and use the comparator and the held-out tasks as regression checks:

```sh
.venv/bin/python training/calibrate.py --model nli-deberta-large-ft1
.venv/bin/local-jev serve &
.venv/bin/python evals/compare/run_compare.py --systems nli-deberta-large nli-deberta-large-ft1 llm-qwen3.5-4b
.venv/bin/python evals/run_eval.py --model nli-deberta-large --n 100          # before
.venv/bin/python evals/run_eval.py --model nli-deberta-large-ft1 --n 100      # after
```

Keep it only if it beats the untuned model on JevBench **and** on the held-out tasks. The card gives it priority 75: above the untuned entailment model (50) and `llm-qwen3.5-2b` (70), below the two 4B LLMs. It becomes the server's default only on a machine that has not fetched those; delete the card to undo that.
