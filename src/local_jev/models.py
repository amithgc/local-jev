"""Model cards: what a model is called, which backend runs it, and how it was calibrated.

Two places are searched, the second winning on a name clash:

1. ``local_jev/cards/*.json``      built-in models that ship with the package
2. ``models/cards/*.json``         your own cards: other Hugging Face models, fine-tuned models, recalibrated built-ins

A card is a small JSON file. ``name``, ``description``, ``release_date``, ``backend``, ``license``,
``calibration`` and ``priority`` are understood everywhere; every other key is passed to the backend as
an option (``hf_model``, ``context_tokens``, ``batch_size``, ``device``, ``members``, ``weights``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("LOCAL_JEV_HOME", Path(__file__).resolve().parents[2]))
MODELS_DIR = Path(os.environ.get("LOCAL_JEV_MODELS", ROOT / "models"))
USER_CARDS_DIR = MODELS_DIR / "cards"
BUILTIN_CARDS_DIR = Path(__file__).parent / "cards"
ALIASES = ("jev-latest", "jev-preview", "local-jev-latest")
CORE_KEYS = {"name", "description", "release_date", "backend", "license", "calibration", "priority", "evaluation"}


@dataclass
class ModelCard:
    name: str
    description: str = ""
    release_date: str = "2026-09-20"
    backend: str = "nli"
    license: str = ""
    calibration: dict | None = None
    priority: int = 10                 # which model the server starts on: the highest-priority one that is ready
    evaluation: dict | None = None     # measured accuracy and speed, shown in the UI's model picker (see evals/)
    options: dict = field(default_factory=dict)
    source: str = "built-in"

    @property
    def memory_gb(self) -> float:
        """Memory the weights take once loaded: the card's ``size.memory_gb``, else a cautious guess by backend."""
        size = self.options.get("size") or {}
        if isinstance(size, dict) and size.get("memory_gb"):
            return float(size["memory_gb"])
        return {"ensemble": 0.0, "nli": 1.0}.get(self.backend, 8.0)      # an ensemble's members are counted on their own

    @property
    def temperature(self) -> dict:
        return (self.calibration or {}).get("temperature") or {}

    def unavailable(self) -> str | None:
        """Why this model cannot be loaded right now, or None if it can."""
        if self.backend == "ensemble":
            return None                                   # its members are checked when it loads
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return "needs PyTorch and transformers: pip install torch transformers"
        return None


def on_disk(card: ModelCard, cards: dict) -> bool:
    """True if loading this model will not start a download. The default model is never one that would."""
    if card.backend == "ensemble":
        return all(m in cards and on_disk(cards[m], cards) and not cards[m].unavailable() for m in card.options.get("members", []))
    repo = str(card.options.get("hf_model", ""))
    if Path(repo).is_dir():                       # a fine-tuned model saved locally (training/train_nli.py)
        return (Path(repo) / "config.json").exists()
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:
        return False


def _read(path: Path, source: str) -> ModelCard | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    core = {k: data[k] for k in CORE_KEYS if k in data}
    core.setdefault("name", path.stem)
    return ModelCard(**core, options={k: v for k, v in data.items() if k not in CORE_KEYS}, source=source)


def discover() -> dict[str, ModelCard]:
    found: dict[str, ModelCard] = {}
    for folder, source in ((BUILTIN_CARDS_DIR, "built-in"), (USER_CARDS_DIR, "user")):
        if folder.is_dir():
            for path in sorted(folder.glob("*.json")):
                card = _read(path, source)
                if card:
                    found[card.name] = card
    return found


FIRST_RUN_MODEL = "nli-deberta-large"       # the smallest download (about 0.9 GB); used when nothing is on disk yet


def default_model(cards: dict[str, ModelCard]) -> str:
    """The highest-priority model that can be loaded right now without downloading anything.

    The server therefore never starts a multi-gigabyte download on its own. Fetch a more accurate model
    once by asking for it (``--model llm-qwen3.5-4b``) and it becomes the default from then on. On a
    fresh machine nothing is on disk, so the smallest model is fetched. Ties go to the newest name.
    """
    ready = [c for c in cards.values() if not c.unavailable() and on_disk(c, cards)]
    if ready:
        return max(ready, key=lambda c: (c.priority, c.name)).name
    return FIRST_RUN_MODEL if FIRST_RUN_MODEL in cards else next(iter(cards))
