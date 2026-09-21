"""Shared helpers for the PyTorch / Hugging Face backends."""
from __future__ import annotations

HINT = "local-jev needs PyTorch and transformers: pip install torch transformers"


def require_torch():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as err:
        raise RuntimeError(HINT) from err
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    return torch


def load_pretrained(factory, repo: str, dtype=None, **kwargs):
    """``from_pretrained`` across transformers versions: the dtype argument was renamed from torch_dtype to dtype."""
    if dtype is None:
        return factory.from_pretrained(repo, **kwargs)
    try:
        return factory.from_pretrained(repo, dtype=dtype, **kwargs)
    except TypeError:
        return factory.from_pretrained(repo, torch_dtype=dtype, **kwargs)


def pick_device(torch, requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_text(tokenizer, text: str, budget: int, truncate: bool, what: str) -> tuple[str, bool]:
    """Keep the head and tail of an over-long state: the middle is usually the least informative part."""
    from ..questions import ContextOverflow
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text, False
    if not truncate:
        raise ContextOverflow(f"State is {len(ids)} tokens; {what} leaves room for {budget}")
    keep = max(budget - 8, 16)
    head = int(keep * 0.75)
    return tokenizer.decode(ids[:head]) + "\n[...]\n" + tokenizer.decode(ids[-(keep - head):]), True
