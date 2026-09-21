"""Backends: the models that actually answer questions.

A backend takes (question, state) pairs and returns one probability vector per pair, aligned
with ``Question.answers`` and *uncalibrated*. Everything around that -- validation, calibration,
Jev-shaped answers, the API, the UI, evaluation -- is shared and lives outside this package.

Adding a backend: subclass ``Backend``, register it in ``BACKENDS`` below, and describe a model
that uses it with a card (see ``local_jev/cards/`` and docs/architecture.md).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..questions import Question


@dataclass
class Item:
    question: Question
    state: Any


@dataclass
class Scored:
    probs: np.ndarray            # one probability per Question.answers entry, summing to 1
    tokens: int = 0              # input tokens the model read for this item
    truncated: bool = False      # the state was shortened to fit


# One lock for all model work in the process. Every backend here runs on the same accelerator, and PyTorch's
# Apple-GPU (MPS) backend is not safe when two threads use it at once: two models answering concurrent requests
# crashed the server with "failed assertion _status < MTLCommandBufferStatusCommitted". Loading weights onto the
# device counts as model work too. Re-entrant, because an ensemble scores through its members.
DEVICE_LOCK = threading.RLock()


class Backend(ABC):
    """One loaded model. ``score`` implementations take ``self.lock`` around their model calls."""

    kind: str = ""
    context_tokens: int = 0

    def __init__(self):
        self.lock = DEVICE_LOCK

    @abstractmethod
    def score(self, items: list[Item], truncate: bool = False, max_state_tokens: int | None = None) -> list[Scored]:
        """Score every item. Raise ``ContextOverflow`` if an item does not fit and ``truncate`` is false."""

    def warmup(self) -> float:
        """Answer a few throwaway questions so one-off GPU setup (kernel selection, allocator growth) happens at
        load time instead of on the first real request. Returns seconds spent."""
        import time
        from ..questions import parse_question
        started = time.perf_counter()
        state = "Hi, I was charged twice for my subscription this month. Could you refund one of the charges?"
        questions = [parse_question({"type": "noul", "instructions": "The customer asks for a refund."}),
                     parse_question({"type": "choice", "instructions": "Which team should handle this?",
                                     "criteria": {"billing": "Payments and refunds", "technical": "Bugs", "sales": "Pricing"}}),
                     parse_question({"type": "score", "instructions": "How frustrated is the customer?",
                                     "criteria": ["Calm", "Annoyed", "Angry"]})]
        self.score([Item(questions[0], state)])                       # one question on its own
        self.score([Item(q, state) for q in questions])               # several questions on one state
        return time.perf_counter() - started


def load_backend(card, **options) -> Backend:
    """Instantiate the backend a model card asks for. Imports are lazy, so listing models never loads PyTorch."""
    kind = card.backend
    if kind == "nli":
        from .nli import NLIBackend
        return NLIBackend(card, **options)
    if kind == "llm":
        from .llm import LLMBackend
        return LLMBackend(card, **options)
    if kind == "ensemble":
        from .ensemble import EnsembleBackend
        return EnsembleBackend(card, **options)
    raise ValueError(f"model {card.name!r} asks for unknown backend {kind!r}; known: {', '.join(BACKENDS)}")


BACKENDS = {
    "nli": "An entailment cross-encoder (e.g. DeBERTa-v3 zero-shot): each answer becomes a hypothesis.",
    "llm": "A small instruction-tuned language model read at the answer token (e.g. Qwen 1.5-4B).",
    "ensemble": "Several registry models pooled (weighted geometric mean of their calibrated probabilities). Costs the sum of their latencies.",
}
