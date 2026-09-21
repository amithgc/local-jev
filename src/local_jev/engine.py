"""The System One engine: questions in, Jev-shaped answers out, whichever backend is underneath."""
from __future__ import annotations

import gc
import os

import numpy as np

from . import primitives
from .backends import DEVICE_LOCK, Backend, Item, load_backend
from .models import ALIASES, ModelCard, default_model, discover
from .questions import Question, QuestionError, parse_question

MAX_LOADED = 4          # a hard cap on loaded backends, on top of the memory budget


def memory_budget_gb() -> float:
    """How much memory loaded models may use: ``LOCAL_JEV_MEMORY_GB``, else half the machine's RAM.

    Half leaves room for the operating system, activations and the key-value cache of long prompts. On a
    36 GB Mac that is 18 GB: two 8-9 GB models, or one plus several small ones.
    """
    configured = os.environ.get("LOCAL_JEV_MEMORY_GB")
    if configured:
        return float(configured)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9 / 2
    except (ValueError, OSError, AttributeError):
        return 16.0


def calibrate(probs, temperature: float = 1.0) -> np.ndarray:
    """Temperature scaling in log space. It cannot change which answer wins; for a yes/no it is sigmoid(logit / T)."""
    z = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)) / max(float(temperature), 1e-6)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def to_answer(question: Question, probs) -> dict:
    if question.type == "choice":
        return primitives.choice_answer(list(question.keys), probs)
    if question.type == "score":
        return primitives.score_answer(list(question.legend), probs)
    return primitives.noul_answer(probs[0])


class Engine:
    def __init__(self, default: str | None = None, warmup: bool = True):
        self.warmup = warmup                     # warm every model as it loads, so no first request pays for GPU setup
        self.models: dict[str, ModelCard] = discover()
        self.default_model = default or os.environ.get("LOCAL_JEV_MODEL") or default_model(self.models)
        if self.default_model not in self.models:
            raise ValueError(f"unknown model {self.default_model!r}; available: {', '.join(self.models)}")
        self._backends: dict[str, Backend] = {}

    def resolve(self, name: str | None) -> str | None:
        if not name or name in ALIASES:
            return self.default_model
        return name if name in self.models else None

    def backend(self, name: str) -> Backend:
        # Loading, evicting and freeing all touch the accelerator, so all of it happens under DEVICE_LOCK:
        # a model released while another is computing is freed mid-flight on the GPU.
        with DEVICE_LOCK:
            if name in self._backends:
                self._backends[name] = self._backends.pop(name)             # mark as most recently used
                return self._backends[name]
            card = self.models[name]
            problem = card.unavailable()
            if problem:
                raise RuntimeError(f"model {name!r} is not available: {problem}")
            self._make_room(name)
            backend = load_backend(card, resolve=self._member)
            if self.warmup:
                backend.warmup()
            self._backends[name] = backend
            return backend

    def _needs(self, name: str) -> float:
        """Memory ``name`` would add: its own weights, plus any ensemble members not loaded yet."""
        card = self.models[name]
        if card.backend == "ensemble":
            return sum(self.models[m].memory_gb for m in card.options.get("members", [])
                       if m in self.models and m not in self._backends)
        return card.memory_gb

    def loaded_gb(self) -> float:
        return sum(self.models[n].memory_gb for n in self._backends)

    def _make_room(self, name: str) -> None:
        """Unload least recently used models until ``name`` fits the memory budget (always allowing one model)."""
        budget, need, evicted = memory_budget_gb(), self._needs(name), False
        keep = set(self.models[name].options.get("members", [])) if self.models[name].backend == "ensemble" else set()
        while self._backends and (self.loaded_gb() + need > budget or len(self._backends) >= MAX_LOADED):
            victim = next((n for n in self._backends if n not in keep), None)
            if victim is None:
                break
            self._evict(victim)
            evicted = True
        if evicted:
            gc.collect()

    def _evict(self, name: str) -> None:
        """Unload a model, and any loaded ensemble built on it (it would otherwise keep the weights alive)."""
        self._backends.pop(name, None)
        for other in [n for n in self._backends if self.models[n].backend == "ensemble"]:
            if name in self.models[other].options.get("members", []):
                self._backends.pop(other, None)

    def _member(self, name: str):
        """For ensembles: a member's backend and its calibration. Called while DEVICE_LOCK is held."""
        if name not in self.models:
            raise ValueError(f"ensemble member {name!r} is not a known model")
        if self.models[name].backend == "ensemble":
            raise ValueError("an ensemble cannot contain another ensemble")
        if name not in self._backends:
            problem = self.models[name].unavailable()
            if problem:
                raise RuntimeError(f"ensemble member {name!r} is not available: {problem}")
            self._backends[name] = load_backend(self.models[name])       # room was made for it by _make_room
        return self._backends[name], self.models[name].temperature

    @staticmethod
    def question(wire) -> Question:
        return parse_question(wire)

    def answer_many(self, model: str, states: list, questions: dict[str, Question], truncate: bool = False,
                    max_state_tokens: int | None = None) -> tuple[list[dict], list[dict]]:
        """Evaluate every question against every state. Returns (answers, stats), one entry per state."""
        temperature = self.models[model].temperature
        items = [Item(q, state) for state in states for q in questions.values()]
        scored = self.backend(model).score(items, truncate, max_state_tokens) if items else []
        answers, stats, cursor = [], [], 0
        for _ in states:
            row, tokens, truncated = {}, 0, False
            for qid, q in questions.items():
                result = scored[cursor]
                row[qid] = to_answer(q, calibrate(result.probs, temperature.get(q.type, 1.0)))
                tokens += result.tokens
                truncated |= result.truncated
                cursor += 1
            answers.append(row)
            stats.append({"input_tokens": tokens, "output_tokens": len(questions), "truncated": truncated})
        return answers, stats

    def system_one(self, state, questions: dict[str, dict], model: str | None = None, truncate: bool = True) -> dict:
        """One request of the Jev wire format.

        A state longer than the model's window is shortened (head and tail kept) rather than refused: Jev reads
        up to 32k tokens, and a small model that refused everything past 512 would fail requests Jev answers.
        ``truncated`` in the result says when that happened; the HTTP layer surfaces it as a header."""
        name = self.resolve(model)
        if name is None:
            raise KeyError(model)
        parsed = {}
        for qid, wire in questions.items():
            try:
                parsed[qid] = parse_question(wire)
            except QuestionError as err:
                raise QuestionError(err.msg, ("questions", qid) + err.loc) from None
        answers, stats = self.answer_many(name, [state], parsed, truncate=truncate)
        return {"model": name, "answers": answers[0], "truncated": stats[0]["truncated"],
                "usage": {k: stats[0][k] for k in ("input_tokens", "output_tokens")}}
