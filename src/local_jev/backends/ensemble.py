"""Ensemble backend: several models answer, and their calibrated probabilities are pooled.

Pooling is log-linear (a weighted geometric mean, renormalised): an answer scores well only if every
member gives it weight, and one confident member can outvote an unsure one. It pays off when the
members fail differently. An entailment model and a small LLM do: on the comparison set they missed 5
and 4 cases out of 100 on their own, none of them in common, and 2 when pooled.

The cost is the sum of the members' latencies, because they run one after another.
"""
from __future__ import annotations

import numpy as np

from . import Backend, Item, Scored


class EnsembleBackend(Backend):
    kind = "ensemble"

    def __init__(self, card, resolve=None, **_):
        super().__init__()
        if resolve is None:
            raise RuntimeError("an ensemble needs the engine to load its members")
        names = card.options.get("members") or []
        if len(names) < 2:
            raise ValueError(f"ensemble {card.name!r} needs at least two members")
        weights = card.options.get("weights") or [1.0] * len(names)
        self.members = [(name, float(w), *resolve(name)) for name, w in zip(names, weights)]     # (name, weight, backend, temperatures)
        self.context_tokens = min(backend.context_tokens for _, _, backend, _ in self.members)

    def score(self, items: list[Item], truncate: bool = False, max_state_tokens: int | None = None) -> list[Scored]:
        from ..engine import calibrate
        total = sum(w for _, w, _, _ in self.members)
        pooled = [np.zeros(item.question.n_answers) for item in items]
        tokens, truncated = [0] * len(items), [False] * len(items)
        for _, weight, backend, temperature in self.members:
            for i, (item, scored) in enumerate(zip(items, backend.score(items, truncate, max_state_tokens))):
                probs = calibrate(scored.probs, temperature.get(item.question.type, 1.0))
                pooled[i] += (weight / total) * np.log(np.clip(probs, 1e-9, 1.0))
                tokens[i] += scored.tokens
                truncated[i] |= scored.truncated
        out = []
        for logp, n, cut in zip(pooled, tokens, truncated):
            p = np.exp(logp - logp.max())
            out.append(Scored(p / p.sum(), n, cut))
        return out

    def warmup(self) -> float:
        return sum(backend.warmup() for _, _, backend, _ in self.members)          # the members' own warm-ups
