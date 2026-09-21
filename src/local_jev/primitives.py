"""The maths behind the three System One primitives.

Everything here is pure: probabilities in, Jev-shaped answer dicts out.
"""
from __future__ import annotations

import numpy as np

ROUND = 4


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def confidence(probs) -> float:
    """How peaked a distribution is: 1.0 with all mass on one option, 0.0 when uniform.

    The n-option generalisation of the three-option formula in Jev's docs,
    (3 * p_max - 1) / 2.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    return float(max(0.0, (n * float(np.max(probs)) - 1.0) / (n - 1.0)))


def choice_answer(keys: list[str], probs) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    return {
        "type": "choice",
        "choice": keys[int(np.argmax(probs))],
        "confidence": round(confidence(probs), ROUND),
        "probabilities": {k: round(float(p), ROUND) for k, p in zip(keys, probs)},
    }


def score_answer(levels: list, probs) -> dict:
    """Score is the expected level, so it can land between levels (0.57/0.43 on 1-2 -> 1.43)."""
    probs = np.asarray(probs, dtype=np.float64)
    expected = float(np.dot(np.arange(len(levels)), probs))
    return {
        "type": "score",
        "score": round(expected, ROUND),
        "confidence": round(confidence(probs), ROUND),
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": {str(i): round(float(p), ROUND) for i, p in enumerate(probs)},
    }


def noul_answer(p_true: float) -> dict:
    return {"type": "noul", "noul": round(float(p_true), ROUND)}
