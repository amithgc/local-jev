"""Accuracy and calibration metrics over (probability vector, gold index) pairs."""
from __future__ import annotations

import numpy as np


def summarise(probs: list[np.ndarray], gold: list[int], kind: str) -> dict:
    n = len(gold)
    if n == 0:
        return {"n": 0}
    pred = np.array([int(np.argmax(p)) for p in probs])
    gold_arr = np.array(gold)
    p_gold = np.array([max(float(p[g]), 1e-9) for p, g in zip(probs, gold)])
    p_max = np.array([float(np.max(p)) for p in probs])
    brier = float(np.mean([np.sum((p - np.eye(len(p))[g]) ** 2) for p, g in zip(probs, gold)]))
    out = {"n": n, "accuracy": float(np.mean(pred == gold_arr)), "nll": float(-np.mean(np.log(p_gold))),
           "brier": brier, "ece": ece(p_max, pred == gold_arr),
           "chance": float(np.mean([1.0 / len(p) for p in probs]))}
    if kind == "score":
        expected = np.array([float(np.dot(np.arange(len(p)), p)) for p in probs])
        out["mae"] = float(np.mean(np.abs(expected - gold_arr)))
        out["accuracy"] = float(np.mean(np.clip(np.rint(expected), 0, None).astype(int) == gold_arr))
    return out


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: the gap between how sure the model is and how often it is right."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                         "confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean())})
    return rows
