"""Entailment backend: every answer becomes a hypothesis, and a cross-encoder judges it against the state.

This is the standard strong approach to zero-shot classification with small models: a
natural-language-inference model reads (premise = the state, hypothesis = "the answer is X")
and says whether the premise entails it. No fine-tuning is needed to follow a new question,
because judging a sentence against a text is exactly what these models were trained to do.
"""
from __future__ import annotations

import numpy as np

from ..questions import Question, as_text
from . import Backend, Item, Scored
from ._hf import fit_text, load_pretrained, pick_device, require_torch


def hypotheses(q: Question) -> list[str]:
    """One declarative sentence per answer (for a yes/no: a single sentence)."""
    ask = q.instructions.rstrip(" ?.:")
    if q.type == "noul":
        text = q.instructions.strip()
        if not text or text.endswith("?"):
            text = f'The answer to the question "{ask}" is yes.' if ask else "This is the case."
        return [f"{text} ({q.yes})" if q.yes else text]
    names = q.keys if q.type == "choice" else tuple(f"level {i + 1}" for i in range(len(q.descriptions)))
    out = []
    for name, desc in zip(names, q.descriptions):
        answer = f"{name}: {desc}" if desc and q.type == "choice" else (desc or name)
        out.append(f'Asked "{ask}", the right answer for this text is "{answer}".' if ask else f'This text is best described as "{answer}".')
    return out


class NLIBackend(Backend):
    kind = "nli"

    def __init__(self, card, device: str | None = None, **_):
        super().__init__()
        torch = require_torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = pick_device(torch, device or card.options.get("device"))
        repo = card.options["hf_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        # Half precision on a GPU (how the checkpoint is published); float32 on the CPU, where float16 is slow or unsupported.
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = load_pretrained(AutoModelForSequenceClassification, repo, dtype).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels.get("entailment", 0)
        # Two-way models say entailment / not_entailment; three-way models add neutral and contradiction.
        self.against = [i for name, i in labels.items() if name != "entailment"]
        self.context_tokens = int(card.options.get("context_tokens", 512))
        self.batch = int(card.options.get("batch_size", 16))

    def _logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Entailment log-odds for each (premise, hypothesis) pair."""
        torch, out = self.torch, []
        for start in range(0, len(pairs), self.batch):
            chunk = pairs[start:start + self.batch]
            enc = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk], return_tensors="pt", padding=True,
                                 truncation="only_first", max_length=self.context_tokens).to(self.device)
            with torch.no_grad():
                logp = torch.log_softmax(self.model(**enc).logits.float(), dim=-1)
            against = torch.logsumexp(logp[:, self.against], dim=-1)
            out.append((logp[:, self.entail] - against).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, items: list[Item], truncate: bool = False, max_state_tokens: int | None = None) -> list[Scored]:
        pairs, spans, meta = [], [], []
        for item in items:
            hyps = hypotheses(item.question)
            room = self.context_tokens - max(len(self.tokenizer.encode(h, add_special_tokens=False)) for h in hyps) - 8
            budget = min(room, max_state_tokens) if max_state_tokens else room
            premise, cut = fit_text(self.tokenizer, as_text(item.state), budget, truncate, f"a {self.context_tokens}-token window")
            spans.append((len(pairs), len(hyps)))
            pairs += [(premise, h) for h in hyps]
            meta.append((cut, len(self.tokenizer.encode(premise, add_special_tokens=False)) * len(hyps)))
        with self.lock:
            z = self._logits(pairs)
        scored = []
        for item, (start, n), (cut, tokens) in zip(items, spans, meta):
            if item.question.type == "noul":
                p = 1.0 / (1.0 + np.exp(-z[start]))
                probs = np.array([p, 1.0 - p])
            else:
                e = np.exp(z[start:start + n] - z[start:start + n].max())
                probs = e / e.sum()
            scored.append(Scored(probs, tokens, cut))
        return scored
