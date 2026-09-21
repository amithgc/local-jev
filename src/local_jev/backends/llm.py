"""Small-LLM backend: an instruction-tuned language model, read at the answer token.

The question is laid out as a lettered list, the model is asked to reply with one letter, and
instead of generating we read the next-token probabilities over the letters. A 1B+ instruction-tuned
model reliably binds a letter to the option it stands for (a 121M model did not: see docs/history.md),
so one forward pass yields the whole distribution. There is no sampling, so answers are deterministic.

The state comes first in every prompt, so all the questions asked about one state share a long
token prefix. That prefix is run through the model once; its cache is then copied for each question
and only the short remainder (question, options, "answer with one letter") is run per question, all
in one batch. The logits are the same as running each prompt in full -- the prefix tokens are
literally the first tokens of every prompt -- but the state is read once instead of once per question.
"""
from __future__ import annotations

import copy
import json
import string
from dataclasses import dataclass

import numpy as np

from ..questions import Question, as_text
from . import Backend, Item, Scored
from ._hf import fit_text, load_pretrained, pick_device, require_torch

SYSTEM = ("You are a careful, literal classifier. Read the input and the question, then answer with the single "
          "letter of the best option and nothing else.")
# The "json" layout follows SemIf (github.com/TheoLeeCJ/SemIf): evidence, criterion and lettered options as one JSON object.
SYSTEM_JSON = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
               "Respond with only its uppercase letter, with no explanation or reasoning.")
PROMPT_STYLES = ("plain", "json")
DTYPES = ("float16", "bfloat16", "float32")
MAX_LETTERS = 26
MIN_SHARED = 16          # below this many shared tokens, sharing is not worth a separate prefill
SUFFIX_BATCH = 32        # suffixes are short, so many fit in one pass
MAX_PADDING = 1.3        # start a new suffix batch rather than pad more than 30% extra tokens


def options_of(q: Question) -> list[str]:
    if q.type == "noul":
        return [f"Yes{' - ' + q.yes if q.yes else ''}", f"No{' - ' + q.no if q.no else ''}"]
    if q.type == "choice":
        return [f"{k} - {d}" if d else k for k, d in zip(q.keys, q.descriptions)]
    return [f"Level {i + 1} of {len(q.descriptions)}: {d}" for i, d in enumerate(q.descriptions)]


def task_of(q: Question) -> str:
    if q.instructions:
        return q.instructions
    return {"noul": "Is this true of the input?", "choice": "Which option best fits the input?",
            "score": "Which level best describes the input?"}[q.type]


def shared_prefix(sequences: list[list[int]]) -> int:
    """Length of the longest token prefix every sequence shares, leaving each at least one token of its own."""
    if not sequences:
        return 0
    limit = min(len(s) for s in sequences) - 1
    n = 0
    while n < limit and all(s[n] == sequences[0][n] for s in sequences):
        n += 1
    return max(n, 0)


def pack(lengths: list[int], max_batch: int = SUFFIX_BATCH, max_padding: float = MAX_PADDING) -> list[list[int]]:
    """Split suffixes into batches of similar length: indices, shortest first.

    Every row of a batch is padded to its longest member, and padding is computed like real tokens.
    Sorting by length and cutting a batch before padding would exceed ``max_padding`` times the real
    work keeps that waste bounded, at the cost of an occasional extra forward pass.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, current, real = [], [], 0
    for i in order:
        width = lengths[i]                                   # the longest so far, since the order is ascending
        if current and (len(current) >= max_batch or (len(current) + 1) * width > max_padding * (real + width)):
            batches.append(current)
            current, real = [], 0
        current.append(i)
        real += lengths[i]
    if current:
        batches.append(current)
    return batches


@dataclass
class SuffixBatch:
    input_ids: list
    attention_mask: list
    position_ids: list
    ends: list                  # index of each row's last real token


def suffix_layout(suffixes: list[list[int]], prefix_len: int, pad_id: int) -> SuffixBatch:
    """Right-pad the per-question suffixes that continue a shared, cached prefix."""
    width = max(len(s) for s in suffixes)
    ids, mask, pos, ends = [], [], [], []
    for s in suffixes:
        pad = width - len(s)
        ids.append(s + [pad_id] * pad)
        mask.append([1] * (prefix_len + len(s)) + [0] * pad)
        pos.append(list(range(prefix_len, prefix_len + len(s))) + [0] * pad)
        ends.append(len(s) - 1)
    return SuffixBatch(ids, mask, pos, ends)


class LLMBackend(Backend):
    kind = "llm"

    def __init__(self, card, device: str | None = None, **_):
        super().__init__()
        torch = require_torch()
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = pick_device(torch, device or card.options.get("device"))
        repo = card.options["hf_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_name = card.options.get("dtype") or ("float32" if self.device == "cpu" else "float16")
        if dtype_name not in DTYPES:
            raise ValueError(f"model {card.name!r}: dtype must be one of {DTYPES}")
        dtype = getattr(torch, dtype_name)
        self.style = card.options.get("prompt", "plain")
        if self.style not in PROMPT_STYLES:
            raise ValueError(f"model {card.name!r}: prompt must be one of {PROMPT_STYLES}")
        config = AutoConfig.from_pretrained(repo)
        if config.model_type == "qwen3_5":
            # Qwen3.5 checkpoints are multimodal; the text model alone is all this backend needs.
            model = load_pretrained(transformers.Qwen3_5ForCausalLM, repo, dtype, config=config.get_text_config())
        else:
            model = load_pretrained(AutoModelForCausalLM, repo, dtype)
        self.model = model.to(self.device).eval()
        self.context_tokens = int(card.options.get("context_tokens", 8192))
        self.batch = int(card.options.get("batch_size", 8))
        self.share = bool(card.options.get("share_prefix", True))
        self.letters = [self._single_token(ch) for ch in string.ascii_uppercase]

    def _single_token(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{text!r} is not a single token for this tokenizer; the llm backend needs single-token letters")
        return ids[0]

    def _prompt(self, q: Question, state_text: str, options: list[str], letters: str) -> str:
        if self.style == "json":
            payload = {"evidence": state_text, "criterion": task_of(q),
                       "options": [{"letter": letter, "description": opt} for letter, opt in zip(letters, options)]}
            messages = [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        else:
            listing = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))
            user = f"Input:\n{state_text}\n\nQuestion: {task_of(q)}\n\nOptions:\n{listing}\n\nAnswer with one letter."
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        # enable_thinking=False: Qwen3-family templates would otherwise open a reasoning block before the answer.
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # -- forward passes -------------------------------------------------------------------------
    def _plain(self, prompts: list[str]) -> np.ndarray:
        """Next-token logits over A..Z, each prompt run in full (left-padded batches)."""
        torch, out = self.torch, []
        for start in range(0, len(prompts), self.batch):
            enc = self.tokenizer(prompts[start:start + self.batch], return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out.append(logits[:, self.letters].cpu().numpy())
        return np.concatenate(out)

    def _shared(self, prompts: list[str]) -> np.ndarray:
        """The same logits for prompts that share a long prefix, reading that prefix once."""
        torch = self.torch
        seqs = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        n = shared_prefix(seqs)
        if len(prompts) < 2 or n < MIN_SHARED:
            return self._plain(prompts)
        pad = self.tokenizer.pad_token_id
        with torch.no_grad():
            prefix = self.model(input_ids=torch.tensor([seqs[0][:n]], device=self.device),
                                attention_mask=torch.ones((1, n), dtype=torch.long, device=self.device),
                                use_cache=True, logits_to_keep=1).past_key_values
            z = np.zeros((len(seqs), MAX_LETTERS))
            batches = pack([len(s) - n for s in seqs])
            for k, rows in enumerate(batches):
                chunk = [seqs[i][n:] for i in rows]
                cache = prefix if k == len(batches) - 1 else copy.deepcopy(prefix)     # the last batch may consume the original
                cache.reorder_cache(torch.zeros(len(chunk), dtype=torch.long, device=self.device))
                layout = suffix_layout(chunk, n, pad)
                keep = sorted(set(layout.ends))
                logits = self.model(input_ids=torch.tensor(layout.input_ids, device=self.device),
                                    attention_mask=torch.tensor(layout.attention_mask, device=self.device),
                                    position_ids=torch.tensor(layout.position_ids, device=self.device),
                                    past_key_values=cache, use_cache=True,
                                    logits_to_keep=torch.tensor(keep, device=self.device)).logits.float()
                picked = torch.stack([logits[j, keep.index(end), :] for j, end in enumerate(layout.ends)])
                z[rows] = picked[:, self.letters].cpu().numpy()
        return z

    def _letter_logits(self, prompts: list[str], owners: list[str]) -> np.ndarray:
        """Logits for every prompt. Prompts with the same owner (state) share a prefix and are run together."""
        if not self.share:
            return self._plain(prompts)
        groups: dict[str, list[int]] = {}
        for i, owner in enumerate(owners):
            groups.setdefault(owner, []).append(i)
        z = np.zeros((len(prompts), MAX_LETTERS))
        singles = [g[0] for g in groups.values() if len(g) == 1]
        if singles:
            z[singles] = self._plain([prompts[i] for i in singles])
        for g in groups.values():
            if len(g) > 1:
                z[g] = self._shared([prompts[i] for i in g])
        return z

    # -- scoring ------------------------------------------------------------------------------------
    def score(self, items: list[Item], truncate: bool = False, max_state_tokens: int | None = None) -> list[Scored]:
        # Fit each distinct state once, against its hungriest question, so every question sees identical text.
        texts: dict[str, tuple[str, bool]] = {}
        by_state: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            by_state.setdefault(as_text(item.state), []).append(i)
        overheads = {}
        for i, item in enumerate(items):
            opts = options_of(item.question)[:MAX_LETTERS]
            overheads[i] = len(self.tokenizer.encode(self._prompt(item.question, "", opts, string.ascii_uppercase), add_special_tokens=False))
        for raw, members in by_state.items():
            room = self.context_tokens - max(overheads[i] for i in members) - 8
            budget = min(room, max_state_tokens) if max_state_tokens else room
            texts[raw] = fit_text(self.tokenizer, raw, budget, truncate, f"a {self.context_tokens}-token window")

        prompts, plan, meta, owner = [], [], [], []
        for i, item in enumerate(items):
            q, options = item.question, options_of(item.question)
            raw = as_text(item.state)
            text, cut = texts[raw]
            # More than 26 options: ask about them in pages of 26, then let the page winners compete.
            pages = [list(range(k, min(k + MAX_LETTERS, len(options)))) for k in range(0, len(options), MAX_LETTERS)]
            first = len(prompts)
            for page in pages:
                prompts.append(self._prompt(q, text, [options[k] for k in page], string.ascii_uppercase))
                owner.append(raw)
            plan.append((first, pages))
            meta.append((cut, overheads[i] + len(self.tokenizer.encode(text, add_special_tokens=False)), text))

        with self.lock:
            z = self._letter_logits(prompts, owner)
            finals, final_prompts, final_owner = [], [], []
            for item, (first, pages), (_, _, text) in zip(items, plan, meta):
                if len(pages) > 1:
                    winners = [page[int(np.argmax(z[first + p][: len(page)]))] for p, page in enumerate(pages)]
                    options = options_of(item.question)
                    finals.append(winners)
                    final_prompts.append(self._prompt(item.question, text, [options[k] for k in winners], string.ascii_uppercase))
                    final_owner.append(as_text(item.state))
                else:
                    finals.append(None)
            final_z = self._letter_logits(final_prompts, final_owner) if final_prompts else np.zeros((0, MAX_LETTERS))

        scored, cursor = [], 0
        for item, (first, pages), (cut, tokens, _), winners in zip(items, plan, meta, finals):
            n = item.question.n_answers
            if winners is None:
                logits = z[first][:n]
                e = np.exp(logits - logits.max())
                probs = e / e.sum()
            else:                                          # within-page softmax, weighted by the play-off between page winners
                top = final_z[cursor][: len(winners)]
                cursor += 1
                weight = np.exp(top - top.max())
                weight /= weight.sum()
                probs = np.zeros(n)
                for p, page in enumerate(pages):
                    logits = z[first + p][: len(page)]
                    e = np.exp(logits - logits.max())
                    probs[page] = weight[p] * e / e.sum()
            scored.append(Scored(probs / probs.sum(), tokens * len(pages), cut))
        return scored
