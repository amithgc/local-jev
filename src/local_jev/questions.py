"""Questions, validated once and independent of whichever model answers them.

A wire-format question (the JSON Jev's API accepts) becomes a ``Question``: what is being
asked, and the closed set of answers it can have. Backends turn a Question into whatever
their model needs -- tool schemas, entailment hypotheses, a lettered list -- and return one
probability per answer, in ``Question.answers`` order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MAX_CHOICES = 255
MIN_LEVELS, MAX_LEVELS = 1, 10


class QuestionError(ValueError):
    """The question is malformed. ``loc`` is the path inside the question object."""

    def __init__(self, msg: str, loc: tuple = ()):
        super().__init__(msg)
        self.msg, self.loc = msg, tuple(loc)


class ContextOverflow(ValueError):
    """State plus question do not fit the model's context window."""


def as_text(value: Any) -> str:
    """Render instructions, criteria or state. Strings pass through; structure becomes JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


@dataclass(frozen=True)
class Question:
    type: str                                   # "choice" | "score" | "noul"
    instructions: str = ""
    keys: tuple = ()                            # choice: option names, in the caller's order
    descriptions: tuple = ()                    # choice / score: one text per answer ("" if none)
    legend: tuple = ()                          # score: the caller's level objects, returned verbatim
    yes: str = ""                               # noul: what counts as yes
    no: str = ""                                # noul: what counts as no
    raw: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def n_answers(self) -> int:
        return 2 if self.type == "noul" else len(self.descriptions)

    @property
    def answers(self) -> tuple:
        """Human-readable names of the answers, aligned with the probabilities a backend returns."""
        if self.type == "noul":
            return ("yes", "no")
        if self.type == "choice":
            return self.keys
        return tuple(f"{i + 1}. {text}" for i, text in enumerate(self.descriptions))


def parse_question(question: Any) -> Question:
    """Validate a Jev wire-format question."""
    if isinstance(question, Question):
        return question
    if not isinstance(question, dict):
        raise QuestionError("Input should be a valid dictionary")
    qtype = question.get("type")
    instructions = as_text(question.get("instructions"))
    criteria = question.get("criteria")

    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise QuestionError("Choice criteria must be a non-empty map of option name to description", ("criteria",))
        if len(criteria) > MAX_CHOICES:
            raise QuestionError(f"Choice criteria may have at most {MAX_CHOICES} options", ("criteria",))
        keys = tuple(str(k) for k in criteria)
        return Question("choice", instructions, keys, tuple(as_text(criteria[k]) for k in criteria), raw=question)

    if qtype == "score":
        if not isinstance(criteria, list) or not (MIN_LEVELS <= len(criteria) <= MAX_LEVELS):
            raise QuestionError(f"Score criteria must be an ordered list of {MIN_LEVELS}-{MAX_LEVELS} levels", ("criteria",))
        return Question("score", instructions, descriptions=tuple(as_text(level) for level in criteria),
                        legend=tuple(criteria), raw=question)

    if qtype == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise QuestionError("Noul criteria must be an object with optional true/false keys", ("criteria",))
        criteria = criteria or {}
        if set(criteria) - {"true", "false"}:
            raise QuestionError("Noul criteria accepts only the keys 'true' and 'false'", ("criteria",))
        if not instructions and not criteria:
            raise QuestionError("A noul needs instructions or criteria", ("instructions",))
        return Question("noul", instructions, yes=as_text(criteria.get("true")), no=as_text(criteria.get("false")), raw=question)

    raise QuestionError("Input tag does not match any of the expected tags: 'noul', 'choice', 'score'", ("type",))
