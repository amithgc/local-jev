"""Backend-neutral pieces, tested without loading any model."""
import json

import numpy as np
import pytest

from local_jev.backends import BACKENDS, Backend, Item, Scored, load_backend
from local_jev.engine import Engine
from local_jev.models import ModelCard, discover
from local_jev.questions import QuestionError, parse_question

CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "sales": None}}
SCORE = {"type": "score", "instructions": "How urgent", "criteria": ["can wait", {"summary": "today"}]}
NOUL = {"type": "noul", "instructions": "Is this spam?", "criteria": {"true": "ads"}}


def test_questions_are_parsed_once_for_every_backend():
    q = parse_question(CHOICE)
    assert (q.type, q.keys, q.descriptions, q.answers, q.n_answers) == ("choice", ("billing", "sales"), ("Payments", ""), ("billing", "sales"), 2)
    s = parse_question(SCORE)
    assert s.descriptions == ("can wait", '{"summary": "today"}') and s.legend == ("can wait", {"summary": "today"})
    assert s.answers == ("1. can wait", '2. {"summary": "today"}')
    n = parse_question(NOUL)
    assert (n.yes, n.no, n.answers) == ("ads", "", ("yes", "no"))
    assert parse_question(q) is q
    with pytest.raises(QuestionError):
        parse_question({"type": "noul", "criteria": {"maybe": "x"}})


def test_nli_hypotheses_are_declarative_sentences():
    from local_jev.backends.nli import hypotheses
    assert hypotheses(parse_question(CHOICE)) == ['Asked "Which team", the right answer for this text is "billing: Payments".',
                                                  'Asked "Which team", the right answer for this text is "sales".']
    assert hypotheses(parse_question(NOUL)) == ['The answer to the question "Is this spam" is yes. (ads)']
    assert hypotheses(parse_question({"type": "noul", "instructions": "The customer wants a refund."})) == ["The customer wants a refund."]
    assert len(hypotheses(parse_question(SCORE))) == 2


def test_llm_options_cover_all_three_primitives():
    from local_jev.backends.llm import options_of, task_of
    assert options_of(parse_question(CHOICE)) == ["billing - Payments", "sales"]
    assert options_of(parse_question(NOUL)) == ["Yes - ads", "No"]
    assert options_of(parse_question(SCORE))[0] == "Level 1 of 2: can wait"
    assert task_of(parse_question({"type": "choice", "criteria": {"a": None, "b": None}})) == "Which option best fits the input?"


def test_builtin_cards_name_a_known_backend():
    cards = discover()
    assert {"nli-deberta-large", "llm-qwen2.5-1.5b", "llm-qwen3-4b", "llm-qwen3.5-2b", "llm-qwen3.5-4b"} <= set(cards)
    assert {c.backend for c in cards.values()} <= set(BACKENDS)
    for card in cards.values():
        assert card.license, card.name                                  # every shipped model states its licence
        if card.backend == "ensemble":
            assert all(m in cards for m in card.options["members"]), card.name
        else:
            assert card.options.get("hf_model") and card.temperature, card.name      # and ships calibrated


def test_user_cards_override_builtin_ones(tmp_path, monkeypatch):
    import local_jev.models as models
    (tmp_path / "mine.json").write_text(json.dumps({"name": "mine", "backend": "llm", "hf_model": "org/model", "batch_size": 2}))
    (tmp_path / "nli-deberta-large.json").write_text(json.dumps({"name": "nli-deberta-large", "backend": "nli", "hf_model": "x/y",
                                                                  "calibration": {"temperature": {"noul": 1.2}}}))
    (tmp_path / "broken.json").write_text("{not json")
    monkeypatch.setattr(models, "USER_CARDS_DIR", tmp_path)
    cards = models.discover()
    assert cards["mine"].options == {"hf_model": "org/model", "batch_size": 2} and cards["mine"].source == "user"
    assert cards["nli-deberta-large"].temperature == {"noul": 1.2} and cards["nli-deberta-large"].source == "user"
    assert "broken" not in cards


def test_default_is_the_best_model_that_needs_no_download(monkeypatch):
    import local_jev.models as models
    cards = {"nli-deberta-large": ModelCard("nli-deberta-large", priority=70), "tuned": ModelCard("tuned", priority=75),
             "big": ModelCard("big", backend="llm", priority=80)}
    for card in cards.values():
        card.unavailable = lambda: None
    monkeypatch.setattr(models, "on_disk", lambda card, all_cards: card.name != "big")
    assert models.default_model(cards) == "tuned"                     # "big" would have to be downloaded first
    monkeypatch.setattr(models, "on_disk", lambda card, all_cards: True)
    assert models.default_model(cards) == "big"
    monkeypatch.setattr(models, "on_disk", lambda card, all_cards: False)
    assert models.default_model(cards) == "nli-deberta-large"         # fresh machine: fetch the smallest model


def test_unknown_backend_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown backend"):
        load_backend(ModelCard("x", backend="telepathy"))


class ConstantBackend(Backend):
    kind = "constant"

    def score(self, items, truncate=False, max_state_tokens=None):
        return [Scored(np.full(i.question.n_answers, 1.0 / i.question.n_answers) if i.question.type != "noul"
                       else np.array([0.9, 0.1]), tokens=7) for i in items]


def test_engine_is_backend_agnostic_and_applies_the_cards_calibration():
    engine = Engine.__new__(Engine)
    engine.models = {"m": ModelCard("m", backend="constant", calibration={"temperature": {"noul": 2.0}})}
    engine.default_model = "m"
    engine.backend = lambda name: ConstantBackend()
    out = engine.system_one("some text", {"c": CHOICE, "s": SCORE, "n": NOUL})
    assert out["answers"]["c"]["probabilities"] == {"billing": 0.5, "sales": 0.5} and out["answers"]["c"]["confidence"] == 0.0
    assert out["answers"]["s"]["score"] == 0.5 and out["answers"]["s"]["legend"] == {"0": "can wait", "1": {"summary": "today"}}
    assert out["answers"]["n"]["noul"] == 0.75                     # 0.9 cooled by T=2: sigmoid(ln(9) / 2)
    assert out["usage"] == {"input_tokens": 21, "output_tokens": 3}
    assert engine.resolve("jev-latest") == "m" and engine.resolve("nope") is None


def test_ensemble_pools_calibrated_members_log_linearly():
    from local_jev.backends.ensemble import EnsembleBackend

    class Fixed(Backend):
        context_tokens = 100

        def __init__(self, probs):
            super().__init__()
            self.probs = np.array(probs)

        def score(self, items, truncate=False, max_state_tokens=None):
            return [Scored(self.probs, tokens=5) for _ in items]

    members = {"sure": (Fixed([0.98, 0.01, 0.01]), {}), "hedging": (Fixed([0.30, 0.40, 0.30]), {})}
    card = ModelCard("e", backend="ensemble", options={"members": ["sure", "hedging"]})
    out = EnsembleBackend(card, resolve=lambda name: members[name]).score([Item(parse_question(
        {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}), "text")])[0]
    assert out.probs.argmax() == 0 and np.isclose(out.probs.sum(), 1.0) and out.tokens == 10      # the confident member outvotes the unsure one
    expected = np.sqrt(np.array([0.98, 0.01, 0.01]) * np.array([0.30, 0.40, 0.30]))
    assert np.allclose(out.probs, expected / expected.sum())
    with pytest.raises(ValueError):
        EnsembleBackend(ModelCard("solo", backend="ensemble", options={"members": ["sure"]}), resolve=lambda n: members[n])


def test_llm_shared_prefix_leaves_every_prompt_a_suffix():
    from local_jev.backends.llm import shared_prefix
    assert shared_prefix([[1, 2, 3, 4], [1, 2, 3, 9], [1, 2, 5]]) == 2
    assert shared_prefix([[1, 2, 3], [1, 2, 3]]) == 2                  # identical prompts still keep one token each
    assert shared_prefix([[7, 8]]) == 1 and shared_prefix([]) == 0


def test_llm_suffix_layout_continues_the_cached_prefix():
    from local_jev.backends.llm import suffix_layout
    b = suffix_layout([[5, 6, 7], [8]], prefix_len=10, pad_id=0)
    assert b.input_ids == [[5, 6, 7], [8, 0, 0]]
    assert b.attention_mask == [[1] * 13, [1] * 11 + [0, 0]]            # the mask covers the cached prefix too
    assert b.position_ids == [[10, 11, 12], [10, 0, 0]]                  # positions resume after the prefix
    assert b.ends == [2, 0]


def test_llm_pack_bounds_padding_and_keeps_every_suffix():
    from local_jev.backends.llm import pack
    lengths = [50, 64, 62, 133, 56, 99, 134]
    batches = pack(lengths)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) * max(lengths[i] for i in b) <= 1.3 * sum(lengths[i] for i in b) or len(b) == 1
    assert pack([10] * 40, max_batch=32) == [list(range(32)), list(range(32, 40))]


def test_all_backends_share_one_device_lock():
    from local_jev.backends import DEVICE_LOCK

    class A(Backend):
        def score(self, items, truncate=False, max_state_tokens=None):
            return []

    assert A().lock is DEVICE_LOCK and A().lock is A().lock       # two models can never use the GPU at the same time
    with DEVICE_LOCK:
        with A().lock:                                             # re-entrant: an ensemble scores through its members
            pass


def test_engine_unloads_by_memory_not_by_count(monkeypatch):
    import local_jev.engine as engine_mod

    def card(name, gb, backend="llm", **options):
        return ModelCard(name, backend=backend, options={"size": {"memory_gb": gb}, **options})

    engine = Engine.__new__(Engine)
    engine.models = {"big-a": card("big-a", 8.5), "big-b": card("big-b", 8.0), "small": card("small", 0.9, "nli"),
                     "big-c": card("big-c", 8.0), "pair": card("pair", 0, "ensemble", members=["small", "big-b"])}
    engine.default_model, engine._backends, engine.warmup = "big-a", {}, False
    for c in engine.models.values():
        c.unavailable = lambda: None
    monkeypatch.setattr(engine_mod, "load_backend", lambda c, resolve=None: (
        [resolve(m) for m in c.options.get("members", [])], ConstantBackend())[1])
    monkeypatch.setenv("LOCAL_JEV_MEMORY_GB", "18")

    engine.backend("big-a"); engine.backend("small")
    assert set(engine._backends) == {"big-a", "small"} and engine.loaded_gb() == 9.4
    engine.backend("big-b")                                          # 9.4 + 8.0 fits in 18
    assert set(engine._backends) == {"big-a", "small", "big-b"}
    engine.backend("big-a")                                          # already loaded: no reload, just most recent
    assert list(engine._backends) == ["small", "big-b", "big-a"]
    monkeypatch.setenv("LOCAL_JEV_MEMORY_GB", "17")
    engine.backend("big-c")                                          # needs 8.0: evicts least recent first until it fits
    assert list(engine._backends) == ["big-a", "big-c"] and engine.loaded_gb() <= 17
    monkeypatch.setenv("LOCAL_JEV_MEMORY_GB", "1")
    engine.backend("big-b")                                          # over budget alone: still allowed, as the only model
    assert list(engine._backends) == ["big-b"]
    monkeypatch.setenv("LOCAL_JEV_MEMORY_GB", "18")
    engine.backend("pair")                                           # an ensemble loads its members and keeps them
    assert {"pair", "small", "big-b"} <= set(engine._backends)
    engine._evict("small")                                           # unloading a member unloads the ensemble built on it
    assert "pair" not in engine._backends


def test_every_model_is_warmed_when_it_loads(monkeypatch):
    import local_jev.engine as engine_mod
    warmed = []

    class Tracked(ConstantBackend):
        def warmup(self):
            warmed.append(self)
            return 0.0

    def build(warm):
        engine = Engine.__new__(Engine)
        engine.models = {"a": ModelCard("a", backend="nli"), "b": ModelCard("b", backend="nli")}
        engine.default_model, engine._backends, engine.warmup = "a", {}, warm
        for c in engine.models.values():
            c.unavailable = lambda: None
        return engine

    monkeypatch.setattr(engine_mod, "load_backend", lambda c, resolve=None: Tracked())
    engine = build(True)
    engine.backend("a"); engine.backend("b"); engine.backend("a")
    assert len(warmed) == 2                                          # once per load, not per request
    warmed.clear()
    build(False).backend("a")
    assert warmed == []                                              # --no-warmup
