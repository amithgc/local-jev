"""The project API, driven through FastAPI's test client with a fake engine (no model needed)."""
import json

import pytest
from fastapi.testclient import TestClient

from local_jev.api import create_app
from local_jev.engine import to_answer
from local_jev.models import ModelCard
from local_jev.questions import parse_question


class FakeEngine:
    """Answers deterministically from the text so assertions are meaningful."""
    default_model = "fake"
    models = {"fake": ModelCard("fake", "test double", "2026-01-01")}

    question = staticmethod(parse_question)

    def resolve(self, name):
        return "fake" if name in (None, "", "fake", "jev-latest") else None

    def answer_many(self, model, states, questions, truncate=False, max_state_tokens=None):
        answers = []
        for state in states:
            text = json.dumps(state).lower()
            row = {}
            for qid, q in questions.items():
                if q.type == "noul":
                    probs = [0.9, 0.1] if "invoice" in text else [0.1, 0.9]
                elif q.type == "choice":
                    raw = [0.8 if key.lower() in text else 0.1 for key in q.keys]
                    probs = [p / sum(raw) for p in raw]
                else:
                    probs = [1.0 if i == q.n_answers - 1 else 0.0 for i in range(q.n_answers)]
                row[qid] = to_answer(q, probs)
            answers.append(row)
        return answers, [{"input_tokens": 10, "output_tokens": len(questions), "truncated": False} for _ in states]

    def system_one(self, state, questions, model=None):
        answers, _ = self.answer_many("fake", [state], {k: parse_question(v) for k, v in questions.items()})
        return {"model": "fake", "answers": answers[0], "truncated": "LONG" in json.dumps(state),
                "usage": {"input_tokens": 10, "output_tokens": len(questions)}}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_NO_SEED", "1")
    return TestClient(create_app(FakeEngine(), db_path=str(tmp_path / "t.db"), ui=True))


def wait_for_run(client, pid):
    import time
    for _ in range(100):
        status = client.get(f"/api/projects/{pid}/run").json()
        if status["state"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_full_project_flow(client):
    pid = client.post("/api/projects", json={"name": "Inbox", "noun": "emails"}).json()["id"]
    csv_text = "subject,body\nInvoice 42,Please pay the invoice\nLunch?,Are you free on Sunday\nNewsletter,Weekly digest"
    assert client.post(f"/api/projects/{pid}/items/import", json={"filename": "mail.csv", "content": csv_text}).json() == {"added": 3}

    noul = client.post(f"/api/projects/{pid}/questions", json={
        "name": "Invoice", "type": "noul", "instructions": "Is this an invoice?", "threshold": 0.5}).json()
    client.post(f"/api/projects/{pid}/questions", json={
        "name": "Kind", "type": "choice", "instructions": "What kind?", "criteria": {"invoice": None, "lunch": None, "newsletter": None}})
    client.post(f"/api/projects/{pid}/questions", json={
        "name": "Urgency", "type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]})

    before = client.get(f"/api/projects/{pid}/results").json()
    assert before["unsorted"] == 3 and before["questions"][0]["answered"] == 0

    assert client.post(f"/api/projects/{pid}/run", json={}).json()["state"] == "running"
    assert wait_for_run(client, pid)["state"] == "done"

    after = client.get(f"/api/projects/{pid}/results").json()
    assert after["unsorted"] == 0
    rows = {r["value"]: r["count"] for r in after["questions"][0]["rows"]}
    assert rows == {"yes": 1, "no": 2}
    assert after["questions"][2]["average"] == 3.0 and after["questions"][2]["of"] == 3

    filtered = client.get(f"/api/projects/{pid}/items", params={"filters": json.dumps([{"q": noul["id"], "v": "yes"}])}).json()
    assert filtered["total"] == 1 and filtered["items"][0]["title"] == "Invoice 42"

    # Moving the threshold re-buckets existing answers without another run.
    client.put(f"/api/questions/{noul['id']}", json={"name": "Invoice", "type": "noul", "instructions": "Is this an invoice?", "threshold": 0.05})
    moved = client.get(f"/api/projects/{pid}/results").json()
    assert {r["value"]: r["count"] for r in moved["questions"][0]["rows"]} == {"yes": 3, "no": 0} and moved["unsorted"] == 0

    # Rewording the question invalidates its answers.
    client.put(f"/api/questions/{noul['id']}", json={"name": "Invoice", "type": "noul", "instructions": "Is this a bill?", "threshold": 0.5})
    assert client.get(f"/api/projects/{pid}/results").json()["unsorted"] == 3

    export = client.get(f"/api/projects/{pid}/export.csv")
    assert export.status_code == 200 and "Invoice 42" in export.text and export.text.splitlines()[0].startswith("id,title,content")


def test_try_does_not_save(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    client.post(f"/api/projects/{pid}/items/import", json={"content": "an invoice\n\nhello there", "split": "paragraphs"})
    out = client.post(f"/api/projects/{pid}/try", json={"question": {"name": "x", "type": "noul", "instructions": "Invoice?"}, "n": 6}).json()
    assert [r["answer"]["bucket"] for r in out["results"]] == ["no", "yes"] or [r["answer"]["bucket"] for r in out["results"]] == ["yes", "no"]
    assert client.get(f"/api/projects/{pid}").json()["questions"] == []


def test_bad_question_is_rejected(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    res = client.post(f"/api/projects/{pid}/questions", json={"name": "x", "type": "score", "instructions": "r", "criteria": []})
    assert res.status_code == 422


def test_wire_api_shapes(client):
    res = client.post("/v1/systemone", json={"state": "an invoice", "model": "jev-latest",
                                             "questions": {"q": {"type": "noul", "instructions": "Invoice?"}}})
    assert res.status_code == 200 and res.json()["answers"]["q"] == {"type": "noul", "noul": 0.9}
    assert res.headers["x-typesafe-request-id"]
    assert client.post("/v1/systemone", json={"state": "x", "model": "nope", "questions": {"q": {"type": "noul", "instructions": "?"}}}).status_code == 404
    assert client.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
    assert "jev-latest" in [m["name"] for m in client.get("/v1/models").json()["models"]]


def test_importers():
    from local_jev.importer import parse, title_of
    assert parse("a.jsonl", '{"subject":"Hi","body":"x"}\n"plain"') == [{"subject": "Hi", "body": "x"}, "plain"]
    assert parse("a.json", '{"items": ["one", "two"]}') == ["one", "two"]
    assert parse("a.txt", "line one\nline two") == ["line one", "line two"]
    assert parse("a.txt", "para one\nstill one\n\npara two") == ["para one\nstill one", "para two"]
    eml = "From: a@b.c\nTo: d@e.f\nSubject: Hello\n\nBody text here"
    assert parse("m.eml", eml)[0] == {"from": "a@b.c", "to": "d@e.f", "subject": "Hello", "body": "Body text here"}
    assert title_of({"from": "x", "Subject": "Quarterly report"}) == "Quarterly report"


def make_project(client):
    pid = client.post("/api/projects", json={"name": "Inbox", "noun": "emails"}).json()["id"]
    ids = {}
    for body in ({"name": "Invoice", "type": "noul", "instructions": "Is this an invoice?"},
                 {"name": "Kind", "type": "choice", "instructions": "What kind?", "criteria": {"invoice": None, "lunch": None}},
                 {"name": "Urgency", "type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}):
        ids[body["name"]] = client.post(f"/api/projects/{pid}/questions", json=body).json()["id"]
    return pid, ids


def test_ask_answers_every_question_and_saves_nothing(client):
    pid, ids = make_project(client)
    client.post(f"/api/projects/{pid}/items/import", json={"content": "an item"})
    client.post(f"/api/questions/{ids['Kind']}/enabled?enabled=false")          # the tick belongs to batch runs, not to asking
    out = client.post(f"/api/projects/{pid}/ask", json={"state": "Please pay the invoice by Friday"}).json()
    assert out["model"] == "fake" and out["input_tokens"] == 10 and out["truncated"] is False and out["elapsed_ms"] >= 0
    assert set(out["answers"]) == {str(i) for i in ids.values()}
    assert out["answers"][str(ids["Invoice"])]["bucket"] == "yes" and out["answers"][str(ids["Invoice"])]["noul"] == 0.9
    assert out["answers"][str(ids["Kind"])]["bucket"] == "invoice" and "probabilities" in out["answers"][str(ids["Kind"])]
    assert out["answers"][str(ids["Urgency"])]["bucket"] == "2"
    results = client.get(f"/api/projects/{pid}/results").json()
    assert all(q["answered"] == 0 for q in results["questions"]) and results["unsorted"] == 1


def test_ask_honours_question_ids_and_structured_state(client):
    pid, ids = make_project(client)
    out = client.post(f"/api/projects/{pid}/ask", json={"state": {"subject": "Lunch?", "body": "Sunday"}, "question_ids": [ids["Kind"]]}).json()
    assert list(out["answers"]) == [str(ids["Kind"])] and out["answers"][str(ids["Kind"])]["choice"] == "lunch"
    assert client.post(f"/api/projects/{pid}/ask", json={"state": ["an invoice", "attached"]}).json()["answers"][str(ids["Invoice"])]["bucket"] == "yes"


def test_ask_rejects_empty_input_and_projects_without_questions(client):
    pid, _ = make_project(client)
    for empty in ("", "   ", None, {}, []):
        res = client.post(f"/api/projects/{pid}/ask", json={"state": empty})
        assert res.status_code == 422 and "Type or paste" in res.json()["detail"]
    bare = client.post("/api/projects", json={"name": "Bare"}).json()["id"]
    res = client.post(f"/api/projects/{bare}/ask", json={"state": "hello"})
    assert res.status_code == 422 and "no questions" in res.json()["detail"]
    assert client.post(f"/api/projects/{pid}/ask", json={"state": "x", "model": "nope"}).status_code == 404


def test_over_long_state_is_answered_and_flagged(client):
    q = {"q": {"type": "noul", "instructions": "Invoice?"}}
    long = client.post("/v1/systemone", json={"state": "LONG invoice", "questions": q})
    assert long.status_code == 200 and long.headers.get("x-local-jev-truncated") == "true"
    assert "truncated" not in long.json()                                   # the body stays exactly Jev's shape
    short = client.post("/v1/systemone", json={"state": "an invoice", "questions": q})
    assert "x-local-jev-truncated" not in short.headers


def test_meta_exposes_each_models_evaluation(client, monkeypatch):
    models = client.get("/api/meta").json()["models"]
    assert models[0]["name"] == "fake" and models[0]["evaluation"] is None          # not measured yet: the UI shows just the name
    measured = {"benchmark": "JevBench public", "items": 231, "accuracy": 0.541, "p50_ms": 76, "p95_ms": 710,
                "reference": {"jev-1.13.0": 0.866}}
    monkeypatch.setattr(FakeEngine.models["fake"], "evaluation", measured)
    assert client.get("/api/meta").json()["models"][0]["evaluation"] == measured


def test_meta_exposes_guidance_fields_and_leaves_missing_ones_null(client, monkeypatch):
    fields = ("summary", "size", "use_when", "avoid_when", "members", "priority", "license", "release_date", "backend", "default")
    bare = client.get("/api/meta").json()["models"][0]
    assert all(key in bare for key in fields)
    assert bare["default"] is True and bare["summary"] is None and bare["size"] is None and bare["members"] is None
    guidance = {"summary": "The fake one.", "size": {"params": "4B", "download_gb": 8.0, "memory_gb": 9.0},
                "use_when": ["tests"], "avoid_when": ["production"], "members": ["a", "b"]}
    monkeypatch.setattr(FakeEngine.models["fake"], "options", guidance)
    full = client.get("/api/meta").json()["models"][0]
    assert {k: full[k] for k in guidance} == guidance


def test_the_portal_is_off_unless_asked_for(tmp_path):
    api_only = TestClient(create_app(FakeEngine(), db_path=str(tmp_path / "a.db")))
    assert api_only.get("/").json()["api"] == "/v1/systemone" and "--ui" in api_only.get("/").json()["ui"]
    assert api_only.get("/api/projects").status_code == 404 and api_only.get("/static/app.js").status_code == 404
    assert api_only.post("/v1/systemone", json={"state": "an invoice", "questions": {"q": {"type": "noul", "instructions": "Invoice?"}}}).status_code == 200
    assert not (tmp_path / "a.db").exists()                                 # no project database without the portal
    with_ui = TestClient(create_app(FakeEngine(), db_path=str(tmp_path / "b.db"), ui=True))
    assert with_ui.get("/").headers["content-type"].startswith("text/html") and with_ui.get("/api/projects").status_code == 200
