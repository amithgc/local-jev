"""The project API behind the UI: projects, questions, items, runs, results, export."""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .engine import Engine
from .models import ROOT
from .questions import ContextOverflow, QuestionError, parse_question
from .importer import parse, title_of
from .store import Store, verdict_of, wire_question

RUN_BATCH = 6                       # items per forward batch; small enough to keep progress lively
DEFAULT_STATE_TOKENS = 1024


class ProjectIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = ""
    noun: str = "items"
    settings: dict[str, Any] = {}


class QuestionIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    type: str
    instructions: str = ""
    criteria: Any = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    enabled: bool = True


class ImportIn(BaseModel):
    filename: str = "pasted.txt"
    content: str
    split: str = "auto"
    replace: bool = False


class RunIn(BaseModel):
    model: str | None = None
    scope: str = "unsorted"          # unsorted | all
    question_ids: list[int] | None = None
    limit: int | None = None


class TryIn(BaseModel):
    question: QuestionIn
    model: str | None = None
    n: int = Field(6, ge=1, le=24)


class AskIn(BaseModel):
    state: Any = None
    model: str | None = None
    question_ids: list[int] | None = None


def bucket_from(question: dict, verdict, value):
    """Which result row an answer falls into, from its stored verdict/value (see store.verdict_of):
    'yes'/'no' for a yes/no (at the question's current threshold), a choice key, or a level index."""
    if question["type"] == "noul":
        return "yes" if value >= question["threshold"] else "no"
    if question["type"] == "choice":
        return verdict
    levels = len(question["criteria"] or [])
    return str(min(max(int(round(value)), 0), max(levels - 1, 0)))


def bucket_of(question: dict, result: dict):
    """The same, from a full answer."""
    return bucket_from(question, *verdict_of(result))


class Runner:
    """One background run per project."""

    def __init__(self, store: Store, engine: Engine):
        self.store, self.engine = store, engine
        self.jobs: dict[int, dict] = {}
        self.lock = threading.Lock()

    def status(self, pid):
        job = self.jobs.get(pid)
        if not job:
            return {"state": "idle"}
        view = {k: v for k, v in job.items() if k != "cancel"}
        view["elapsed"] = round((job.get("finished_at") or time.time()) - job["started_at"], 2)
        return view

    def start(self, pid, model, questions, items, max_state_tokens):
        with self.lock:
            if self.jobs.get(pid, {}).get("state") == "running":
                raise HTTPException(409, "A run is already in progress for this project.")
            job = {"state": "running", "done": 0, "total": len(items), "questions": len(questions), "model": model,
                   "started_at": time.time(), "finished_at": None, "error": None, "truncated": 0,
                   "input_tokens": 0, "cancel": threading.Event()}
            self.jobs[pid] = job
        threading.Thread(target=self._work, args=(job, model, questions, items, max_state_tokens), daemon=True).start()
        return self.status(pid)

    def cancel(self, pid):
        job = self.jobs.get(pid)
        if job and job["state"] == "running":
            job["cancel"].set()

    def _work(self, job, model, questions, items, max_state_tokens):
        try:
            specs = {q["id"]: parse_question(wire_question(q)) for q in questions}
            hashes = {q["id"]: q["spec_hash"] for q in questions}
            for start in range(0, len(items), RUN_BATCH):
                if job["cancel"].is_set():
                    job["state"] = "cancelled"
                    break
                chunk = items[start:start + RUN_BATCH]
                # Each item only needs the questions it has no current answer for.
                groups: dict[tuple, list] = {}
                for item, missing in chunk:
                    groups.setdefault(tuple(missing), []).append(item)
                rows = []
                for missing, group in groups.items():
                    answers, stats = self.engine.answer_many(
                        model, [i["state"] for i in group], {qid: specs[qid] for qid in missing},
                        truncate=True, max_state_tokens=max_state_tokens)
                    for item, row, stat in zip(group, answers, stats):
                        rows += [(item["id"], qid, model, hashes[qid], result) for qid, result in row.items()]
                        job["truncated"] += int(stat["truncated"])
                        job["input_tokens"] += stat["input_tokens"]
                self.store.save_answers(rows)
                job["done"] = min(start + RUN_BATCH, len(items))
            else:
                job["state"] = "done"
        except Exception as err:                       # surface the failure in the UI rather than dying silently
            job["state"], job["error"] = "error", f"{type(err).__name__}: {err}"
        finally:
            job["finished_at"] = time.time()


def attach(app: FastAPI, engine: Engine, db_path: str | None = None):
    store = Store(db_path or os.environ.get("LOCAL_JEV_DB") or ROOT / "data" / "local-jev.db")
    runner = Runner(store, engine)
    app.state.store = store

    if not store.projects() and not os.environ.get("LOCAL_JEV_NO_SEED"):
        from .seed import seed
        seed(store)

    def project_or_404(pid):
        project = store.project(pid)
        if not project:
            raise HTTPException(404, "No such project.")
        return project

    def check_question(data: QuestionIn) -> dict:
        q = data.model_dump()
        try:
            parse_question(wire_question(q))
        except QuestionError as err:
            raise HTTPException(422, err.msg)
        return q

    def model_or_default(name):
        resolved = engine.resolve(name)
        if resolved is None:
            raise HTTPException(404, f"Unknown model '{name}'.")
        return resolved

    # -- meta -----------------------------------------------------------------
    @app.get("/api/meta", tags=["Projects"])
    def meta():
        # Cards are re-read on every call: evaluation numbers and guidance change when a benchmark is recorded
        # (evals/jevbench_cards.py), and that should not need a server restart. The model list itself is fixed.
        from .models import discover
        fresh = discover()

        def describe(m):
            card = fresh.get(m.name, m)
            extra = lambda key: card.options.get(key)                      # noqa: E731  (guidance keys live in the card's options)
            return {"name": m.name, "description": card.description, "backend": card.backend, "license": card.license,
                    "release_date": card.release_date, "priority": card.priority, "default": m.name == engine.default_model,
                    "unavailable": m.unavailable(), "evaluation": card.evaluation, "summary": extra("summary"),
                    "size": extra("size"), "use_when": extra("use_when"), "avoid_when": extra("avoid_when"),
                    "members": extra("members")}

        return {"version": app.version, "default_model": engine.default_model,
                "models": [describe(m) for m in engine.models.values()]}

    # -- projects -------------------------------------------------------------
    @app.get("/api/projects", tags=["Projects"])
    def list_projects():
        return store.projects()

    @app.post("/api/projects", tags=["Projects"])
    def create_project(body: ProjectIn):
        return project_or_404(store.create_project(body.name.strip(), body.description, body.noun or "items", body.settings))

    @app.get("/api/projects/{pid}", tags=["Projects"])
    def get_project(pid: int):
        project = project_or_404(pid)
        project["questions"] = store.questions(pid)
        return project

    @app.patch("/api/projects/{pid}", tags=["Projects"])
    def update_project(pid: int, body: ProjectIn):
        project_or_404(pid)
        store.update_project(pid, name=body.name.strip(), description=body.description, noun=body.noun or "items", settings=body.settings)
        return get_project(pid)

    @app.delete("/api/projects/{pid}", tags=["Projects"])
    def delete_project(pid: int):
        runner.cancel(pid)
        store.delete_project(pid)
        return {"ok": True}

    # -- questions ------------------------------------------------------------
    @app.post("/api/projects/{pid}/questions", tags=["Projects"])
    def create_question(pid: int, body: QuestionIn):
        project_or_404(pid)
        return store.question(store.save_question(pid, check_question(body)))

    @app.put("/api/questions/{qid}", tags=["Projects"])
    def update_question(qid: int, body: QuestionIn):
        existing = store.question(qid)
        if not existing:
            raise HTTPException(404, "No such question.")
        store.save_question(existing["project_id"], check_question(body), qid)
        return store.question(qid)

    @app.post("/api/questions/{qid}/enabled", tags=["Projects"])
    def toggle_question(qid: int, enabled: bool):
        store.set_enabled(qid, enabled)
        return {"ok": True}

    @app.delete("/api/questions/{qid}", tags=["Projects"])
    def delete_question(qid: int):
        store.delete_question(qid)
        return {"ok": True}

    # -- items ----------------------------------------------------------------
    @app.post("/api/projects/{pid}/items/import", tags=["Projects"])
    def import_items(pid: int, body: ImportIn):
        project_or_404(pid)
        try:
            states = parse(body.filename, body.content, body.split)
        except (ValueError, json.JSONDecodeError, csv.Error) as err:
            raise HTTPException(422, f"Could not read {body.filename}: {err}")
        if not states:
            raise HTTPException(422, "No items found in that input.")
        if body.replace:
            store.clear_items(pid)
        return {"added": store.add_items(pid, [(title_of(s), s) for s in states])}

    @app.delete("/api/projects/{pid}/items", tags=["Projects"])
    def clear_items(pid: int):
        runner.cancel(pid)
        store.clear_items(pid)
        return {"ok": True}

    def scoped(pid, limit):
        return store.items(pid, limit if limit and limit > 0 else None)

    def scoped_ids(pid, limit, with_text=False):
        """Item ids in scope (and optionally their stored text), without decoding any item."""
        return store.item_ids(pid, limit if limit and limit > 0 else None, with_text)

    # -- results --------------------------------------------------------------
    @app.get("/api/projects/{pid}/results", tags=["Projects"])
    def results(pid: int, model: str | None = None, limit: int | None = None):
        project_or_404(pid)
        name = model_or_default(model)
        questions = store.questions(pid)
        ids = [row[0] for row in scoped_ids(pid, limit)]
        table = store.verdicts(pid, name, limit if limit and limit > 0 else None)   # polled during runs: no JSON decoded
        summary, unsorted = [], 0
        enabled = [q["id"] for q in questions if q["enabled"]]
        for item_id in ids:
            have = table.get(item_id, {})
            if any(qid not in have for qid in enabled):
                unsorted += 1
        for q in questions:
            counts: dict[str, int] = {}
            total, answered = 0.0, 0
            for item_id in ids:
                found = table.get(item_id, {}).get(q["id"])
                if found is None:
                    continue
                answered += 1
                bucket = bucket_from(q, *found)
                counts[bucket] = counts.get(bucket, 0) + 1
                if q["type"] == "score":
                    total += found[1]
            if q["type"] == "noul":
                rows = [{"value": v, "label": v.capitalize(), "count": counts.get(v, 0)} for v in ("yes", "no")]
            elif q["type"] == "choice":
                keys = list((q["criteria"] or {}).keys())
                rows = sorted(({"value": k, "label": k, "count": counts.get(k, 0)} for k in keys),
                              key=lambda r: -r["count"])
            else:
                rows = [{"value": str(i), "label": f"{i + 1}. {level if isinstance(level, str) else json.dumps(level)}",
                         "count": counts.get(str(i), 0)} for i, level in enumerate(q["criteria"] or [])]
            entry = {"question_id": q["id"], "name": q["name"], "type": q["type"], "answered": answered, "rows": rows}
            if q["type"] == "score" and answered:
                entry["average"] = round(total / answered + 1, 1)
                entry["of"] = len(q["criteria"] or [])
            summary.append(entry)
        return {"model": name, "items": len(ids), "unsorted": unsorted, "questions": summary, "run": runner.status(pid)}

    @app.get("/api/projects/{pid}/items", tags=["Projects"])
    def list_items(pid: int, model: str | None = None, limit: int | None = None, filters: str = "[]",
                   offset: int = 0, page: int = 50, search: str = ""):
        project_or_404(pid)
        name = model_or_default(model)
        questions = {q["id"]: q for q in store.questions(pid)}
        wanted = [(int(f["q"]), str(f["v"])) for f in json.loads(filters or "[]")]
        term = search.strip().lower()
        table = store.verdicts(pid, name, limit if limit and limit > 0 else None) if wanted else {}
        matched = []
        for row in scoped_ids(pid, limit, with_text=bool(term)):
            if wanted:
                have = table.get(row[0], {})
                if not all(qid in have and qid in questions and bucket_from(questions[qid], *have[qid]) == value
                           for qid, value in wanted):
                    continue
            if term and term not in row[1].lower():          # the stored text is json.dumps(state, ensure_ascii=False)
                continue
            matched.append(row[0])
        shown = matched[offset: offset + page]
        answers = store.answers(pid, name, item_ids=shown)    # decode only what is displayed
        out = []
        for item in store.items_by_id(shown):
            have = answers.get(item["id"], {})
            out.append({"id": item["id"], "title": item["title"], "state": item["state"],
                        "answers": {str(qid): {**res, "bucket": bucket_of(questions[qid], res)}
                                    for qid, res in have.items() if qid in questions}})
        return {"total": len(matched), "offset": offset, "items": out}

    # -- running --------------------------------------------------------------
    @app.post("/api/projects/{pid}/run", tags=["Projects"])
    def run(pid: int, body: RunIn):
        project = project_or_404(pid)
        name = model_or_default(body.model)
        questions = [q for q in store.questions(pid)
                     if (q["id"] in body.question_ids if body.question_ids else q["enabled"])]
        if not questions:
            raise HTTPException(422, "Tick at least one question to run.")
        table = store.verdicts(pid, name, body.limit if body.limit and body.limit > 0 else None) if body.scope != "all" else {}
        pending = []
        for (item_id,) in scoped_ids(pid, body.limit):
            missing = [q["id"] for q in questions if q["id"] not in table.get(item_id, {})]
            if missing:
                pending.append((item_id, missing))
        items = {item["id"]: item for item in store.items_by_id([i for i, _ in pending])}
        work = [(items[item_id], missing) for item_id, missing in pending]
        if not work:
            return {"state": "idle", "message": "Nothing to do: everything is already sorted."}
        tokens = int(project["settings"].get("max_state_tokens") or DEFAULT_STATE_TOKENS)
        return runner.start(pid, name, questions, work, tokens)

    @app.get("/api/projects/{pid}/run", tags=["Projects"])
    def run_status(pid: int):
        return runner.status(pid)

    @app.post("/api/projects/{pid}/run/cancel", tags=["Projects"])
    def run_cancel(pid: int):
        runner.cancel(pid)
        return runner.status(pid)

    @app.post("/api/projects/{pid}/try", tags=["Projects"])
    def try_question(pid: int, body: TryIn):
        """Run a draft question on a few items without saving anything."""
        project = project_or_404(pid)
        name = model_or_default(body.model)
        q = check_question(body.question)
        items = store.items(pid, None)
        if not items:
            raise HTTPException(422, "Add some data first.")
        step = max(1, len(items) // body.n)
        sample = items[::step][: body.n]
        tokens = int(project["settings"].get("max_state_tokens") or DEFAULT_STATE_TOKENS)
        started = time.time()
        try:
            answers, _ = engine.answer_many(name, [i["state"] for i in sample], {"q": parse_question(wire_question(q))},
                                            truncate=True, max_state_tokens=tokens)
        except ContextOverflow as err:
            raise HTTPException(422, str(err))
        except RuntimeError as err:
            raise HTTPException(503, str(err))
        return {"elapsed": round(time.time() - started, 2), "model": name,
                "results": [{"id": i["id"], "title": i["title"], "answer": {**a["q"], "bucket": bucket_of(q, a["q"])}}
                            for i, a in zip(sample, answers)]}

    @app.post("/api/projects/{pid}/ask", tags=["Projects"])
    def ask(pid: int, body: AskIn):
        """Put one state to the project's questions and return the answers. Nothing is saved.

        This is the "Test one" view: every question of the project is asked, whether or not it is ticked
        for batch runs, because the tick belongs to "Sort many".
        """
        project = project_or_404(pid)
        name = model_or_default(body.model)
        state = body.state.strip() if isinstance(body.state, str) else body.state
        if state in (None, "", [], {}):
            raise HTTPException(422, "Type or paste something to ask about.")
        questions = [q for q in store.questions(pid) if not body.question_ids or q["id"] in body.question_ids]
        if not questions:
            raise HTTPException(422, "This project has no questions yet. Add one first.")
        tokens = int(project["settings"].get("max_state_tokens") or DEFAULT_STATE_TOKENS)
        started = time.time()
        try:
            answers, stats = engine.answer_many(name, [state], {str(q["id"]): parse_question(wire_question(q)) for q in questions},
                                                truncate=True, max_state_tokens=tokens)
        except ContextOverflow as err:
            raise HTTPException(422, str(err))
        except RuntimeError as err:
            raise HTTPException(503, str(err))
        return {"model": name, "elapsed_ms": round((time.time() - started) * 1000), "input_tokens": stats[0]["input_tokens"],
                "truncated": bool(stats[0]["truncated"]),
                "answers": {str(q["id"]): {**answers[0][str(q["id"])], "bucket": bucket_of(q, answers[0][str(q["id"])])} for q in questions}}

    # -- export ---------------------------------------------------------------
    @app.get("/api/projects/{pid}/export.csv", tags=["Projects"])
    def export(pid: int, model: str | None = None, limit: int | None = None, filters: str = "[]"):
        project = project_or_404(pid)
        name = model_or_default(model)
        questions = store.questions(pid)
        listing = list_items(pid, name, limit, filters, 0, 10**9)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        header = ["id", "title", "content"]
        for q in questions:
            header += [q["name"], f"{q['name']} (confidence)" if q["type"] != "noul" else f"{q['name']} (probability)"]
        writer.writerow(header)
        for item in listing["items"]:
            state = item["state"]
            row = [item["id"], item["title"], state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)]
            for q in questions:
                res = item["answers"].get(str(q["id"]))
                if not res:
                    row += ["", ""]
                elif q["type"] == "noul":
                    row += [res["bucket"], res["noul"]]
                elif q["type"] == "choice":
                    row += [res["choice"], res["confidence"]]
                else:
                    row += [round(res["score"] + 1, 2), res["confidence"]]
            writer.writerow(row)
        filename = "".join(c if c.isalnum() else "-" for c in project["name"]).strip("-").lower() or "export"
        return Response(buffer.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'})
