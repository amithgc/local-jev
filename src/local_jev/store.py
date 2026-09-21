"""SQLite persistence for projects, questions, items and answers."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, description TEXT DEFAULT '', noun TEXT DEFAULT 'items',
    settings TEXT DEFAULT '{}', created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    position INTEGER DEFAULT 0, name TEXT NOT NULL, type TEXT NOT NULL, instructions TEXT DEFAULT '',
    criteria TEXT DEFAULT 'null', threshold REAL DEFAULT 0.5, enabled INTEGER DEFAULT 1, spec_hash TEXT);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    position INTEGER DEFAULT 0, title TEXT, state TEXT NOT NULL, created_at REAL);
CREATE TABLE IF NOT EXISTS answers (
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    model TEXT NOT NULL, spec_hash TEXT NOT NULL, result TEXT NOT NULL, created_at REAL,
    verdict TEXT, value REAL,
    PRIMARY KEY (item_id, question_id, model));
CREATE INDEX IF NOT EXISTS idx_items_project ON items(project_id, position);
CREATE INDEX IF NOT EXISTS idx_questions_project ON questions(project_id, position);
CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id, model);
"""


def wire_question(q: dict) -> dict:
    """The Jev wire-format question for a stored row. Name and threshold are UI-only."""
    wire = {"type": q["type"], "instructions": q.get("instructions") or None}
    criteria = q.get("criteria")
    if q["type"] == "noul":
        criteria = {k: v for k, v in (criteria or {}).items() if v}
        if criteria:
            wire["criteria"] = criteria
    else:
        wire["criteria"] = criteria
    return wire


def verdict_of(result: dict) -> tuple:
    """The part of an answer that decides its bucket, stored in plain columns so aggregating never decodes JSON:
    a choice's key, a yes/no's probability of yes, or a score's expected level."""
    kind = result.get("type")
    if kind == "choice":
        return result.get("choice"), None
    if kind == "noul":
        return None, float(result["noul"])
    return None, float(result["score"])


def spec_hash(q: dict) -> str:
    """Identifies what was actually asked. Renaming or moving the threshold keeps old answers valid."""
    return hashlib.sha1(json.dumps(wire_question(q), sort_keys=True).encode()).hexdigest()[:16]


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Databases created before answers had verdict/value columns: add and backfill them once."""
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(answers)")}
        if "verdict" in columns:
            return
        self.db.execute("ALTER TABLE answers ADD COLUMN verdict TEXT")
        self.db.execute("ALTER TABLE answers ADD COLUMN value REAL")
        rows = self.db.execute("SELECT rowid, result FROM answers").fetchall()
        self.db.executemany("UPDATE answers SET verdict = ?, value = ? WHERE rowid = ?",
                            [(*verdict_of(json.loads(result)), rowid) for rowid, result in rows])
        self.db.commit()

    def run(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def all(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.all(sql, args)
        return rows[0] if rows else None

    # -- projects -----------------------------------------------------------
    def projects(self):
        return self.all("""SELECT p.*, (SELECT COUNT(*) FROM items i WHERE i.project_id = p.id) AS item_count
                           FROM projects p ORDER BY p.id""")

    def project(self, pid):
        row = self.one("""SELECT p.*, (SELECT COUNT(*) FROM items i WHERE i.project_id = p.id) AS item_count,
                          (SELECT MAX(created_at) FROM items i WHERE i.project_id = p.id) AS items_updated_at
                          FROM projects p WHERE p.id = ?""", (pid,))
        if row:
            row["settings"] = json.loads(row["settings"] or "{}")
        return row

    def create_project(self, name, description="", noun="items", settings=None):
        now = time.time()
        return self.run("INSERT INTO projects(name, description, noun, settings, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                        (name, description, noun, json.dumps(settings or {}), now, now)).lastrowid

    def update_project(self, pid, **fields):
        if "settings" in fields:
            fields["settings"] = json.dumps(fields["settings"])
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.run(f"UPDATE projects SET {sets} WHERE id = ?", (*fields.values(), pid))

    def delete_project(self, pid):
        self.run("DELETE FROM projects WHERE id = ?", (pid,))

    # -- questions ----------------------------------------------------------
    @staticmethod
    def _question(row):
        row["criteria"] = json.loads(row["criteria"]) if row["criteria"] else None
        row["enabled"] = bool(row["enabled"])
        return row

    def questions(self, pid):
        return [self._question(r) for r in
                self.all("SELECT * FROM questions WHERE project_id = ? ORDER BY position, id", (pid,))]

    def question(self, qid):
        row = self.one("SELECT * FROM questions WHERE id = ?", (qid,))
        return self._question(row) if row else None

    def save_question(self, pid, data, qid=None):
        fields = {"name": data["name"], "type": data["type"], "instructions": data.get("instructions") or "",
                  "criteria": json.dumps(data.get("criteria")), "threshold": float(data.get("threshold", 0.5)),
                  "enabled": int(bool(data.get("enabled", True))), "spec_hash": spec_hash(data)}
        if qid is None:
            position = (self.one("SELECT COALESCE(MAX(position), -1) + 1 AS p FROM questions WHERE project_id = ?", (pid,)))["p"]
            cols = ", ".join(fields)
            return self.run(f"INSERT INTO questions(project_id, position, {cols}) VALUES (?, ?, {', '.join('?' * len(fields))})",
                            (pid, position, *fields.values())).lastrowid
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.run(f"UPDATE questions SET {sets} WHERE id = ?", (*fields.values(), qid))
        return qid

    def set_enabled(self, qid, enabled):
        self.run("UPDATE questions SET enabled = ? WHERE id = ?", (int(enabled), qid))

    def delete_question(self, qid):
        self.run("DELETE FROM questions WHERE id = ?", (qid,))

    # -- items --------------------------------------------------------------
    def add_items(self, pid, items):
        now = time.time()
        with self.lock:
            start = self.db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM items WHERE project_id = ?", (pid,)).fetchone()[0]
            self.db.executemany("INSERT INTO items(project_id, position, title, state, created_at) VALUES (?,?,?,?,?)",
                                [(pid, start + n, title, json.dumps(state, ensure_ascii=False), now)
                                 for n, (title, state) in enumerate(items)])
            self.db.commit()
        return len(items)

    def clear_items(self, pid):
        self.run("DELETE FROM items WHERE project_id = ?", (pid,))

    def delete_item(self, item_id):
        self.run("DELETE FROM items WHERE id = ?", (item_id,))

    def items(self, pid, limit=None):
        """The most recently added ``limit`` items, newest first."""
        sql = "SELECT * FROM items WHERE project_id = ? ORDER BY position DESC, id DESC"
        rows = self.all(sql + (" LIMIT ?" if limit else ""), (pid, limit) if limit else (pid,))
        for row in rows:
            row["state"] = json.loads(row["state"])
        return rows

    # -- answers ------------------------------------------------------------
    def save_answers(self, rows):
        now = time.time()
        with self.lock:
            self.db.executemany("""INSERT OR REPLACE INTO answers(item_id, question_id, model, spec_hash, result, created_at, verdict, value)
                                   VALUES (?,?,?,?,?,?,?,?)""",
                                [(i, q, m, h, json.dumps(r), now, *verdict_of(r)) for i, q, m, h, r in rows])
            self.db.commit()

    _CURRENT = """FROM answers a JOIN questions q ON q.id = a.question_id
                  WHERE q.project_id = ? AND a.model = ? AND a.spec_hash = q.spec_hash"""

    def answers(self, pid, model, item_ids=None):
        """{item_id: {question_id: result}} for answers that still match their question's current wording.

        Decodes every answer's JSON: pass ``item_ids`` to limit it to the items actually being shown."""
        sql, args = f"SELECT a.item_id, a.question_id, a.result {self._CURRENT}", [pid, model]
        if item_ids is not None:
            if not item_ids:
                return {}
            sql += f" AND a.item_id IN ({','.join('?' * len(item_ids))})"
            args += list(item_ids)
        table: dict = {}
        with self.lock:
            for item_id, question_id, result in self.db.execute(sql, args):
                table.setdefault(item_id, {})[question_id] = json.loads(result)
        return table

    def verdicts(self, pid, model, limit=None):
        """{item_id: {question_id: (verdict, value)}}: enough to bucket, count and filter, with no JSON decoding.

        ``limit`` restricts it to the most recently added items, like ``item_ids``."""
        sql, args = f"SELECT a.item_id, a.question_id, a.verdict, a.value {self._CURRENT}", [pid, model]
        if limit:
            sql += " AND a.item_id IN (SELECT id FROM items WHERE project_id = ? ORDER BY position DESC, id DESC LIMIT ?)"
            args += [pid, limit]
        table: dict = {}
        with self.lock:
            for item_id, question_id, verdict, value in self.db.execute(sql, args):
                table.setdefault(item_id, {})[question_id] = (verdict, value)
        return table

    def item_ids(self, pid, limit=None, with_text=False):
        """The most recently added ``limit`` items, newest first, without decoding them.

        Rows are (id,) or, with ``with_text``, (id, stored JSON text) for searching."""
        sql = f"SELECT id{', state' if with_text else ''} FROM items WHERE project_id = ? ORDER BY position DESC, id DESC"
        with self.lock:
            return self.db.execute(sql + (" LIMIT ?" if limit else ""), (pid, limit) if limit else (pid,)).fetchall()

    def items_by_id(self, ids):
        """Full items for the given ids, in the given order."""
        if not ids:
            return []
        rows = {r["id"]: r for r in self.all(f"SELECT * FROM items WHERE id IN ({','.join('?' * len(ids))})", list(ids))}
        out = []
        for i in ids:
            row = rows[i]
            row["state"] = json.loads(row["state"])
            out.append(row)
        return out
