"""The project store, including its one-time migration for databases from before verdict columns."""
import json
import sqlite3

from local_jev.store import Store, verdict_of

OLD_ANSWERS = """CREATE TABLE answers (
    item_id INTEGER NOT NULL, question_id INTEGER NOT NULL, model TEXT NOT NULL, spec_hash TEXT NOT NULL,
    result TEXT NOT NULL, created_at REAL, PRIMARY KEY (item_id, question_id, model))"""


def test_verdict_of_keeps_only_what_buckets_need():
    assert verdict_of({"type": "choice", "choice": "billing", "confidence": 0.9}) == ("billing", None)
    assert verdict_of({"type": "noul", "noul": 0.73}) == (None, 0.73)
    assert verdict_of({"type": "score", "score": 1.4, "confidence": 0.2}) == (None, 1.4)


def test_old_databases_are_backfilled_once(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute(OLD_ANSWERS)
    db.execute("INSERT INTO answers VALUES (1, 2, 'm', 'h', ?, 0)", (json.dumps({"type": "noul", "noul": 0.8}),))
    db.execute("INSERT INTO answers VALUES (1, 3, 'm', 'h', ?, 0)", (json.dumps({"type": "choice", "choice": "a"}),))
    db.commit()
    db.close()
    store = Store(path)
    rows = [tuple(r) for r in store.db.execute("SELECT question_id, verdict, value FROM answers ORDER BY question_id")]
    assert rows == [(2, None, 0.8), (3, "a", None)]
    Store(path)                                                  # opening again is a no-op
    assert store.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 2


def test_item_queries_skip_decoding_until_asked(tmp_path):
    store = Store(tmp_path / "s.db")
    pid = store.create_project("p")
    store.add_items(pid, [("a", {"body": "first"}), ("b", "second"), ("c", ["third"])])
    ids = [row[0] for row in store.item_ids(pid)]
    assert len(ids) == 3 and [row[0] for row in store.item_ids(pid, limit=2)] == ids[:2]     # newest first
    assert [row[1] for row in store.item_ids(pid, with_text=True)][0] == json.dumps(["third"])
    assert [i["state"] for i in store.items_by_id(ids[::-1])] == [{"body": "first"}, "second", ["third"]]
