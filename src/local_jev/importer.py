"""Turn uploaded files or pasted text into items. Each item's state is a string or a JSON object."""
from __future__ import annotations

import csv
import email
import email.policy
import io
import json
import mailbox
import re
import tempfile

TITLE_KEYS = ("subject", "title", "name", "headline", "summary", "id")
MAX_ITEMS = 50_000


def title_of(state) -> str:
    if isinstance(state, dict):
        for key in TITLE_KEYS:
            for k, v in state.items():
                if k.lower() == key and isinstance(v, (str, int, float)) and str(v).strip():
                    return str(v).strip()[:140]
        state = " · ".join(str(v) for v in state.values() if isinstance(v, (str, int, float)))
    elif isinstance(state, list):
        state = " · ".join(str(v) for v in state)
    return re.sub(r"\s+", " ", str(state)).strip()[:140]


def _email_state(msg) -> dict:
    body = msg.get_body(preferencelist=("plain", "html")) if hasattr(msg, "get_body") else None
    text = ""
    if body is not None:
        try:
            text = body.get_content()
        except Exception:
            text = str(body.get_payload(decode=True) or b"", "utf-8", "replace")
        if body.get_content_type() == "text/html":
            text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
    state = {"from": str(msg.get("From", "")), "to": str(msg.get("To", "")), "subject": str(msg.get("Subject", "")),
             "date": str(msg.get("Date", "")), "body": re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()}
    return {k: v for k, v in state.items() if v}


def _clean_row(row: dict):
    row = {str(k).strip(): v.strip() if isinstance(v, str) else v for k, v in row.items() if k is not None}
    row = {k: v for k, v in row.items() if v not in ("", None)}
    if len(row) == 1:
        return next(iter(row.values()))
    return row or None


def parse(filename: str, content: str, split: str = "auto") -> list:
    """Return a list of states. ``split`` controls plain text: auto | lines | paragraphs | whole."""
    name = (filename or "").lower()
    states: list = []
    if name.endswith((".csv", ".tsv")):
        dialect = "excel-tab" if name.endswith(".tsv") else "excel"
        states = [_clean_row(r) for r in csv.DictReader(io.StringIO(content), dialect=dialect)]
    elif name.endswith((".jsonl", ".ndjson")):
        states = [json.loads(line) for line in content.splitlines() if line.strip()]
    elif name.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        states = list(data)
    elif name.endswith(".eml"):
        states = [_email_state(email.message_from_string(content, policy=email.policy.default))]
    elif name.endswith(".mbox"):
        with tempfile.NamedTemporaryFile("w", suffix=".mbox", delete=True, encoding="utf-8") as tmp:
            tmp.write(content)
            tmp.flush()
            box = mailbox.mbox(tmp.name, factory=lambda f: email.message_from_binary_file(f, policy=email.policy.default))
            states = [_email_state(m) for m in box]
    else:
        text = content.replace("\r\n", "\n").strip()
        if split == "auto":
            split = "paragraphs" if re.search(r"\n\s*\n", text) else "lines"
        if split == "whole":
            states = [text]
        elif split == "paragraphs":
            states = [p.strip() for p in re.split(r"\n\s*\n", text)]
        else:
            states = [line.strip() for line in text.split("\n")]
    cleaned = []
    for state in states:
        if isinstance(state, dict):
            state = _clean_row(state) if all(not isinstance(v, (dict, list)) for v in state.values()) else state
        if isinstance(state, (int, float)):
            state = str(state)
        if state in (None, "", [], {}):
            continue
        cleaned.append(state)
    if len(cleaned) > MAX_ITEMS:
        raise ValueError(f"That file has {len(cleaned):,} items; the limit per import is {MAX_ITEMS:,}.")
    return cleaned
