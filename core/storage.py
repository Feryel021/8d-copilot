"""
storage.py
----------
Small SQLite database so analyses are not lost when the page reloads.
Each saved row = one complaint with its classification, root cause and 8D.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created TEXT, complaint TEXT, category TEXT, severity TEXT,
        classification TEXT, rca TEXT, report TEXT, model TEXT)""")
    return con


def save(complaint: str, classification: dict, rca: dict | None,
         report: str | None, model: str) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO analyses (created, complaint, category, severity, classification, rca, report, model) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), complaint,
             classification.get("category"), classification.get("severity"),
             json.dumps(classification), json.dumps(rca) if rca else None, report, model),
        )
        return cur.lastrowid


def load_all() -> list[dict]:
    with _conn() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM analyses ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def delete_all() -> None:
    with _conn() as con:
        con.execute("DELETE FROM analyses")
