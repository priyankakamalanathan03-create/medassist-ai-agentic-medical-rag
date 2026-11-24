# analytics.py – lightweight SQLite logging for interview demo
# Stores each query with basic metadata, used by the /api/analytics endpoint.
# Works offline – no external DB required.

import os
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Dict, Any

# SQLite DB file lives alongside this module.
DB_PATH = os.path.join(os.path.dirname(__file__), "analytics.db")

def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the analytics DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_schema() -> None:
    """Create the query_log table if it does not exist yet."""
    with closing(_get_connection()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                model TEXT,
                token_count INTEGER,
                intent TEXT,
                quality INTEGER
            )
            """
        )
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(query_log)").fetchall()
        }
        for column in ("answer_score", "prompt_score"):
            if column not in columns:
                conn.execute(f"ALTER TABLE query_log ADD COLUMN {column} INTEGER")
        conn.commit()

# Ensure schema at import time.
_ensure_schema()

def log_query(
    *,
    session_id: str,
    query: str,
    model: str | None = None,
    token_count: int | None = None,
    intent: str | None = None,
    quality: int | None = None,
    answer_score: int | None = None,
    prompt_score: int | None = None,
) -> None:
    """Insert a single query record.

    All arguments are optional except *session_id* and *query*.
    Errors are printed but do not raise, keeping the API robust.
    """
    try:
        with closing(_get_connection()) as conn:
            conn.execute(
                "INSERT INTO query_log "
                "(session_id, timestamp, query, model, token_count, intent, quality, answer_score, prompt_score)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    query,
                    model,
                    token_count,
                    intent,
                    quality,
                    answer_score,
                    prompt_score,
                ),
            )
            conn.commit()
    except Exception as e:  # pragma: no cover – defensive logging
        print(f"[Analytics] Failed to log query: {e}")

def get_stats() -> Dict[str, Any]:
    """Return aggregated analytics used by the front‑end dashboard.

    Returns a dict with:
        * total_queries – int
        * avg_token_count – float or None
        * top_intents – list of (intent, count) tuples (max 3)
    """
    with closing(_get_connection()) as conn:
        cur = conn.execute("SELECT COUNT(*) AS cnt FROM query_log")
        total = cur.fetchone()["cnt"]

        cur = conn.execute("SELECT AVG(token_count) AS avg FROM query_log WHERE token_count IS NOT NULL")
        avg = cur.fetchone()["avg"]

        cur = conn.execute(
            """
            SELECT intent, COUNT(*) AS cnt FROM query_log
            WHERE intent IS NOT NULL GROUP BY intent ORDER BY cnt DESC LIMIT 3
            """
        )
        top_intents = [(row["intent"], row["cnt"]) for row in cur.fetchall()]
        cur = conn.execute(
            """
            SELECT quality, COUNT(*) AS cnt FROM query_log
            WHERE quality IS NOT NULL AND answer_score IS NOT NULL
            GROUP BY quality
            """
        )
        quality_counts = {row["quality"]: row["cnt"] for row in cur.fetchall()}
        cur = conn.execute(
            "SELECT AVG(answer_score) AS avg_answer, AVG(prompt_score) AS avg_prompt FROM query_log"
        )
        score_averages = cur.fetchone()

    return {
        "total_queries": total,
        "avg_token_count": round(avg, 2) if avg is not None else None,
        "top_intents": top_intents,
        "avg_answer_score": (
            round(score_averages["avg_answer"], 1)
            if score_averages["avg_answer"] is not None else None
        ),
        "avg_prompt_score": (
            round(score_averages["avg_prompt"], 1)
            if score_averages["avg_prompt"] is not None else None
        ),
        "quality_breakdown": {
            "high": quality_counts.get(3, 0),
            "moderate": quality_counts.get(2, 0),
            "low": quality_counts.get(1, 0),
        },
    }
