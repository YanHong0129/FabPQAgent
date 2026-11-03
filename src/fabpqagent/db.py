"""Read-only access to the SECOM SQLite database.

The agent can send SQL written by the model, so `run_query` is a guarded gate:
SELECT/WITH only, one statement, read-only connection, and a row cap.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from fabpqagent.data import DB_PATH

MAX_ROWS = 200
_ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class QueryRejected(ValueError):
    """Raised for SQL the agent is not allowed to run. The message is safe to show the model."""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} missing. Run: uv run python -m fabpqagent.data")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_lots(db_path: Path = DB_PATH) -> pd.DataFrame:
    """All lots as a DataFrame: lot_id, ts, failed, s000..s589."""
    with connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM lots ORDER BY lot_id", conn)


def run_query(sql: str, db_path: Path = DB_PATH, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    statement = sql.strip().rstrip(";")
    if ";" in statement:
        raise QueryRejected("Only one SQL statement is allowed.")
    if not _ALLOWED_START.match(statement):
        raise QueryRejected("Only SELECT (or WITH ... SELECT) queries are allowed.")
    with connect(db_path) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM ({statement}) LIMIT {max_rows + 1}", conn)
        except (sqlite3.Error, pd.errors.DatabaseError) as err:
            raise QueryRejected(f"SQL error: {err}") from err
    if len(df) > max_rows:
        df = df.head(max_rows)
        df.attrs["truncated"] = True
    return df
