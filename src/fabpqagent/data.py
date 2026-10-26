"""PROVIDED. Downloads the UCI SECOM dataset and loads it into SQLite.

SECOM: 1,567 production lots from a semiconductor fab, 590 anonymised sensor readings each,
and a pass/fail label from in-house line testing (104 fails, about 6.6%).
Source: https://archive.ics.uci.edu/dataset/179/secom

Run once:  uv run python -m fabpqagent.data
"""
from __future__ import annotations

import sqlite3
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "secom.db"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
FILES = ("secom.data", "secom_labels.data")
N_SENSORS = 590


def sensor_columns() -> list[str]:
    return [f"s{i:03d}" for i in range(N_SENSORS)]


def download(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw_dir / name
        if target.exists():
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, target)


def _parse_value(token: str) -> float | None:
    return None if token == "NaN" else float(token)


def build_db(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> int:
    """Create table `lots(lot_id, ts, failed, s000..s589)`. Returns number of rows."""
    sensors = (raw_dir / "secom.data").read_text().strip().splitlines()
    labels = (raw_dir / "secom_labels.data").read_text().strip().splitlines()
    if len(sensors) != len(labels):
        raise ValueError(f"row mismatch: {len(sensors)} sensor rows vs {len(labels)} labels")

    cols = sensor_columns()
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE lots (lot_id INTEGER PRIMARY KEY, ts TEXT NOT NULL, failed INTEGER NOT NULL, "
            + ", ".join(f"{c} REAL" for c in cols) + ")"
        )
        rows = []
        for lot_id, (sensor_line, label_line) in enumerate(zip(sensors, labels)):
            values = [_parse_value(t) for t in sensor_line.split()]
            if len(values) != N_SENSORS:
                raise ValueError(f"lot {lot_id}: expected {N_SENSORS} values, got {len(values)}")
            label, ts = label_line.split(" ", 1)
            # SECOM: -1 = pass, 1 = fail. Timestamps are "19/07/2008 11:55:00" -> ISO for sortable SQL.
            day, month, rest = ts.strip().strip('"').split("/", 2)
            year, clock = rest.split(" ")
            rows.append((lot_id, f"{year}-{month}-{day} {clock}", int(label == "1"), *values))
        placeholders = ", ".join("?" * (3 + N_SENSORS))
        conn.executemany(f"INSERT INTO lots VALUES ({placeholders})", rows)
        conn.execute("CREATE INDEX idx_lots_ts ON lots(ts)")
    return len(rows)


if __name__ == "__main__":
    download()
    n = build_db()
    print(f"Loaded {n} lots into {DB_PATH}")
