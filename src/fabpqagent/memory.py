"""PROVIDED. Long-term memory of past investigations, stored as a JSON list.

You decide (Day 2) what the agent saves and how past cases are fed back in, e.g. inject the
3 most recent cases that mention the same lot or sensor into the system prompt.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from fabpqagent.data import ROOT

MEMORY_PATH = ROOT / "data" / "memory.json"


@dataclass(frozen=True)
class Case:
    question: str
    finding: str            # one-line conclusion
    lots: tuple[int, ...]
    sensors: tuple[str, ...]
    created_at: str


def load_cases(path: Path | None = None) -> list[Case]:
    path = path or MEMORY_PATH  # resolved at call time so tests can monkeypatch MEMORY_PATH
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [Case(r["question"], r["finding"], tuple(r["lots"]), tuple(r["sensors"]), r["created_at"]) for r in raw]


def save_case(question: str, finding: str, lots: list[int], sensors: list[str],
              path: Path | None = None) -> Case:
    path = path or MEMORY_PATH
    case = Case(question, finding, tuple(lots), tuple(sensors), datetime.now().isoformat(timespec="seconds"))
    cases = [*load_cases(path), case]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(c) for c in cases], indent=1))
    os.replace(tmp, path)
    return case
