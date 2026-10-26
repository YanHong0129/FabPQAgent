"""DAY 2 — YOU WRITE THIS. The agent's tools: JSON schemas the model sees + the code that runs them.

Tests: tests/test_tools.py

Same rule as CS329z P5/P6: the model never touches your code. It writes a call as text; you
validate it, run it, and return a STRING observation. Errors are observations too, never
exceptions: "Error: lot 99999 does not exist" lets the model fix its own mistake.

Keep observations small (every token in an observation is paid for on every later turn): summarise, round floats to
3 decimals, never dump all 590 sensors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fabpqagent.model import TrainedModel
from fabpqagent.retrieval import NoteIndex

MAX_OBSERVATION_CHARS = 2000


@dataclass(frozen=True)
class ToolContext:
    lots: pd.DataFrame       # full lots table (db.load_lots())
    model: TrainedModel
    notes: NoteIndex
    baseline_lots: int = 300 # first N lots (by time) define SPC control limits


# One schema is done for you as a template. Write the other four the same way.
QUERY_LOTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_lots",
        "description": (
            "Run a read-only SQL SELECT on table lots(lot_id INTEGER, ts TEXT 'YYYY-MM-DD HH:MM:SS', "
            "failed INTEGER 0/1, s000..s589 REAL sensor readings). Returns at most 200 rows as CSV. "
            "Use for counts, fail rates over time, and finding lots."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["sql"],
        },
    },
}

# TODO: PREDICT_FAIL_SCHEMA   args: lot_id (integer)       -> fail probability + flagged yes/no + actual label
# TODO: TOP_FEATURES_SCHEMA   args: lot_id (integer), k?   -> model.top_features() as compact text
# TODO: SPC_CHECK_SCHEMA      args: sensor (string "s059") -> spc.spc_check over all lots in time order,
#                                                             baseline = first ctx.baseline_lots lots
# TODO: SEARCH_NOTES_SCHEMA   args: query (string)         -> top-3 note titles + snippets

TOOL_SCHEMAS: list[dict[str, Any]] = [QUERY_LOTS_SCHEMA]  # TODO: add the other four


def execute(name: str, arguments: str, ctx: ToolContext) -> str:
    """Run tool `name` with the model's raw JSON `arguments`. ALWAYS returns a string.

    Must handle, each as an "Error: ..." observation (see tests):
      - arguments that are not valid JSON
      - unknown tool name
      - missing / wrong-typed required argument
      - lot_id or sensor that does not exist
      - db.QueryRejected from query_lots
    Truncate any observation longer than MAX_OBSERVATION_CHARS and say that you did.
    """
    raise NotImplementedError
