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

from fabpqagent.db import QueryRejected, run_query
from fabpqagent.model import TrainedModel, predict_fail, top_features
from fabpqagent.retrieval import NoteIndex
from fabpqagent.spc import spc_check

MAX_OBSERVATION_CHARS = 2000
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
NOTE_HITS = 3
LISTED_LOTS = 10   # how many lot ids an SPC observation lists before summarising


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

PREDICT_FAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_fail",
        "description": (
            "Fail probability for one lot from the trained classifier, whether it is above the "
            "flagging threshold, and the lot's actual pass/fail label."
        ),
        "parameters": {
            "type": "object",
            "properties": {"lot_id": {"type": "integer", "description": "Lot id, 0-1566."}},
            "required": ["lot_id"],
        },
    },
}

TOP_FEATURES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "top_features",
        "description": (
            "The sensors that look most unusual on one lot: among the model's most important "
            "sensors, those furthest from the pass-lot mean (z-score). Use to explain a failure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lot_id": {"type": "integer", "description": "Lot id, 0-1566."},
                "k": {"type": "integer", "description": f"How many sensors to return (1-{MAX_TOP_K}, default {DEFAULT_TOP_K})."},
            },
            "required": ["lot_id"],
        },
    },
}

SPC_CHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "spc_check",
        "description": (
            "Statistical process control check for one sensor over all lots in time order. Control "
            "limits (mean +/- 3 sigma) come from the first lots as a baseline. Reports points beyond "
            "the limits (rule 1) and runs of 8 on one side of the centre (a shift)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sensor": {"type": "string", "description": "Sensor name like 's059' (s000-s589)."}},
            "required": ["sensor"],
        },
    },
}

SEARCH_NOTES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": (
            "Keyword search (BM25) over the fab's troubleshooting notes: likely causes and what to "
            "check for a sensor or symptom. Pass keywords such as sensor IDs and symptoms "
            "('s510 high'), not a full sentence. Returns the top 3 notes with snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords, e.g. 's059 drift'."}},
            "required": ["query"],
        },
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    QUERY_LOTS_SCHEMA, PREDICT_FAIL_SCHEMA, TOP_FEATURES_SCHEMA, SPC_CHECK_SCHEMA, SEARCH_NOTES_SCHEMA,
]


class ToolError(ValueError):
    """A problem with the call that the model can fix. The message is shown to the model."""


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
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(_HANDLERS)}."
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as err:
        return f"Error: arguments are not valid JSON ({err.msg}). Send an object like {{\"lot_id\": 2}}."
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object."
    try:
        observation = handler(args, ctx)
    except (ToolError, QueryRejected) as err:
        return f"Error: {err}"
    except Exception as err:  # safety net: a bug in a tool becomes an observation, never a crash
        return f"Error: {name} failed unexpectedly ({type(err).__name__}: {err})."
    return _cap(observation)


def _cap(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return (text[:MAX_OBSERVATION_CHARS] + f"\n[truncated: showing {MAX_OBSERVATION_CHARS} of {len(text)} "
            "characters. Ask for fewer columns or rows.]")


# --- argument checks ---------------------------------------------------------------------------

def _int_arg(args: dict[str, Any], key: str, default: int | None = None) -> int:
    value = args.get(key, default)
    if value is None:
        raise ToolError(f"missing required argument '{key}' (an integer).")
    if isinstance(value, bool) or not isinstance(value, int):  # bool is a subclass of int in Python
        raise ToolError(f"'{key}' must be an integer, got {value!r}.")
    return value


def _str_arg(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"missing required argument '{key}' (a non-empty string).")
    return value.strip()


def _lot(ctx: ToolContext, lot_id: int) -> pd.Series:
    rows = ctx.lots[ctx.lots["lot_id"] == lot_id]
    if rows.empty:
        first, last = ctx.lots["lot_id"].min(), ctx.lots["lot_id"].max()
        raise ToolError(f"lot {lot_id} does not exist. Valid lot_id range is {first}-{last}.")
    return rows.iloc[0]


def _label(lot: pd.Series) -> str:
    return "fail" if lot["failed"] == 1 else "pass"


# --- the five tools: each takes checked-in args and returns a short string ----------------------

def _query_lots(args: dict[str, Any], ctx: ToolContext) -> str:
    df = run_query(_str_arg(args, "sql"))
    note = " (row cap reached: first 200 rows only)" if df.attrs.get("truncated") else ""
    return f"{len(df)} rows{note}\n{df.to_csv(index=False, float_format='%.3f')}"


def _predict_fail(args: dict[str, Any], ctx: ToolContext) -> str:
    lot = _lot(ctx, _int_arg(args, "lot_id"))
    proba = predict_fail(ctx.model, lot)
    flagged = "flagged" if proba > ctx.model.threshold else "not flagged"
    return (f"Lot {lot['lot_id']}: fail probability {proba:.3f} "
            f"(threshold {ctx.model.threshold:.3f} -> {flagged}). Actual label: {_label(lot)}.")


def _top_features(args: dict[str, Any], ctx: ToolContext) -> str:
    lot = _lot(ctx, _int_arg(args, "lot_id"))
    k = _int_arg(args, "k", default=DEFAULT_TOP_K)
    if not 1 <= k <= MAX_TOP_K:
        raise ToolError(f"'k' must be between 1 and {MAX_TOP_K}, got {k}.")
    lines = [f"{f['sensor']}: value {f['value']:.3f} vs pass mean {f['pass_mean']:.3f} (z = {f['z']:+.2f})"
             for f in top_features(ctx.model, lot, k=k)]
    header = f"Lot {lot['lot_id']} (actual: {_label(lot)}), most unusual of the model's important sensors:"
    return "\n".join([header, *lines])


def _spc_check(args: dict[str, Any], ctx: ToolContext) -> str:
    sensor = _str_arg(args, "sensor").lower()
    if sensor not in ctx.lots.columns or not sensor.startswith("s"):
        raise ToolError(f"sensor '{sensor}' does not exist. Sensor names look like 's059' (s000-s589).")
    series = ctx.lots.sort_values("ts", kind="stable").set_index("lot_id")[sensor]  # index = lot_id
    try:
        result = spc_check(sensor, series, baseline=series.iloc[:ctx.baseline_lots])
    except ValueError as err:  # e.g. a sensor with almost no readings in the baseline
        raise ToolError(str(err)) from err
    status = "in control" if result.in_control else "out of control"
    return (f"{sensor} SPC, baseline = first {ctx.baseline_lots} lots: centre {result.center:.3f}, "
            f"UCL {result.ucl:.3f}, LCL {result.lcl:.3f}. Status: {status}.\n"
            f"Rule 1 (beyond 3 sigma): {_lots_summary(result.beyond_limits)}.\n"
            f"Rule of eight (shift): {_lots_summary(result.shift_runs, 'runs ending at')}.")


def _lots_summary(lot_ids: list[int], prefix: str = "lots") -> str:
    if not lot_ids:
        return "none"
    shown = ", ".join(str(i) for i in lot_ids[:LISTED_LOTS])
    more = f" and {len(lot_ids) - LISTED_LOTS} more" if len(lot_ids) > LISTED_LOTS else ""
    return f"{len(lot_ids)} ({prefix} {shown}{more})"


def _search_notes(args: dict[str, Any], ctx: ToolContext) -> str:
    query = _str_arg(args, "query")
    hits = ctx.notes.search(query, k=NOTE_HITS)
    if not hits:
        return f"No notes matched '{query}'. Try sensor IDs or symptom keywords."
    return "\n\n".join(f"[{h.note_id}] {h.title} (score {h.score:.2f})\n{h.snippet}" for h in hits)


_HANDLERS = {
    "query_lots": _query_lots,
    "predict_fail": _predict_fail,
    "top_features": _top_features,
    "spc_check": _spc_check,
    "search_notes": _search_notes,
}
