"""DAY 2 — YOU WRITE THIS. BM25 search over troubleshooting notes (the "R" in RAG).

Tests: tests/test_retrieval.py

You built BM25 in CS329z Problem 3; write it again here from memory (don't copy course code
into a public repo). The notes are short markdown files in data/notes/: the first line is
"# <title>", the rest is the body.

Design question: why BM25 and not embeddings for 15 short notes full of sensor IDs like "s059"?
(Exact-token match matters; embeddings blur identifiers. A real system would do hybrid search.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fabpqagent.data import ROOT

NOTES_DIR = ROOT / "data" / "notes"
K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Note:
    note_id: str    # file stem, e.g. "01-s059-chamber-pressure"
    title: str
    body: str


@dataclass(frozen=True)
class Hit:
    note_id: str
    title: str
    score: float
    snippet: str    # first 300 characters of the body


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_notes(notes_dir: Path = NOTES_DIR) -> list[Note]:
    """All *.md files in notes_dir, sorted by file name. Title = first line without '# '."""
    raise NotImplementedError


class NoteIndex:
    def __init__(self, notes: list[Note]):
        """Build the inverted index once. Index the title AND the body."""
        raise NotImplementedError

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Top-k notes by BM25 score, highest first; ties broken by note_id. Notes scoring 0 are
        not returned. Repeated query words count once."""
        raise NotImplementedError
