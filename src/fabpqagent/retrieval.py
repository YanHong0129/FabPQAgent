"""BM25 search over troubleshooting notes (the "R" in RAG).

Tests: tests/test_retrieval.py

The notes are short markdown files in data/notes/: the first line is "# <title>", the rest is
the body. BM25 rather than embeddings because the 12 notes are full of sensor IDs like "s059",
where an exact token match matters and embeddings blur identifiers (a larger system would use
hybrid search).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from fabpqagent.data import ROOT

NOTES_DIR = ROOT / "data" / "notes"
K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"[a-z0-9]+")
SNIPPET_CHARS = 600   # notes are 300-480 chars: carry the whole note, including its "Action:" line


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
    snippet: str    # first SNIPPET_CHARS characters of the body (whole note today)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_notes(notes_dir: Path = NOTES_DIR) -> list[Note]:
    """All *.md files in notes_dir, sorted by file name. Title = first line without '# '."""
    return [_read_note(path) for path in sorted(Path(notes_dir).glob("*.md"))]


def _read_note(path: Path) -> Note:
    first_line, _, body = path.read_text(encoding="utf-8").partition("\n")
    return Note(note_id=path.stem, title=first_line.removeprefix("#").strip(), body=body.strip())


class NoteIndex:
    def __init__(self, notes: list[Note]):
        """Build the inverted index once. Index the title AND the body."""
        self._notes = list(notes)
        docs = [tokenize(f"{note.title} {note.body}") for note in self._notes]
        self._lengths = [len(doc) for doc in docs]
        self._avg_length = sum(self._lengths) / len(docs) if docs else 0.0
        # Inverted index: term -> {position of the note in self._notes: times the term appears in it}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        for position, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                self._postings[term][position] = count
        n_docs = len(docs)
        self._idf = {term: math.log(1 + (n_docs - len(hits) + 0.5) / (len(hits) + 0.5))
                     for term, hits in self._postings.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Top-k notes by BM25 score, highest first; ties broken by note_id. Notes scoring 0 are
        not returned. Repeated query words count once."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):  # set(): "s059 s059" counts once
            for position, tf in self._postings.get(term, {}).items():
                length_norm = 1 - B + B * self._lengths[position] / self._avg_length
                scores[position] += self._idf[term] * tf * (K1 + 1) / (tf + K1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._notes[item[0]].note_id))
        return [self._hit(position, score) for position, score in ranked[:k] if score > 0]

    def _hit(self, position: int, score: float) -> Hit:
        note = self._notes[position]
        return Hit(note.note_id, note.title, score, snippet=note.body[:SNIPPET_CHARS])
