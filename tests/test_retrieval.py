"""Day 2: retrieval.py"""
from fabpqagent.retrieval import NOTES_DIR, NoteIndex, load_notes


def test_load_notes_sorted_with_titles(notes_dir):
    notes = load_notes(notes_dir)
    assert [n.note_id for n in notes] == ["01-s059-pressure", "02-etch-rate", "03-handbook"]
    assert notes[0].title == "s059 chamber pressure drift"
    assert not notes[0].body.startswith("#")


def test_exact_sensor_id_ranks_first(notes_dir):
    hits = NoteIndex(load_notes(notes_dir)).search("why is s519 high", k=3)
    assert hits[0].note_id == "02-etch-rate"
    assert hits[0].score > 0 and hits[0].snippet


def test_zero_score_notes_excluded_and_k_respected(notes_dir):
    index = NoteIndex(load_notes(notes_dir))
    assert index.search("s059", k=3)[0].note_id == "01-s059-pressure"
    assert len(index.search("s059", k=3)) == 1
    assert index.search("banana", k=3) == []
    assert len(index.search("chamber", k=1)) == 1


def test_repeated_query_words_count_once(notes_dir):
    index = NoteIndex(load_notes(notes_dir))
    assert index.search("s059 s059 s059")[0].score == index.search("s059")[0].score


def test_shipped_notes_load():
    notes = load_notes(NOTES_DIR)
    assert len(notes) >= 10
    assert NoteIndex(notes).search("s059")[0].note_id.startswith("01")
