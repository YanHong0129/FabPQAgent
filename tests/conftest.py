from pathlib import Path

import pandas as pd
import pytest

from fabpqagent.data import DB_PATH


@pytest.fixture(scope="session")
def lots() -> pd.DataFrame:
    if not DB_PATH.exists():
        pytest.skip("data/secom.db missing. Run: uv run python -m fabpqagent.data")
    from fabpqagent.db import load_lots
    return load_lots()


@pytest.fixture(scope="session")
def trained(lots):
    from fabpqagent.model import train
    return train(lots, seed=0)


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    notes = {
        "01-s059-pressure.md": "# s059 chamber pressure drift\nWhen s059 drifts high, lots fail final test. Check chamber pressure and pump.",
        "02-etch-rate.md": "# Etch rate excursion\nLow etch rate shows up as high s519. Verify etch chamber seasoning.",
        "03-handbook.md": "# SPC basics\nUse control limits from a stable baseline period. Rule of eight detects a shift.",
    }
    for name, text in notes.items():
        (tmp_path / name).write_text(text)
    return tmp_path
