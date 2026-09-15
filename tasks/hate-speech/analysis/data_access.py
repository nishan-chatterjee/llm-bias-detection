"""Resolve hate-speech analysis tables from the pinned release or Git fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def table_path(name: str) -> Path:
    release_root = Path(
        os.environ.get("LLM_BIAS_DATA_DIR", REPO_ROOT / "data" / "release")
    ).expanduser().resolve()
    candidates = [
        release_root / "data" / "analysis_ready" / "hate_speech" / f"{name}.parquet",
        HERE / "data" / f"{name}.parquet",
        HERE / "data" / f"{name}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No hate-speech analysis table {name!r}. Run "
        "`python scripts/download_dataset.py --component analysis`."
    )


def read_table(name: str) -> pd.DataFrame:
    path = table_path(name)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
