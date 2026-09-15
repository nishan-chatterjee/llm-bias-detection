#!/usr/bin/env python3
"""Add notebook-ready Parquet tables to an existing downloaded dataset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.csv as pacsv
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "political_compass": {
        "pct_configuration_scores": REPO_ROOT / "tasks/political-compass/analysis/data/pct_configuration_scores.csv",
        "chat_configuration_scores": REPO_ROOT / "tasks/political-compass/analysis/data/chat_configuration_scores.parquet",
        "chat_stage_agreement_summary": REPO_ROOT / "tasks/political-compass/analysis/data/chat_stage_agreement_summary.csv",
        "pct_factor_sensitivity": REPO_ROOT / "tasks/political-compass/analysis/data/pct_factor_sensitivity.csv",
    },
    "sentiment": {
        name: REPO_ROOT / "tasks/sentiment/analysis/data" / f"{name}.csv"
        for name in (
            "coverage_by_model", "config_level_metrics", "summary_by_model",
            "summary_by_model_persona", "factor_sensitivity_macro_f1",
            "topic_descriptive_metrics",
        )
    },
    "hate_speech": {
        name: REPO_ROOT / "tasks/hate-speech/analysis/data" / f"{name}.csv"
        for name in (
            "hs_configuration_scores", "hs_factor_sensitivity",
            "hs_item_metrics_by_persona_target", "hs_calibration_by_model",
        )
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def convert(source: Path, target: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(source) if source.suffix == ".parquet" else pacsv.read_csv(source)
    pq.write_table(table, target, compression="zstd", compression_level=5)
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data/release")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not (root / "metadata/manifest.json").is_file():
        raise FileNotFoundError(f"Not a downloaded release tree: {root}")

    old_manifest = json.loads((root / "metadata/manifest.json").read_text(encoding="utf-8"))
    row_counts = {item["path"]: item.get("rows") for item in old_manifest}
    for task, tables in TABLES.items():
        for name, source in tables.items():
            target = root / "data/analysis_ready" / task / f"{name}.parquet"
            row_counts[str(target.relative_to(root))] = convert(source, target)
            print(f"{source.name} -> {target.relative_to(root)}")

    shutil.copy2(REPO_ROOT / "dataset/DATASET_CARD.md", root / "README.md")
    manifest = []
    for path in sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and path != root / "metadata/manifest.json"
    ):
        relative = str(path.relative_to(root))
        manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "rows": row_counts.get(relative),
            }
        )
    (root / "metadata/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rebuilt manifest with {len(manifest)} files")


if __name__ == "__main__":
    main()
