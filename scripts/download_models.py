#!/usr/bin/env python3
"""Download pinned release-model snapshots into ``models/serve/<alias>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "models" / "serve"


def parse_models(value: str, config: dict[str, dict]) -> list[str]:
    primary = [
        name
        for name, entry in config.items()
        if isinstance(entry, dict) and entry.get("primary")
    ]
    if value.strip().lower() == "all":
        return primary
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(config))
    if unknown:
        raise ValueError(f"Unknown model aliases: {', '.join(unknown)}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="Comma-separated aliases or 'all'.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    config = json.loads((models_dir / "model_config.json").read_text(encoding="utf-8"))
    selected = parse_models(args.models, config)
    for alias in selected:
        entry = config[alias]
        target = models_dir / alias
        print(f"{alias}: {entry['repo_id']}@{entry['revision']} -> {target}")
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download

        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            local_dir=target,
        )


if __name__ == "__main__":
    main()
