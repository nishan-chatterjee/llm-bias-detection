#!/usr/bin/env python3
"""Resolve a released model alias to a local checkpoint or pinned Hub snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "models" / "serve"


def resolve(alias: str, models_dir: Path) -> dict[str, object]:
    config_path = models_dir / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if alias not in config:
        raise ValueError(f"Unknown model alias {alias!r}; choose from {', '.join(config)}")
    entry = config[alias]
    local = models_dir / alias
    if (local / "config.json").is_file():
        source = str(local.resolve())
        revision = ""
        location = "local"
    else:
        source = str(entry["repo_id"])
        revision = str(entry.get("revision") or "")
        location = "hub"
    return {
        "alias": alias,
        "source": source,
        "revision": revision,
        "location": location,
        "family": entry.get("family"),
        "primary": bool(entry.get("primary", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--field",
        choices=["alias", "source", "revision", "location", "family", "primary"],
        help="Print one machine-readable field instead of JSON.",
    )
    args = parser.parse_args()
    result = resolve(args.alias, args.models_dir.resolve())
    if args.field:
        value = result[args.field]
        print(str(value).lower() if isinstance(value, bool) else value)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
