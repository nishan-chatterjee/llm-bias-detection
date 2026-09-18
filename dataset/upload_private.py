#!/usr/bin/env python3
"""Create/update the personal Hugging Face Dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--repo-id", default="nishan-chatterjee/llm-bias-detection")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or retain a private Dataset repository (public by default).",
    )
    args = parser.parse_args()
    if not args.private and (args.stage / 'restricted_inputs').exists():
        raise ValueError('Refusing to publish restricted questionnaire inputs; keep these locally.')
    owner, _name = args.repo_id.split("/", 1)
    api = HfApi()
    who = api.whoami()
    if owner != who["name"]:
        raise ValueError(
            f"Refusing non-personal owner {owner!r}; active user is {who['name']!r}"
        )
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.stage.resolve()),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add primary evaluation traces and validated release metadata",
    )
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if bool(getattr(info, "private", False)) != args.private:
        expected = "private" if args.private else "public"
        raise RuntimeError(f"Dataset repository is not {expected} after upload")
    visibility = "private" if args.private else "public"
    print(f"Uploaded {visibility} dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
