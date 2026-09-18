#!/usr/bin/env python3
"""Download the companion dataset and optionally install licensed hate inputs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "nishan-chatterjee/llm-bias-detection"
DEFAULT_TARGET = ROOT / "data" / "release"
DEFAULT_REVISION = "33a2f02a19eeea7ddd42706d8a5d6f3e26ac2ea2"
COMPONENT_PATTERNS = {
    "inputs": ["README.md", "inputs/**", "metadata/**",
               "LICENSE-CC-BY-4.0.md", "THIRD_PARTY_NOTICES.md"],
    "analysis": ["README.md", "data/analysis_ready/**", "metadata/**",
                 "LICENSE-CC-BY-4.0.md", "THIRD_PARTY_NOTICES.md"],
    "hate-inputs": ["README.md", "inputs/hate_speech/**", "data/hate_speech_inputs/**",
                    "LICENSE-CC-BY-4.0.md", "THIRD_PARTY_NOTICES.md", "metadata/**"],
    "political-compass": [
        "README.md",
        "data/political_compass_*/**",
        "data/analysis_ready/political_compass/**",
        "inputs/political_compass/**",
        "metadata/**",
    ],
    "sentiment": [
        "README.md", "data/ibm_sentiment/**", "data/analysis_ready/sentiment/**",
        "inputs/ibm_sentiment/**", "metadata/**",
    ],
    "hate-speech": [
        "README.md", "data/hate_speech*/**", "data/analysis_ready/hate_speech/**",
        "inputs/hate_speech/**", "LICENSE-CC-BY-4.0.md", "THIRD_PARTY_NOTICES.md",
        "metadata/**",
    ],
}


def install_hate_inputs(snapshot: Path, root: Path = ROOT) -> int:
    import json
    source = snapshot / 'inputs/hate_speech'
    manifest = {item['path']: item for item in
                json.loads((snapshot / 'metadata/manifest.json').read_text())}
    pending = []
    for relative in ['prompts/english.json', 'corpus/english.jsonl']:
        path = source / relative
        wanted = manifest[f'inputs/hate_speech/{relative}']['sha256']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError(f'Downloaded hate input checksum mismatch: {relative}')
        target = root / 'tasks/hate-speech/data' / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != actual:
            raise FileExistsError(f'Refusing to overwrite different local hate input: {target}')
        pending.append((path, target))
    for path, target in pending:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return len(pending)


def install_pct_questions(snapshot: Path) -> int:
    raise ValueError('Political Compass questionnaires are not distributed; obtain authorized inputs locally.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset revision (defaults to the release commit pinned by this code).",
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--component",
        choices=["all", *COMPONENT_PATTERNS],
        default="all",
        help="Download all files or only one task/input subset.",
    )
    parser.add_argument(
        "--install-pct-questions",
        action="store_true",
        help=(
            "Deprecated: Political Compass questionnaires are no longer distributed. "
            "Provide authorized local inputs yourself."
        ),
    )
    parser.add_argument(
        "--acknowledge-pct-terms",
        action="store_true",
        help="Required with --install-pct-questions; confirms you reviewed the upstream terms.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument('--install-hate-inputs', action='store_true',
                        help='Install verified corpus/prompt files without overwriting different local inputs.')
    args = parser.parse_args()
    if args.install_pct_questions:
        parser.error(
            "Political Compass questionnaires are not distributed. Provide authorized "
            "local files; see THIRD_PARTY_NOTICES.md."
        )
    if args.install_hate_inputs and args.component not in ['all', 'inputs', 'hate-speech', 'hate-inputs']:
        parser.error('--install-hate-inputs requires a component that downloads the hate inputs')
    target = args.target.resolve()
    print(f"dataset {args.repo_id}@{args.revision} ({args.component}) -> {target}")
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=target,
            allow_patterns=COMPONENT_PATTERNS.get(args.component),
            ignore_patterns=['restricted_inputs/**'],
        )
    )
    if args.install_pct_questions:
        count = install_pct_questions(snapshot)
        print(f"installed {count} Political Compass question files")
    if args.install_hate_inputs:
        print(f'installed/verified {install_hate_inputs(snapshot)} hate input files')
    print(f"downloaded dataset snapshot: {snapshot}")


if __name__ == "__main__":
    main()
