#!/usr/bin/env python3
"""Check each advertised HF configuration without downloading all raw traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='nishan-chatterjee/llm-bias-detection')
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    frontmatter = (ROOT / 'dataset/DATASET_CARD.md').read_text().split('---', 2)[1]
    configs = yaml.safe_load(frontmatter)['configs']
    for config in configs:
        name = config['config_name']
        if name.startswith('analysis_'):
            dataset = load_dataset(args.repo_id, name, revision=args.revision, split='train')
            assert len(dataset) > 0
            rows = len(dataset)
            sample = dataset[0]
        else:
            dataset = load_dataset(args.repo_id, name, revision=args.revision,
                                   split='train', streaming=True)
            sample = next(iter(dataset))
            rows = 'streaming sample'
        assert sample
        print(json.dumps({'config': name, 'rows': rows,
                          'column_count': len(sample)}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
