#!/usr/bin/env python3
"""Validate the bounded outputs from the maintainer's GPU smoke suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    files = [
        'political-compass/smoke/chat/gemma-3-1b-it.jsonl',
        'political-compass/smoke/chat-think/Qwen3-4B_think.jsonl',
        'qwen-no-think-exact/Qwen3-4B_no_think.jsonl',
        'sentiment/smoke/gemma-3-1b-it.jsonl',
        'hate-synthetic/smoke/gemma-3-1b-it.jsonl',
    ]
    for relative in files:
        path = args.root / relative
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(records) == 2, (path, len(records))
        for record in records:
            assert not record.get('error'), (path, record.get('error'))
            probs = record['classification_probs']
            values = list(probs.values()) if isinstance(probs, dict) else probs
            assert np.isfinite(values).all() and np.isclose(sum(values), 1, atol=1e-5)
            assert min(values) >= 0 and max(values) <= 1
            if 'stage1_text' in record:
                assert record['stage1_text'].strip()
        print(json.dumps({'file': relative, 'rows': len(records), 'errors': 0}), flush=True)
    path = args.root / 'political-compass/smoke/mcq/results_gemma-3-1b-it_bf16.csv'
    frame = pd.read_csv(path)
    assert len(frame) == 1
    for question in range(2):
        values = frame[[f'q{question}_prob_ans{idx}' for idx in range(4)]].to_numpy()
        assert np.isfinite(values).all() and np.isclose(values.sum(), 1, atol=1e-5)
    print(json.dumps({'file': str(path.name), 'configs': 1, 'questions': 2, 'errors': 0}))


if __name__ == '__main__':
    main()
