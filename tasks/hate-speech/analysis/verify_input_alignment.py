#!/usr/bin/env python3
"""Check candidate corpus order/labels/provenance against archived item outputs.

Archived outputs contain no source text or prompt hashes. Passing this check
confirms positional metadata alignment, not a historical text-byte identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_hate_speech import load_corpus, load_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()
    load_prompts(args.prompts)
    records, grouped = load_corpus(args.corpus)
    assert len(records) == 13_320
    expected = {}
    for target, items in grouped.items():
        gold = np.array([item['hate'] for item in items])
        assert len(items) == 1332 and gold.sum() == 666, target
        meta = np.array([f"{item['dataset']};{item['grouping']}" for item in items])
        expected[target] = (gold, meta)
    paths = sorted(args.outputs.glob('*.parquet'))
    assert len(paths) == 8, len(paths)
    total = 0
    for path in paths:
        checked = 0
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=65536, columns=['target', 'item_index', 'gold_hate', 'source_meta']):
            targets = batch.column('target').to_numpy(zero_copy_only=False)
            positions = batch.column('item_index').to_numpy(zero_copy_only=False).astype(int) - 1
            gold = batch.column('gold_hate').to_numpy(zero_copy_only=False)
            meta = batch.column('source_meta').to_numpy(zero_copy_only=False)
            for target in np.unique(targets):
                mask = targets == target
                indices = positions[mask]
                assert indices.min() >= 0 and indices.max() < 1332, (path, target)
                wanted_gold, wanted_meta = expected[target]
                assert np.array_equal(gold[mask], wanted_gold[indices]), (path, target, 'gold')
                assert np.array_equal(meta[mask], wanted_meta[indices]), (path, target, 'source')
            checked += batch.num_rows
        assert checked == 2_397_600, (path, checked)
        print(json.dumps({'model_file': path.name, 'aligned_rows': checked}), flush=True)
        total += checked
    print(json.dumps({'total_aligned_rows': total,
                      'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                      'prompt_sha256': hashlib.sha256(args.prompts.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
