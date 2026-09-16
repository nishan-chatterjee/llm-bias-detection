#!/usr/bin/env python3
"""Add verified hate inputs and scoped licences to an existing release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def package_inputs(stage: Path, release: Path = ROOT) -> dict[str, int]:
    sys.path.insert(0, str(release / 'tasks/hate-speech'))
    from run_hate_speech import load_corpus, load_prompts
    data = release / 'tasks/hate-speech/data'
    load_prompts(data / 'prompts/english.json')
    records, grouped = load_corpus(data / 'corpus/english.jsonl')
    assert len(records) == 13_320
    assert all(len(items) == 1332 and sum(i['hate'] for i in items) == 666
               for items in grouped.values())
    for relative in ['prompts/english.json', 'corpus/english.jsonl', 'README.md']:
        target = stage / 'inputs/hate_speech' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / relative, target)
    for name in ['LICENSE-CC-BY-4.0.md', 'THIRD_PARTY_NOTICES.md']:
        shutil.copy2(release / name, stage / name)
    table = pajson.read_json(data / 'corpus/english.jsonl')
    positions = {}
    targets, indices = [], []
    for item in records:
        target = item['target']
        positions[target] = positions.get(target, 0) + 1
        targets.append(target)
        indices.append(positions[target])
    table = table.append_column('question_id', pa.array(range(1, len(records) + 1)))
    table = table.append_column('target', pa.array(targets))
    table = table.append_column('item_index', pa.array(indices, type=pa.int16()))
    target = stage / 'data/hate_speech_inputs/english.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression='zstd', compression_level=5)
    return {'data/hate_speech_inputs/english.parquet': len(records),
            'inputs/hate_speech/corpus/english.jsonl': len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    args = parser.parse_args()
    stage = args.dataset_root.resolve()
    manifest_path = stage / 'metadata/manifest.json'
    old = json.loads(manifest_path.read_text())
    counts = {item['path']: item.get('rows') for item in old}
    counts.update(package_inputs(stage))
    shutil.copy2(ROOT / 'dataset/DATASET_CARD.md', stage / 'README.md')
    provenance_path = stage / 'metadata/provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        provenance['hate_speech_input_update'] = {
            'date': '2026-09-16', 'items': 13_320,
            'alignment': 'gold labels and source metadata at all 19,180,800 within-target output positions',
            'historical_text_hashes': 'not recorded; byte identity cannot be established',
            'original_predictions': 'unchanged',
        }
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    manifest = []
    for path in sorted(stage.rglob('*')):
        if not path.is_file() or '.cache' in path.parts or path == manifest_path:
            continue
        relative = str(path.relative_to(stage))
        manifest.append({'path': relative, 'bytes': path.stat().st_size,
                         'sha256': digest(path), 'rows': counts.get(relative)})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged verified hate inputs; {len(manifest)} manifested files')


if __name__ == '__main__':
    main()
