#!/usr/bin/env python3
"""Create synthetic benign inputs for software smoke tests, NOT the paper corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        'hate_runner_fixture', ROOT / 'tasks/hate-speech/run_hate_speech.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = {
        kind: [{'id': f'{kind}_{idx}', 'text': '{name}'} for idx in range(5)]
        for kind in ('short', 'long')
    }
    insertions = {
        ideology: {template['id']: {'name': ideology}
                   for values in templates.values() for template in values}
        for ideology in module.IDEOLOGIES if ideology != 'base'
    }
    prompts = {
        'contexts': [{'text': 'Synthetic software test.'} for _ in range(5)],
        'instructions': [{'text': 'Classify whether this contains hate speech. '
                          'Answer True or False: {text}'} for _ in range(10)],
        'persona_templates': templates, 'ideology_insertions': insertions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output / 'prompts.synthetic.json'
    corpus_path = args.output / 'corpus.synthetic.jsonl'
    if prompt_path.exists() or corpus_path.exists():
        raise FileExistsError('Use a fresh fixture directory; existing files are not overwritten.')
    prompt_path.write_text(json.dumps(prompts, indent=2) + '\n')
    rows = [{'text': 'Everyone deserves respect.', 'hate': False,
             'target_groups': [target], 'dataset': 'synthetic-software-smoke',
             'grouping': 'NOT-paper-data'}
            for target in module.TARGETS for _ in range(2)]
    corpus_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(f'SYNTHETIC SOFTWARE FIXTURE ONLY: {args.output}')


if __name__ == '__main__':
    main()
