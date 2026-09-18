#!/usr/bin/env python3
"""Build a non-overwriting, <30 MB canonical-code supplement from a Git revision."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 30_000_000


def excluded(name: str) -> bool:
    path = PurePosixPath(name)
    return (path.parts[0] in {'legacy', 'docs', 'restricted_inputs'}
            or 'restricted_inputs' in path.parts
            or path.suffix.lower() in {'.npz', '.gguf', '.safetensors', '.pt', '.pth', '.bin'}
            or (name.startswith('tasks/political-compass/data/questions/') and path.suffix == '.json')
            or name == 'tasks/political-compass/instrument/political_compass_data.csv')


def build(revision: str, output: Path) -> tuple[int, str]:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    commit = subprocess.check_output(['git', 'rev-parse', f'{revision}^{{commit}}'], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'archive', '--format=zip', commit], cwd=ROOT)
    prefix = f'llm-bias-detection-{revision}/'
    packaged = io.BytesIO()
    count = 0
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(packaged, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in original.infolist():
            name = item.filename.rstrip('/')
            if not name or excluded(name):
                continue
            item.filename = prefix + item.filename
            archive.writestr(item, original.read(name + ('/' if item.is_dir() else '')))
            count += not item.is_dir()
        archive.comment = f'Canonical code subset of {commit}; legacy/docs/restricted inputs excluded; file bytes unchanged'.encode()
    payload = packaged.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError(f'Archive is {len(payload):,} bytes; exceeds 30 MB. Nothing written.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(payload)
    return count, hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, help='An existing immutable release tag')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # A simple tag avoids unsafe or ambiguous archive wrapper paths.
    if not args.revision.replace('-', '').replace('_', '').replace('.', '').isalnum():
        parser.error('Use a simple release-tag name')
    count, checksum = build(args.revision, args.output.resolve())
    print(f'{count} files; {args.output.stat().st_size:,} bytes; SHA-256 {checksum}')


if __name__ == '__main__':
    main()
