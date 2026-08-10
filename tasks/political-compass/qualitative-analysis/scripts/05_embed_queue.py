#!/usr/bin/env python3
"""Embed one deterministic queue shard with SentenceTransformers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import EMBEDDING_DIR  # noqa: E402
from io_utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--queue-dir", type=Path, default=EMBEDDING_DIR / "queue")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDING_DIR / "vectors")
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="SentenceTransformers model ID or local checkpoint path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    queue_path = args.queue_dir / f"queue_{args.shard_id:02d}.parquet"
    vector_path = args.output_dir / f"vectors_{args.shard_id:02d}.npy"
    index_path = args.output_dir / f"index_{args.shard_id:02d}.parquet"
    manifest_path = args.output_dir / f"manifest_{args.shard_id:02d}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not queue_path.exists():
        raise FileNotFoundError(
            f"{queue_path} does not exist. Finish run_cpu_pipeline.sh before "
            "launching the embedding workers."
        )
    if (vector_path.exists() or index_path.exists()) and not args.overwrite:
        if vector_path.exists() and index_path.exists() and manifest_path.exists():
            print(f"Embedding shard {args.shard_id} already exists; reusing it.")
            return
        raise FileExistsError("Partial embedding outputs exist; pass --overwrite")

    frame = pd.read_parquet(queue_path)
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    vectors = model.encode(
        frame["embedding_text"].fillna("").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(args.dtype)
    np.save(vector_path, vectors)
    frame.drop(columns=["embedding_text"]).to_parquet(index_path, index=False)
    write_json(
        manifest_path,
        {
            "shard_id": args.shard_id,
            "rows": len(frame),
            "dimensions": int(vectors.shape[1]),
            "model": args.model,
            "max_seq_length": args.max_seq_length,
            "dtype": args.dtype,
            "normalized": True,
            "vectors": str(vector_path),
            "index": str(index_path),
        },
    )
    print(f"Wrote {vectors.shape} to {vector_path}")


if __name__ == "__main__":
    main()
