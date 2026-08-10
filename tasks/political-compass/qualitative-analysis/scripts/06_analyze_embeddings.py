#!/usr/bin/env python3
"""Evaluate trace separability, content-vs-length, and unsupervised clusters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import EMBEDDING_DIR, TABLE_DIR  # noqa: E402
from io_utils import write_json  # noqa: E402


def load_shards(vector_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    indexes, vectors = [], []
    for path in sorted(vector_dir.glob("manifest_*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        indexes.append(pd.read_parquet(metadata["index"]))
        vectors.append(np.load(metadata["vectors"]))
    if not indexes:
        raise FileNotFoundError(f"No embedding manifests in {vector_dir}")
    return pd.concat(indexes, ignore_index=True), np.concatenate(vectors)


def grouped_cv(x: np.ndarray, labels: pd.Series, groups: pd.Series) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    valid = labels.notna() & groups.notna()
    x = x[valid.to_numpy()]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels[valid].astype(str))
    group_values = groups[valid].astype(str).to_numpy()
    splits = min(10, len(np.unique(group_values)))
    if len(np.unique(y)) < 2 or splits < 2:
        return {"status": "insufficient_classes_or_groups"}
    predictions = np.full(len(y), -1)
    for train, test in GroupKFold(n_splits=splits).split(x, y, group_values):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train])
        test_x = scaler.transform(x[test])
        model = LogisticRegression(
            max_iter=2_000, class_weight="balanced", solver="lbfgs"
        )
        model.fit(train_x, y[train])
        predictions[test] = model.predict(test_x)
    return {
        "status": "ok",
        "rows": len(y),
        "classes": encoder.classes_.tolist(),
        "folds": splits,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-dir", type=Path, default=EMBEDDING_DIR / "vectors")
    parser.add_argument("--output-dir", type=Path, default=TABLE_DIR / "embeddings")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--bertopic", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

    index, vectors = load_shards(args.vector_dir)
    if len(index) != len(vectors):
        raise ValueError("Embedding index/vector row mismatch")
    print(
        f"Loaded {len(index):,} embeddings with {vectors.shape[1]:,} dimensions.",
        flush=True,
    )
    components = min(args.pca_components, vectors.shape[1], len(vectors) - 1)
    print(f"Fitting PCA ({components} components)...", flush=True)
    reduced = PCA(n_components=components, random_state=42).fit_transform(
        vectors.astype(np.float32)
    )
    print("PCA complete; running grouped classification diagnostics...", flush=True)
    length = np.log1p(
        index[["stage1_tokens", "think_words", "visible_words"]]
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    rationale = index["representation"].eq("rationale")
    tasks = {
        "ideology_embedding_question_holdout": (
            reduced[rationale],
            index.loc[rationale, "ideology"],
            index.loc[rationale, "question_id"],
        ),
        "family_embedding_question_holdout": (
            reduced[rationale],
            index.loc[rationale, "family"],
            index.loc[rationale, "question_id"],
        ),
        "reasoning_instruction_embedding_question_holdout": (
            reduced[rationale],
            index.loc[rationale, "reasoning_mode"],
            index.loc[rationale, "question_id"],
        ),
    }
    for representation in index["representation"].unique():
        qwen = index["think_outcome"].isin(["think_rescue", "think_harm"]) & index[
            "representation"
        ].eq(representation)
        if not qwen.any():
            continue
        prefix = f"rescue_vs_harm__{representation}"
        tasks[f"{prefix}__length_only"] = (
            length[qwen],
            index.loc[qwen, "think_outcome"],
            index.loc[qwen, "question_id"],
        )
        tasks[f"{prefix}__content_only"] = (
            reduced[qwen],
            index.loc[qwen, "think_outcome"],
            index.loc[qwen, "question_id"],
        )
        tasks[f"{prefix}__content_plus_length"] = (
            np.concatenate([reduced[qwen], length[qwen]], axis=1),
            index.loc[qwen, "think_outcome"],
            index.loc[qwen, "question_id"],
        )
    metrics = {name: grouped_cv(*values) for name, values in tasks.items()}
    print("Grouped classification diagnostics complete.", flush=True)

    projection = index.copy()
    projection["pca_0"] = reduced[:, 0]
    projection["pca_1"] = reduced[:, 1] if reduced.shape[1] > 1 else 0.0
    cluster_labels = np.full(len(index), -1, dtype=int)
    if not args.skip_umap:
        import hdbscan
        import umap

        print(
            "Fitting reproducible UMAP on all embeddings. This is CPU-bound, "
            "single-threaded because random_state=42, and may be silent for a while...",
            flush=True,
        )
        manifold = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=0.05,
            n_components=2,
            metric="cosine",
            random_state=42,
        ).fit_transform(reduced)
        print("UMAP complete; fitting HDBSCAN...", flush=True)
        cluster_labels = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size
        ).fit_predict(manifold)
        print("HDBSCAN complete.", flush=True)
        projection["umap_0"] = manifold[:, 0]
        projection["umap_1"] = manifold[:, 1]
    projection["cluster"] = cluster_labels
    projection.to_parquet(args.output_dir / "embedding_projection.parquet", index=False)

    if args.bertopic:
        from bertopic import BERTopic

        print(
            "Starting optional BERTopic fit (a separate topic-model fit after "
            "UMAP/HDBSCAN)...",
            flush=True,
        )
        topic_model = BERTopic(
            min_topic_size=args.min_cluster_size,
            calculate_probabilities=False,
            verbose=True,
        )
        topics, _ = topic_model.fit_transform(
            index["rationale_text"].fillna("").tolist(),
            embeddings=vectors.astype(np.float32),
        )
        projection["bertopic_topic"] = topics
        projection[
            ["trace_id", "bertopic_topic"]
        ].to_parquet(args.output_dir / "bertopic_assignments.parquet", index=False)
        topic_model.get_topic_info().to_csv(
            args.output_dir / "bertopic_topic_info.csv", index=False
        )
        print("BERTopic complete.", flush=True)

    associations = []
    non_noise = cluster_labels >= 0
    for column in [
        "ideology",
        "model_variant",
        "family",
        "protocol",
        "question_id",
        "topic",
        "reasoning_mode",
    ]:
        if non_noise.sum() and projection.loc[non_noise, column].nunique() > 1:
            values = projection.loc[non_noise, column].astype(str)
            labels = cluster_labels[non_noise]
            associations.append(
                {
                    "attribute": column,
                    "adjusted_mutual_information": float(
                        adjusted_mutual_info_score(values, labels)
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(values, labels)
                    ),
                }
            )
    pd.DataFrame(associations).to_csv(
        args.output_dir / "cluster_associations.csv", index=False
    )
    write_json(
        args.output_dir / "metrics.json",
        {
            "rows": len(index),
            "dimensions": int(vectors.shape[1]),
            "classification": metrics,
            "clusters": int(len(set(cluster_labels)) - (-1 in cluster_labels)),
            "noise_fraction": float((cluster_labels < 0).mean()),
        },
    )
    for name, value in metrics.items():
        print(name, value, flush=True)


if __name__ == "__main__":
    main()
