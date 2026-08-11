#!/usr/bin/env python3
"""Run the primary-model hate-speech MCQ experiment.

This is a release-oriented refactor of the historical runner.  It preserves
the original 8-dimensional LHS design and exact last-token scoring of the
literal candidates ``True`` and ``False``, while replacing hard-coded paths,
GPU IDs, and wide CSV checkpoints with an explicit CLI and resumable JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
MODELS_DIR = REPO_ROOT / "models" / "serve"
DEFAULT_PROMPTS = HERE / "data" / "prompts" / "english.json"
DEFAULT_CORPUS = HERE / "data" / "corpus" / "english.jsonl"
DEFAULT_OUTPUT = HERE / "output" / "hate-speech"

MODEL_ALIASES = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]
# Preserve the exact model-block order used by the completed historical run so
# global config_id values match the supplied 14,400-row design.
HISTORICAL_MODEL_ORDER = [
    "gemma-3-12b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-27b-it",
]
IDEOLOGIES = [
    "base",
    "libertarian_right",
    "authoritarian_left",
    "authoritarian_right",
    "libertarian_left",
    "centrism",
]
TARGETS = [
    "women",
    "black people",
    "muslims",
    "asian people",
    "men",
    "latinx people",
    "white people",
    "jews",
    "lgbtq+ people",
    "christians",
]
REQUIRED_ITEM_FIELDS = {"text", "hate", "target_groups", "dataset", "grouping"}


def load_model_config(path: Path = MODELS_DIR / "model_config.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model(alias: str, model_config: dict, models_dir: Path = MODELS_DIR) -> tuple[str, str | None]:
    if alias not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model {alias!r}; choose from {MODEL_ALIASES}")
    local = models_dir / alias
    if (local / "config.json").exists():
        return str(local), None
    entry = model_config.get(alias, {})
    if not entry.get("repo_id"):
        raise FileNotFoundError(f"No local checkpoint or configured Hugging Face ID for {alias}")
    return str(entry["repo_id"]), entry.get("revision")


def map_factors(sample: np.ndarray) -> dict:
    """Map one historical 8-D LHS point to the five varied factors.

    Dimensions 2, 6, and 7 were unused in the archived experiment. They remain
    in the design so a fixed seed reproduces the original schedule.
    """
    return {
        "language": "english",
        "target": TARGETS[min(int(sample[0] * len(TARGETS)), len(TARGETS) - 1)],
        "context_id": None
        if sample[1] < 0.5
        else min(int((sample[1] - 0.5) * 2 * 5), 4),
        "instr_idx": min(int(sample[3] * 10), 9),
        "persona_class": "short" if sample[4] < 0.5 else "long",
        "persona_idx": min(int(sample[5] * 5), 4),
    }


def generate_design(
    samples: int = 300,
    seed: int = 42,
    models: Iterable[str] = HISTORICAL_MODEL_ORDER,
) -> pd.DataFrame:
    # The completed experiment used SciPy's legacy ``seed=`` constructor.
    # ``rng=seed`` is not stream-equivalent and generates a different design.
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    points = sampler.random(n=samples)
    rows = []
    for alias in models:
        if alias not in MODEL_ALIASES:
            raise ValueError(f"Unsupported model {alias!r}")
        for ideology in IDEOLOGIES:
            for point in points:
                row = map_factors(point)
                row.update({"model_alias": alias, "ideology": ideology, "quantization": "bf16"})
                if ideology == "base":
                    row["persona_class"] = None
                    row["persona_idx"] = None
                rows.append(row)
    design = pd.DataFrame(rows)
    design.insert(0, "config_id", np.arange(len(design), dtype=int))
    return design


def validate_prompts(prompts: dict) -> None:
    if len(prompts.get("contexts", [])) != 5:
        raise ValueError("Expected exactly five hate-speech context templates")
    if len(prompts.get("instructions", [])) != 10:
        raise ValueError("Expected exactly ten hate-speech instruction templates")
    for idx, entry in enumerate(prompts["instructions"]):
        if "{text}" not in entry.get("text", ""):
            raise ValueError(f"Instruction {idx} must contain {{text}}")
    for persona_class in ("short", "long"):
        if len(prompts.get("persona_templates", {}).get(persona_class, [])) != 5:
            raise ValueError(f"Expected five {persona_class} persona templates")
    for ideology in IDEOLOGIES[1:]:
        if ideology not in prompts.get("ideology_insertions", {}):
            raise ValueError(f"Missing persona insertions for {ideology}")


def load_prompts(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exact hate-speech prompt file: {path}. "
            "The Political Compass prompt JSON is not a valid substitute; see README.md."
        )
    prompts = json.loads(path.read_text(encoding="utf-8"))
    validate_prompts(prompts)
    return prompts


def load_corpus(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing hate-speech corpus: {path}; see README.md")
    rows: list[dict] = []
    grouped = {target: [] for target in TARGETS}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = json.loads(line)
            missing = REQUIRED_ITEM_FIELDS.difference(raw)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields {sorted(missing)}")
            groups = raw["target_groups"]
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"{path}:{line_no} has no primary target group")
            target = str(groups[0])
            if target not in grouped:
                raise ValueError(f"{path}:{line_no} has unexpected primary target {target!r}")
            item = {
                "question_id": line_no,
                "text": str(raw["text"]),
                "hate": bool(raw["hate"]),
                "target": target,
                "dataset": str(raw["dataset"]),
                "grouping": str(raw["grouping"]),
            }
            rows.append(item)
            grouped[target].append(item)
    if not rows:
        raise ValueError(f"No records found in {path}")
    empty = [target for target, items in grouped.items() if not items]
    if empty:
        raise ValueError(f"No corpus items for target groups: {empty}")
    return rows, grouped


def assemble_prompt(config: pd.Series | dict, item: dict, prompts: dict) -> str:
    segments: list[str] = []
    context_id = config.get("context_id")
    if pd.notna(context_id):
        segments.append(prompts["contexts"][int(context_id)]["text"])

    ideology = str(config["ideology"])
    if ideology != "base":
        persona_class = str(config["persona_class"])
        persona_idx = int(config["persona_idx"])
        template = prompts["persona_templates"][persona_class][persona_idx]
        insertion = prompts["ideology_insertions"][ideology][template["id"]]
        segments.append(template["text"].format(**insertion))

    instruction = prompts["instructions"][int(config["instr_idx"])]["text"]
    segments.append(instruction.format(text=item["text"]))
    return "\n\n".join(segments)


def completed_pairs(path: Path) -> set[tuple[int, int]]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("error"):
                done.add((int(record["config_id"]), int(record["question_id"])))
    return done


def append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(
    design_path: Path,
    prompt_path: Path,
    corpus_path: Path,
    output_dir: Path,
    model_alias: str,
    gpu_ids: list[int],
    batch_size: int,
    max_configs: int | None,
    max_items: int | None,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = load_model_config()
    source, revision = resolve_model(model_alias, model_config)
    repo_id = model_config[model_alias]["repo_id"]
    prompts = load_prompts(prompt_path)
    _all_items, by_target = load_corpus(corpus_path)
    design = pd.read_csv(design_path)
    design = design[design["model_alias"].eq(model_alias)].sort_values("config_id")
    if max_configs is not None:
        design = design.head(max_configs)
    if design.empty:
        raise ValueError(f"No design rows for {model_alias}")

    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="right", **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        **kwargs,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    model.eval()

    candidate_keys = ["True", "False"]
    candidate_ids = [tokenizer.encode(key, add_special_tokens=False)[-1] for key in candidate_keys]
    result_path = output_dir / f"{model_alias}.jsonl"
    done = completed_pairs(result_path)

    for _, config in design.iterrows():
        config_id = int(config["config_id"])
        target_items = by_target[str(config["target"])]
        if max_items is not None:
            target_items = target_items[:max_items]
        pending = [item for item in target_items if (config_id, item["question_id"]) not in done]
        for start in range(0, len(pending), batch_size):
            items = pending[start : start + batch_size]
            prompts_batch = [assemble_prompt(config, item, prompts) for item in items]
            try:
                inputs = tokenizer(
                    prompts_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=3000,
                ).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                last = inputs.attention_mask.sum(1) - 1
                logits = outputs.logits[torch.arange(len(items), device=model.device), last, :]
                selected = logits[:, candidate_ids]
                probs = torch.softmax(selected, dim=1, dtype=torch.float64)
                selected_cpu = selected.float().cpu().numpy()
                probs_cpu = probs.cpu().numpy()
                records = []
                for item, prompt, candidate_logits, candidate_probs in zip(
                    items, prompts_batch, selected_cpu, probs_cpu
                ):
                    records.append(
                        {
                            "config_id": config_id,
                            "model": repo_id,
                            "model_alias": model_alias,
                            "model_revision": revision,
                            "quantization": "bf16",
                            "language": "english",
                            "ideology": str(config["ideology"]),
                            "target": str(config["target"]),
                            "context_id": -1 if pd.isna(config["context_id"]) else int(config["context_id"]),
                            "instr_idx": int(config["instr_idx"]),
                            "persona_class": "" if pd.isna(config["persona_class"]) else str(config["persona_class"]),
                            "persona_idx": -1 if pd.isna(config["persona_idx"]) else int(config["persona_idx"]),
                            "question_id": int(item["question_id"]),
                            "gold_hate": bool(item["hate"]),
                            "source_dataset": item["dataset"],
                            "source_grouping": item["grouping"],
                            "candidate_keys": candidate_keys,
                            "candidate_token_ids": candidate_ids,
                            "candidate_logits": dict(zip(candidate_keys, map(float, candidate_logits))),
                            "classification_probs": dict(zip(candidate_keys, map(float, candidate_probs))),
                            "predicted_hate": bool(candidate_probs[0] >= candidate_probs[1]),
                            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
                            "error": None,
                        }
                    )
                append_records(result_path, records)
                done.update((config_id, item["question_id"]) for item in items)
            except Exception as exc:
                append_records(
                    result_path,
                    [
                        {
                            "config_id": config_id,
                            "model": repo_id,
                            "model_alias": model_alias,
                            "question_id": int(item["question_id"]),
                            "error": f"batch_failed: {type(exc).__name__}: {exc}",
                        }
                        for item in items
                    ],
                )
                raise
    print(f"Results: {result_path}")


def parse_models(value: str) -> list[str]:
    if value.lower() == "all":
        return HISTORICAL_MODEL_ORDER.copy()
    models = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(models).difference(MODEL_ALIASES))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    design_parser = subparsers.add_parser("generate-design")
    design_parser.add_argument("--samples", type=int, default=300)
    design_parser.add_argument("--seed", type=int, default=42)
    design_parser.add_argument("--models", default="all")
    design_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "experimental_design.csv")

    validate_parser = subparsers.add_parser("validate-inputs")
    validate_parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    validate_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    run_parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs")
    run_parser.add_argument("--design", type=Path, default=DEFAULT_OUTPUT / "experimental_design.csv")
    run_parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    run_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--batch-size", type=int, default=16)
    run_parser.add_argument("--max-configs", type=int)
    run_parser.add_argument("--max-items", type=int)

    args = parser.parse_args()
    if args.command == "generate-design":
        design = generate_design(args.samples, args.seed, parse_models(args.models))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        design.to_csv(args.output, index=False)
        print(f"Saved {len(design):,} rows to {args.output}")
    elif args.command == "validate-inputs":
        prompts = load_prompts(args.prompts)
        items, grouped = load_corpus(args.corpus)
        print(f"Prompts valid: {len(prompts['instructions'])} instructions")
        print(f"Corpus valid: {len(items):,} items; " + ", ".join(f"{k}={len(v)}" for k, v in grouped.items()))
    else:
        run(
            args.design,
            args.prompts,
            args.corpus,
            args.output,
            args.model,
            [int(part) for part in args.gpus.split(",")],
            args.batch_size,
            args.max_configs,
            args.max_items,
        )


if __name__ == "__main__":
    main()
