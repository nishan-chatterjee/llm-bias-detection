#!/usr/bin/env python3
"""Generate and run the Political Compass MCQ experiment.

The released entry point preserves the original experimental design and
candidate-only softmax while replacing workstation-specific paths, GPU IDs,
model lists, and defaults with explicit command-line arguments.

Examples
--------
Generate the paper design::

    python tasks/political-compass/mcq.py generate-design

Run one checkpoint/quantization on GPUs 0 and 1::

    python tasks/political-compass/mcq.py run \
      --models gemma-3-27b-it --quantizations bf16 --gpus 0,1

Use ``--max-configs 1 --max-questions 2`` for a tiny execution smoke test.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
DEFAULT_DATA_DIR = TASK_DIR / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "serve"
DEFAULT_OUTPUT_DIR = TASK_DIR / "output" / "mcq"

PRIMARY_MODELS = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]
# Global block positions used by the completed run. Qwen3-4B was appended after
# the original ten-model schedule, so its released IDs begin at block 10.
RELEASE_MODEL_BLOCKS = {
    "gemma-3-27b-it": 1,
    "Qwen3-32B": 2,
    "Qwen3-8B": 3,
    "gemma-3-12b-it": 4,
    "Qwen3-14B": 6,
    "gemma-3-1b-it": 8,
    "gemma-3-4b-it": 9,
    "Qwen3-4B": 10,
}
QUANTIZATIONS = ["bf16", "8bit", "4bit"]
LANGUAGES = [
    "bulgarian",
    "czech",
    "english",
    "french",
    "german",
    "italian",
    "persian",
    "polish",
    "portuguese",
    "romanian",
    "russian",
    "slovene",
    "spanish",
    "turkish",
]
IDEOLOGIES = [
    "base",
    "libertarian_left",
    "libertarian_right",
    "authoritarian_left",
    "authoritarian_right",
    "centrism",
]
KEY_TYPES = ["numeric", "numeric_zero_index", "lowercase_start", "uppercase_start"]


def comma_list(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def load_model_config(models_dir: Path = DEFAULT_MODELS_DIR) -> dict[str, Any]:
    path = models_dir / "model_config.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_source(
    alias: str,
    models_dir: Path,
    config: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve a populated local checkpoint first, then its pinned Hub ID."""
    local = models_dir / alias
    if (local / "config.json").exists():
        return str(local), None
    model_cfg = config.get(alias)
    if not model_cfg:
        raise KeyError(f"No model_config.json entry for {alias!r}")
    repo_id = model_cfg.get("repo_id")
    if not repo_id:
        raise ValueError(f"Model {alias!r} has neither a local checkpoint nor repo_id")
    return str(repo_id), model_cfg.get("revision")


def map_factors(sample: np.ndarray, languages: list[str]) -> dict[str, Any]:
    context_id = None
    if sample[1] >= 0.5:
        context_id = min(int((sample[1] - 0.5) * 10), 4)
    return {
        "language": languages[min(int(sample[0] * len(languages)), len(languages) - 1)],
        "context_id": context_id,
        "instr_type": "question_first" if sample[2] < 0.5 else "options_first",
        "instr_idx": min(int(sample[3] * 5), 4),
        "persona_class": "short" if sample[4] < 0.5 else "long",
        "persona_idx": min(int(sample[5] * 5), 4),
        "key_type": KEY_TYPES[min(int(sample[6] * len(KEY_TYPES)), len(KEY_TYPES) - 1)],
        "perm_id": min(int(sample[7] * 4), 3),
    }


def generate_experimental_design(
    samples_per_block: int = 300,
    seed: int = 42,
    models: Iterable[str] = PRIMARY_MODELS,
    languages: Iterable[str] = LANGUAGES,
) -> pd.DataFrame:
    """Generate one shared 8-dimensional LHS block per model/persona block."""
    selected_models = list(models)
    selected_languages = list(languages)
    if not selected_models or not selected_languages:
        raise ValueError("At least one model and language are required")
    if samples_per_block < 1:
        raise ValueError("samples_per_block must be positive")

    # Use the legacy ``seed`` keyword deliberately. SciPy's newer ``rng``
    # keyword initializes a different stream for the same integer and does not
    # reproduce the completed experiment.
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    base_samples = sampler.random(n=samples_per_block)

    records: list[dict[str, Any]] = []
    for model in selected_models:
        for ideology in IDEOLOGIES:
            for sample in base_samples:
                record = map_factors(sample, selected_languages)
                record["model"] = model
                record["ideology"] = ideology
                if ideology == "base":
                    record["persona_class"] = None
                    record["persona_idx"] = None
                records.append(record)
    design = pd.DataFrame(records)
    block_size = len(IDEOLOGIES) * samples_per_block
    within_model = design.groupby("model", sort=False).cumcount().to_numpy(dtype=int)
    model_offsets = design["model"].map(RELEASE_MODEL_BLOCKS).to_numpy(dtype=int) * block_size
    design.insert(0, "config_id", model_offsets + within_model)
    return design


def assemble_prompt_for_question(
    config_row: pd.Series | dict[str, Any],
    question: dict[str, Any],
    prompts: dict[str, Any],
) -> tuple[str, list[str]]:
    """Assemble the exact condition prompt and permuted answer-key order."""
    row = dict(config_row)
    segments: list[str] = []

    context_id = row.get("context_id")
    if context_id is not None and not pd.isna(context_id) and int(context_id) >= 0:
        segments.append(prompts["contexts"][int(context_id)]["text"])

    if row["ideology"] != "base":
        persona_class = str(row["persona_class"])
        persona_idx = int(row["persona_idx"])
        template = prompts["persona_templates"][persona_class][persona_idx]
        insertion = prompts["ideology_insertions"][row["ideology"]][template["id"]]
        segments.append(template["text"].format(**insertion))

    permutation = int(row["perm_id"])
    labels = list(question["choices"])
    keys = list(prompts["experiment_settings"]["answer_keys"][row["key_type"]])
    if permutation in {2, 3}:
        labels.reverse()
    if permutation in {1, 3}:
        keys.reverse()

    options = "\n" + "\n".join(f"{key}. {label}" for key, label in zip(keys, labels))
    instruction = prompts["instructions"][row["instr_type"]][int(row["instr_idx"])]["text"]
    segments.append(instruction.format(question=question["statement"], options_formatted=options))
    return "\n\n".join(segments), keys


def load_language_resources(data_dir: Path, languages: Iterable[str]) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    missing: list[str] = []
    for language in languages:
        prompt_path = data_dir / "prompts" / f"{language}.json"
        question_path = data_dir / "questions" / f"{language}.json"
        if not prompt_path.exists() or not question_path.exists():
            missing.append(language)
            continue
        with prompt_path.open(encoding="utf-8") as handle:
            prompts = json.load(handle)
        with question_path.open(encoding="utf-8") as handle:
            questions = json.load(handle)
        resources[language] = {"prompts": prompts, "questions": questions}
    if missing:
        raise FileNotFoundError(
            "Missing prompt/question resources for: " + ", ".join(missing)
        )
    return resources


def result_path(output_dir: Path, model: str, quantization: str) -> Path:
    safe_model = model.replace("/", "_")
    return output_dir / f"results_{safe_model}_{quantization}.csv"


def completed_config_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["config_id"])["config_id"].astype(int))
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        return set()


def append_configuration(
    path: Path,
    config_row: pd.Series,
    quantization: str,
    question_results: list[dict[str, Any]],
) -> None:
    row: dict[str, Any] = {
        "config_id": int(config_row["config_id"]),
        "model": config_row["model"],
        "quantization": quantization,
        "language": config_row["language"],
        "ideology": config_row["ideology"],
        "context_id": -1 if pd.isna(config_row["context_id"]) else int(config_row["context_id"]),
        "instr_type": config_row["instr_type"],
        "instr_idx": int(config_row["instr_idx"]),
        "persona_class": "" if pd.isna(config_row["persona_class"]) else config_row["persona_class"],
        "persona_idx": -1 if pd.isna(config_row["persona_idx"]) else int(config_row["persona_idx"]),
        "key_type": config_row["key_type"],
        "perm_id": int(config_row["perm_id"]),
    }
    for result in question_results:
        for answer_index, probability in enumerate(result["probabilities"]):
            row[f"q{result['question_id']}_prob_ans{answer_index}"] = probability
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def candidate_softmax(logits: Any) -> Any:
    """Softmax over only the four candidate-key logits, as in the experiment."""
    import torch

    return torch.nn.functional.softmax(logits, dim=0)


def load_transformers_model(
    model_source: str,
    revision: str | None,
    quantization: str,
    attention_implementation: str,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    common: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        common["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, **common)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        **common,
        "device_map": "auto",
        "torch_dtype": compute_dtype,
    }
    if attention_implementation != "auto":
        model_kwargs["attn_implementation"] = attention_implementation
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != "bf16":
        raise ValueError(f"Unsupported quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()
    return tokenizer, model


def score_configuration(
    tokenizer: Any,
    model: Any,
    config_row: pd.Series,
    prompts: dict[str, Any],
    questions: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    prompts_and_keys = [
        assemble_prompt_for_question(config_row, question, prompts) for question in questions
    ]
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(questions), batch_size):
        chunk_questions = questions[start : start + batch_size]
        chunk = prompts_and_keys[start : start + batch_size]
        texts = [entry[0] for entry in chunk]
        key_lists = [entry[1] for entry in chunk]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3000,
        ).to(model.device)
        with torch.no_grad():
            model_output = model(**inputs)
        last_indices = inputs.attention_mask.sum(dim=1) - 1
        next_logits = model_output.logits[
            torch.arange(len(texts), device=last_indices.device), last_indices, :
        ]

        for index, (question, keys) in enumerate(zip(chunk_questions, key_lists)):
            candidate_ids = []
            for key in keys:
                token_ids = tokenizer.encode(key, add_special_tokens=False)
                if not token_ids:
                    raise ValueError(f"Candidate key {key!r} tokenized to an empty sequence")
                candidate_ids.append(token_ids[-1])
            probabilities = candidate_softmax(next_logits[index, candidate_ids])
            values = probabilities.detach().cpu().float().tolist()
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
                raise RuntimeError("Candidate probabilities do not sum to one")
            outputs.append({"question_id": question["id"], "probabilities": values})
    return outputs


def run(args: argparse.Namespace) -> None:
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    data_dir = Path(args.data_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    output_dir = Path(args.output).resolve()
    design_path = Path(args.design).resolve() if args.design else output_dir / "experimental_design.csv"
    if not design_path.exists():
        raise FileNotFoundError(
            f"Design not found: {design_path}. Run the generate-design subcommand first."
        )

    design = pd.read_csv(design_path)
    models = comma_list(args.models, PRIMARY_MODELS)
    quantizations = comma_list(args.quantizations, QUANTIZATIONS)
    unsupported = sorted(set(models) - set(PRIMARY_MODELS))
    if unsupported:
        raise ValueError("MCQ release defaults support only primary models: " + ", ".join(unsupported))
    if set(quantizations) - set(QUANTIZATIONS):
        raise ValueError("Quantizations must be drawn from bf16,8bit,4bit")

    design = design[design["model"].isin(models)]
    resources = load_language_resources(data_dir, sorted(design["language"].unique()))
    config = load_model_config(models_dir)

    for model_name in models:
        model_rows = design[design["model"] == model_name].sort_values("config_id")
        for quantization in quantizations:
            path = result_path(output_dir, model_name, quantization)
            done = completed_config_ids(path)
            pending = model_rows[~model_rows["config_id"].isin(done)]
            if args.max_configs is not None:
                pending = pending.head(args.max_configs)
            if pending.empty:
                print(f"complete: {model_name} {quantization}")
                continue

            source, revision = resolve_model_source(model_name, models_dir, config)
            print(f"loading {model_name} ({quantization}) from {source}@{revision or 'local'}")
            tokenizer, model = load_transformers_model(
                source,
                revision,
                quantization,
                args.attention_implementation,
            )
            try:
                for _, row in pending.iterrows():
                    resource = resources[row["language"]]
                    questions = resource["questions"]
                    if args.max_questions is not None:
                        questions = questions[: args.max_questions]
                    results = score_configuration(
                        tokenizer,
                        model,
                        row,
                        resource["prompts"],
                        questions,
                        args.batch_size,
                    )
                    if len(results) != len(questions):
                        raise RuntimeError(
                            f"Configuration {row['config_id']} produced {len(results)}/{len(questions)} questions"
                        )
                    append_configuration(path, row, quantization, results)
                    print(f"saved config {int(row['config_id'])} -> {path.name}")
            finally:
                del model
                del tokenizer
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser("generate-design", help="generate deterministic LHS schedule")
    design.add_argument("--samples", type=int, default=300, help="samples per model/persona block")
    design.add_argument("--seed", type=int, default=42)
    design.add_argument("--models", help="comma-separated aliases; default: all primary models")
    design.add_argument("--languages", help="comma-separated languages; default: all 14")
    design.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "experimental_design.csv"))

    execute = subparsers.add_parser("run", help="run one or more schedule subsets")
    execute.add_argument("--models", help="comma-separated aliases; default: all primary models")
    execute.add_argument("--quantizations", default="bf16,8bit,4bit")
    execute.add_argument("--gpus", default="0", help="CUDA-visible GPU IDs, e.g. 0 or 0,1")
    execute.add_argument("--batch-size", type=int, default=16)
    execute.add_argument("--max-configs", type=int, help="limit pending configs (smoke tests)")
    execute.add_argument("--max-questions", type=int, help="limit questions (smoke tests)")
    execute.add_argument("--design", help="schedule CSV; defaults to <output>/experimental_design.csv")
    execute.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    execute.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    execute.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    execute.add_argument(
        "--attention-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate-design":
        models = comma_list(args.models, PRIMARY_MODELS)
        languages = comma_list(args.languages, LANGUAGES)
        unknown_models = sorted(set(models) - set(PRIMARY_MODELS))
        unknown_languages = sorted(set(languages) - set(LANGUAGES))
        if unknown_models:
            parser.error("unsupported models: " + ", ".join(unknown_models))
        if unknown_languages:
            parser.error("unsupported languages: " + ", ".join(unknown_languages))
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = generate_experimental_design(args.samples, args.seed, models, languages)
        frame.to_csv(output, index=False)
        print(f"wrote {len(frame):,} configurations to {output}")
        return
    run(args)


if __name__ == "__main__":
    main()
