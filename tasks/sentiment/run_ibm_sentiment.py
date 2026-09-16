#!/usr/bin/env python3
"""Run the IBM Claim Stance *topic-sentiment* experiment.

This release runner intentionally supports only the 30-item topic-sentiment
task used in the paper.  Stance datasets and stance experiments are excluded.
Its operational shape is:
  - LHS prompt-configuration design generation
  - one JSONL checkpoint per model with item-level resume
  - scheduling model workers across explicit GPU IDs
  - vLLM scoring through tasks/political-compass/model_loader.py

It deliberately uses only the eight primary base models. Qwen thinking variants
are not expanded for this MCQ pipeline.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
POLITICAL_COMPASS_DIR = PROJECT_ROOT / "tasks" / "political-compass"
if str(POLITICAL_COMPASS_DIR) not in sys.path:
    sys.path.insert(0, str(POLITICAL_COMPASS_DIR))

import model_loader as vllm_loader

MODEL_DIR = PROJECT_ROOT / "models" / "serve"
PROMPTS_DIR = SCRIPT_DIR / "data" / "prompts"
QUESTIONS_DIR = SCRIPT_DIR / "data" / "questions"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "output" / "ibm-sentiment"

DEFAULT_SAMPLES_PER_BLOCK = 300
MAX_RUN_TIME_SECONDS = 3600 * 95
DEFAULT_ITEM_RETRIES = 1

PRIMARY_BASE_MODELS = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
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

EXPERIMENT = "ibm_sentiment"
EXPERIMENT_SPEC = {
    "slug": "ibm-sentiment",
    "dataset": "ibm-research/claim_stance",
    "dataset_revision": "ec4e2c2ec3e0c70087c67a28a7bce58b682b8109",
    "task_type": "sentiment",
    "prompt_file": "english_2-way-sentiment.json",
    "choices": ["NEGATIVE", "POSITIVE"],
}


def parse_csv_arg(value):
    if not value:
        return None
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        return None
    if any(part.lower() == "all" for part in values):
        return list(PRIMARY_BASE_MODELS)
    return values


def _jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _safe_int(value, default=-1):
    if pd.isna(value):
        return default
    return int(value)


def _sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def resolve_output_root(output_path=None):
    if output_path:
        return Path(output_path)
    return DEFAULT_OUTPUT_BASE


def get_results_path(output_root, model_variant):
    return Path(output_root) / f"{model_variant}.jsonl"


def expand_models(model_filter=None):
    requested = set(model_filter or [])
    models = []
    for model_name in PRIMARY_BASE_MODELS:
        if requested and model_name not in requested:
            continue
        models.append(
            {
                "model": model_name,
                "model_variant": model_name,
                "generation_mode": "standard",
            }
        )
    return models


def detect_requested_gpus(requested_gpus):
    """
    Detect requested physical GPUs one at a time.

    Probing a non-contiguous mask such as 0,1,3 can trigger PyTorch lazy CUDA
    capability bugs on nodes with one unhealthy GPU. Single-GPU probes are much
    more robust, and the current H100 launch scripts schedule these models as
    one vLLM instance per GPU.
    """
    if not requested_gpus:
        return 0, 0.0, "none", []

    script = (
        "import json, torch; "
        "r = {'count': torch.cuda.device_count(), "
        "'vram': torch.cuda.get_device_properties(0).total_memory / (1024**3), "
        "'name': torch.cuda.get_device_properties(0).name} "
        "if torch.cuda.is_available() else "
        "{'count': 0, 'vram': 0, 'name': 'none'}; "
        "print(json.dumps(r))"
    )
    detected = []
    usable_gpu_ids = []
    timeout_seconds = int(os.environ.get("GPU_DETECT_TIMEOUT", "180"))
    for gpu_id in requested_gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
        except Exception as exc:
            print(f"GPU detection failed for GPU {gpu_id}: {exc}")
            continue

        if result.returncode != 0:
            print(f"WARNING: skipping GPU {gpu_id}; detection stderr:\n{result.stderr}")
            continue

        info = json.loads(result.stdout.strip())
        if int(info.get("count", 0)) < 1:
            print(f"WARNING: skipping GPU {gpu_id}; no CUDA device visible")
            continue
        detected.append(info)
        usable_gpu_ids.append(gpu_id)

    if not detected:
        return 0, 0.0, "error", []

    vram = min(float(info["vram"]) for info in detected)
    names = sorted({str(info["name"]) for info in detected})
    return len(detected), vram, "/".join(names), usable_gpu_ids


def load_prompt_dict(experiment):
    if experiment != EXPERIMENT:
        raise ValueError(f"This release runner only supports {EXPERIMENT}")
    prompt_file = EXPERIMENT_SPEC["prompt_file"]
    prompt_path = PROMPTS_DIR / prompt_file
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)

    persona_source = prompt_dict.get("persona_source") or prompt_dict.get("meta_info", {}).get("persona_source")
    if persona_source:
        persona_path = (prompt_path.parent / persona_source).resolve()
        with open(persona_path, "r", encoding="utf-8") as f:
            persona_dict = json.load(f)
        for key in [
            "instructions_for_translators",
            "persona_templates",
            "ideology_insertions",
        ]:
            if key not in prompt_dict and key in persona_dict:
                prompt_dict[key] = persona_dict[key]

    return prompt_dict


def map_factors(sample):
    factors = {}

    if sample[0] < 0.5:
        factors["context_id"] = None
    else:
        ctx_idx = int((sample[0] - 0.5) * 2 * 5)
        factors["context_id"] = min(ctx_idx, 4)

    factors["instr_type"] = "question_first" if sample[1] < 0.5 else "options_first"
    factors["instr_idx"] = min(int(sample[2] * 5), 4)
    factors["persona_class"] = "short" if sample[3] < 0.5 else "long"
    factors["persona_idx"] = min(int(sample[4] * 5), 4)
    factors["key_type"] = KEY_TYPES[min(int(sample[5] * len(KEY_TYPES)), len(KEY_TYPES) - 1)]
    factors["perm_id"] = min(int(sample[6] * 4), 3)
    return factors


def generate_experimental_design(samples_per_block=DEFAULT_SAMPLES_PER_BLOCK, seed=42, model_filter=None):
    models = expand_models(model_filter)
    try:
        sampler = qmc.LatinHypercube(d=7, rng=seed)  # type: ignore[call-arg]
    except TypeError:
        sampler = qmc.LatinHypercube(d=7, seed=seed)  # type: ignore[call-arg]
    base_block_samples = sampler.random(n=samples_per_block)

    records = []
    for model_spec in models:
        for ideology in IDEOLOGIES:
            for sample in base_block_samples:
                config = map_factors(sample)
                config.update(model_spec)
                config["experiment"] = EXPERIMENT
                config["language"] = "english"
                config["prompt_file"] = EXPERIMENT_SPEC["prompt_file"]
                config["task_type"] = EXPERIMENT_SPEC["task_type"]
                config["ideology"] = ideology
                if ideology == "base":
                    config["persona_class"] = None
                    config["persona_idx"] = None
                records.append(config)

    df = pd.DataFrame(records)
    df["config_id"] = range(len(df))
    return df


def _make_item(index, source_item_id, experiment, text, target, gold_label, gold_label_id, topic=None, claim=None, metadata=None):
    if experiment != EXPERIMENT:
        raise ValueError(f"This release runner only supports {EXPERIMENT}")
    spec = EXPERIMENT_SPEC
    return {
        "item_index": int(index),
        "source_item_id": str(source_item_id),
        "experiment": experiment,
        "dataset": spec["dataset"],
        "task_type": spec["task_type"],
        "text": text or "",
        "topic": topic or target or text or "",
        "target": target or "",
        "claim": claim or "",
        "gold_label": gold_label,
        "gold_label_id": int(gold_label_id),
        "choices": list(spec["choices"]),
        "metadata": metadata or {},
    }


def _load_ibm_rows():
    topic_csv = QUESTIONS_DIR / "ibm-claim-stance" / "ibm_test_topics.csv"
    if topic_csv.exists():
        return pd.read_csv(topic_csv).to_dict(orient="records")

    # Compatibility fallback for a local Hugging Face datasets Arrow cache.
    arrow_paths = sorted((QUESTIONS_DIR / "ibm-claim-stance").glob("**/claim_stance-test.arrow"))
    if not arrow_paths:
        raise FileNotFoundError(
            "No ibm_test_topics.csv or claim_stance-test.arrow found under "
            f"{QUESTIONS_DIR / 'ibm-claim-stance'}. See the task README."
        )

    import pyarrow.ipc as ipc

    with open(arrow_paths[0], "rb") as f:
        table = ipc.open_stream(f).read_all()
    return table.to_pylist()


def load_ibm_sentiment_items(experiment):
    sentiment_to_label = {-1: ("NEGATIVE", 0), 1: ("POSITIVE", 1)}
    items = []
    seen_topic_ids = set()
    for row in _load_ibm_rows():
        topic_id = row["topicId"]
        if topic_id in seen_topic_ids:
            continue
        seen_topic_ids.add(topic_id)
        label, label_id = sentiment_to_label[int(row["topicSentiment"])]
        items.append(
            _make_item(
                len(items),
                topic_id,
                experiment,
                row["topicText"],
                row["topicTarget"],
                label,
                label_id,
                topic=row["topicText"],
                metadata={
                    "topic_id": topic_id,
                    "topic_sentiment": row["topicSentiment"],
                    "unique_topic": True,
                },
            )
        )
    if len(items) != 30:
        raise ValueError(f"Expected 30 unique IBM test topics, found {len(items)}")
    return items


def load_experiment_items(experiment=EXPERIMENT):
    return load_ibm_sentiment_items(experiment)


def assemble_prompt_for_item(config_row, item, prompts_dict):
    segments = []

    if pd.notna(config_row.get("context_id")):
        ctx_id = int(config_row["context_id"])
        try:
            segments.append(prompts_dict["contexts"][ctx_id]["text"])
        except (IndexError, KeyError):
            pass

    if config_row["ideology"] != "base":
        p_class = config_row["persona_class"]
        p_idx = int(config_row["persona_idx"])
        try:
            template = prompts_dict["persona_templates"][p_class][p_idx]
            insertions = prompts_dict["ideology_insertions"][config_row["ideology"]][template["id"]]
            segments.append(template["text"].format(**insertions))
        except (KeyError, IndexError):
            pass

    perm_id = int(config_row["perm_id"])
    rev_labels = perm_id in [2, 3]
    rev_keys = perm_id in [1, 3]

    labels = list(item["choices"])
    keys = list(prompts_dict["experiment_settings"]["answer_keys"][config_row["key_type"]])

    if rev_labels:
        labels = labels[::-1]
    if rev_keys:
        keys = keys[::-1]

    options_formatted = "\n" + "\n".join(f"{key}. {label}" for key, label in zip(keys, labels))
    instr_entries = prompts_dict["instructions"][config_row["instr_type"]]
    instr_template = instr_entries[int(config_row["instr_idx"]) % len(instr_entries)]["text"]

    final_instr = instr_template.format(
        text=item.get("text", ""),
        topic=item.get("topic", ""),
        target=item.get("target", ""),
        claim=item.get("claim", ""),
        options_formatted=options_formatted,
    )
    segments.append(final_instr)

    prompt_text = "\n\n".join(segments)
    candidates = [
        {"index": idx, "key": key, "text": label, "score_text": f" {key}"}
        for idx, (key, label) in enumerate(zip(keys, labels))
    ]
    return prompt_text, candidates


def build_items_for_config(config_row, prompts_dict, experiment_items, num_items):
    config_id = int(config_row["config_id"])
    items = []
    for item in experiment_items:
        prompt_text, candidates = assemble_prompt_for_item(config_row, item, prompts_dict)
        item_uid = config_id * num_items + int(item["item_index"])
        items.append(
            {
                "item_id": f"cfg{config_id}_i{int(item['item_index'])}",
                "item_uid": int(item_uid),
                "input_index": int(item["item_index"]),
                "prompt": prompt_text,
                "classification_suffix": "",
                "candidates": candidates,
                "metadata": {
                    "item_uid": int(item_uid),
                    "config_id": config_id,
                    "item_index": int(item["item_index"]),
                    "source_item_id": item["source_item_id"],
                    "experiment": item["experiment"],
                    "dataset": item["dataset"],
                    "task_type": item["task_type"],
                    "target": item["target"],
                    "topic": item["topic"],
                    "claim_present": bool(item.get("claim")),
                    "gold_label": item["gold_label"],
                    "gold_label_id": int(item["gold_label_id"]),
                    "prompt_sha1": _sha1_text(prompt_text),
                    "item_metadata": dict(item.get("metadata", {})),
                    "input_text": item.get("text", ""),
                    "claim": item.get("claim", ""),
                },
            }
        )
    return items


def iter_result_records(results_path):
    if not results_path.exists():
        return
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_completed_success_uids(results_path):
    completed = set()
    if not results_path.exists():
        return completed
    for record in iter_result_records(results_path):
        if record.get("error"):
            continue
        item_uid = record.get("item_uid")
        if item_uid is not None:
            completed.add(int(item_uid))
    return completed


def count_successes(results_path):
    return len(load_completed_success_uids(results_path))


def append_records_to_jsonl(results_path, config_row, records, save_prompts=False, save_input_text=False):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    base_meta = {
        "config_id": int(config_row["config_id"]),
        "experiment": config_row["experiment"],
        "task_type": config_row["task_type"],
        "prompt_file": config_row["prompt_file"],
        "model": config_row["model"],
        "model_variant": config_row["model_variant"],
        "generation_mode": config_row["generation_mode"],
        "language": config_row["language"],
        "ideology": config_row["ideology"],
        "context_id": _safe_int(config_row.get("context_id")),
        "instr_type": config_row["instr_type"],
        "instr_idx": int(config_row["instr_idx"]),
        "persona_class": config_row["persona_class"] if pd.notna(config_row.get("persona_class")) else "",
        "persona_idx": _safe_int(config_row.get("persona_idx")),
        "key_type": config_row["key_type"],
        "perm_id": int(config_row["perm_id"]),
        "answer_mode": "mcq",
        "inference_backend": "vllm",
        "quantization": "model-native",
        "written_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(results_path, "a", encoding="utf-8") as f:
        for record in records:
            metadata = dict(record.get("metadata", {}))
            candidate_keys = record.get("candidate_keys") or []
            candidate_texts = record.get("candidate_texts") or []
            pred_key = record.get("classification_pred_key")
            pred_label = None
            if pred_key is not None:
                pred_label = dict(zip(candidate_keys, candidate_texts)).get(pred_key)
            gold_label = metadata.get("gold_label")

            output_record = dict(base_meta)
            output_record.update(
                {
                    "item_uid": metadata.get("item_uid"),
                    "item_id": record["item_id"],
                    "item_index": metadata.get("item_index"),
                    "source_item_id": metadata.get("source_item_id"),
                    "dataset": metadata.get("dataset"),
                    "target": metadata.get("target"),
                    "topic": metadata.get("topic"),
                    "claim_present": metadata.get("claim_present"),
                    "gold_label": gold_label,
                    "gold_label_id": metadata.get("gold_label_id"),
                    "candidate_keys": candidate_keys,
                    "candidate_texts": candidate_texts,
                    "classification_logprobs": record.get("classification_logprobs"),
                    "classification_probs": record.get("classification_probs"),
                    "classification_pred_key": pred_key,
                    "classification_pred_label": pred_label,
                    "is_correct": None if pred_label is None or gold_label is None else pred_label == gold_label,
                    "prompt_sha1": metadata.get("prompt_sha1"),
                    "item_metadata": metadata.get("item_metadata", {}),
                    "attempt": record.get("attempt", 1),
                    "error": record.get("error"),
                }
            )
            if save_prompts:
                output_record["prompt"] = record.get("prompt")
            if save_input_text:
                output_record["input_text"] = metadata.get("input_text", "")
                output_record["claim"] = metadata.get("claim", "")
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def split_task_df(task_df, num_shards):
    if num_shards <= 1:
        return [task_df.copy()]
    ordered = task_df.sort_values("config_id").reset_index(drop=True)
    shards = []
    for shard_idx in range(num_shards):
        shard = ordered.iloc[shard_idx::num_shards].copy()
        if not shard.empty:
            shards.append(shard)
    return shards


def worker_process(
    gpu_ids,
    model_name,
    model_variant,
    generation_mode,
    task_df,
    model_config,
    experiment,
    classify_batch_size,
    results_path,
    write_lock,
    stop_event,
    max_item_retries,
    gpu_memory_utilization,
    max_model_len,
    enforce_eager,
    save_prompts,
    save_input_text,
    max_items,
):
    print(
        f"[{datetime.now()}] START {model_variant} | GPUs {gpu_ids} | "
        f"configs={len(task_df)} | experiment={experiment}"
    )

    runner = None
    try:
        model_source, _revision = vllm_loader.resolve_model_source(
            model_name, MODEL_DIR, model_config
        )
        runner = vllm_loader.VLLMRunner(
            model_source,
            model_name,
            generation_mode,
            gpu_ids,
            model_config,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        prompts_dict = load_prompt_dict(experiment)
        experiment_items = load_experiment_items(experiment)
        num_items = len(experiment_items)
        if max_items is not None:
            experiment_items = experiment_items[:max_items]
        completed_uids = load_completed_success_uids(results_path)

        for _, config_row in task_df.sort_values("config_id").iterrows():
            if stop_event.is_set():
                break

            config_id = int(config_row["config_id"])
            first_uid = config_id * num_items
            if all(
                (first_uid + item_idx) in completed_uids
                for item_idx in range(len(experiment_items))
            ):
                continue

            items = build_items_for_config(config_row, prompts_dict, experiment_items, num_items)
            item_lookup = {item["item_id"]: item for item in items}
            pending_items = [
                item for item in items if int(item["metadata"]["item_uid"]) not in completed_uids
            ]
            if not pending_items:
                continue

            for attempt_idx in range(1, max_item_retries + 2):
                try:
                    records = runner.run_items(
                        pending_items,
                        answer_mode="mcq",
                        batch_size=classify_batch_size,
                        classify_batch_size=classify_batch_size,
                    )
                except Exception as exc:
                    print(
                        f"[{datetime.now()}] ERROR {model_variant} | config={config_id} "
                        f"| attempt={attempt_idx}: {exc}"
                    )
                    records = []
                    for item in pending_items:
                        records.append(
                            {
                                "item_id": item["item_id"],
                                "input_index": item["input_index"],
                                "prompt": item["prompt"],
                                "answer_mode": "mcq",
                                "classification_suffix": "",
                                "candidate_keys": [candidate["key"] for candidate in item.get("candidates", [])],
                                "candidate_texts": [candidate.get("text", "") for candidate in item.get("candidates", [])],
                                "metadata": dict(item.get("metadata", {})),
                                "classification_logprobs": None,
                                "classification_probs": None,
                                "classification_pred_key": None,
                                "error": f"worker_config_failed: {exc}",
                            }
                        )

                success_records = []
                failed_records = []
                for record in records:
                    record["attempt"] = attempt_idx
                    if record.get("error"):
                        failed_records.append(record)
                    else:
                        success_records.append(record)

                if success_records:
                    with write_lock:
                        append_records_to_jsonl(
                            results_path,
                            config_row,
                            success_records,
                            save_prompts=save_prompts,
                            save_input_text=save_input_text,
                        )
                    for record in success_records:
                        completed_uids.add(int(record["metadata"]["item_uid"]))

                if not failed_records:
                    break

                if attempt_idx <= max_item_retries:
                    print(
                        f"[{datetime.now()}] RETRY {model_variant} | config={config_id} "
                        f"| remaining_failed={len(failed_records)} | next_attempt={attempt_idx + 1}"
                    )
                    pending_items = [item_lookup[record["item_id"]] for record in failed_records]
                else:
                    with write_lock:
                        append_records_to_jsonl(
                            results_path,
                            config_row,
                            failed_records,
                            save_prompts=save_prompts,
                            save_input_text=save_input_text,
                        )

    finally:
        if runner is not None:
            runner.close()
        print(f"[{datetime.now()}] DONE {model_variant}")


def run_experiments(
    available_gpus,
    experiment,
    model_filter=None,
    dp_threshold=vllm_loader.DP_THRESHOLD,
    min_tp_size=vllm_loader.DEFAULT_MIN_TP_SIZE,
    long_gen_token_threshold=vllm_loader.DEFAULT_LONG_GEN_TOKEN_THRESHOLD,
    long_gen_min_tp_size=vllm_loader.DEFAULT_LONG_GEN_MIN_TP_SIZE,
    classify_batch_size=vllm_loader.DEFAULT_CLASSIFY_BATCH_SIZE,
    gpu_memory_utilization=vllm_loader.DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=vllm_loader.DEFAULT_VLLM_MAX_MODEL_LEN,
    enforce_eager=False,
    output_path=None,
    progress_log_name="_progress_mcq_v1.log",
    max_item_retries=DEFAULT_ITEM_RETRIES,
    save_prompts=False,
    save_input_text=False,
    max_configs=None,
    max_items=None,
):
    if not 0.0 < dp_threshold <= 1.0:
        raise ValueError(f"dp_threshold must be in (0, 1], got {dp_threshold}")

    start_time = time.time()
    mp.set_start_method("spawn", force=True)

    output_root = resolve_output_root(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / progress_log_name

    design_path = output_root / "experimental_design.csv"
    if not design_path.exists():
        print(f"ERROR: No experimental design found at {design_path}")
        print("Run: python tasks/sentiment/run_ibm_sentiment.py generate-design")
        sys.exit(1)

    df_schedule = pd.read_csv(design_path)
    df_schedule["experiment"] = df_schedule["experiment"].astype(str)
    df_schedule = df_schedule[df_schedule["experiment"] == experiment]
    if df_schedule.empty:
        print(f"ERROR: No rows for experiment={experiment} in {design_path}")
        sys.exit(1)

    if model_filter:
        requested = set(model_filter)
        df_schedule = df_schedule[df_schedule["model"].isin(requested)]
        if df_schedule.empty:
            print("ERROR: No matching models found in the schedule.")
            sys.exit(1)
        order_map = {name: idx for idx, name in enumerate(model_filter)}
        df_schedule["_model_order"] = df_schedule["model"].map(order_map).fillna(len(order_map)).astype(int)
        df_schedule = (
            df_schedule
            .sort_values(["_model_order", "config_id"], kind="stable")
            .drop(columns=["_model_order"])
        )

    if max_configs is not None:
        if max_configs < 1:
            raise ValueError("max_configs must be positive")
        df_schedule = (
            df_schedule.groupby("model_variant", sort=False, group_keys=False)
            .head(max_configs)
            .copy()
        )
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")

    experiment_items = load_experiment_items(experiment)
    num_items = min(len(experiment_items), max_items) if max_items is not None else len(experiment_items)
    model_config = vllm_loader.load_model_config(MODEL_DIR)

    detected_gpus, vram_per_gpu, gpu_name, usable_gpus = detect_requested_gpus(available_gpus)
    if detected_gpus == 0:
        print("ERROR: No GPUs detected.")
        sys.exit(1)
    if usable_gpus != list(available_gpus):
        print(f"WARNING: requested GPUs {available_gpus}, but only {usable_gpus} passed CUDA detection.")
        available_gpus = usable_gpus

    grouped_specs = []
    for model_variant, task_df in df_schedule.groupby("model_variant", sort=False):
        model_name = task_df["model"].iloc[0]
        generation_mode = task_df["generation_mode"].iloc[0]
        model_cfg = model_config.get(model_name, {})
        effective_min_tp = vllm_loader.compute_effective_min_tp_size(
            model_cfg,
            generation_mode,
            base_min_tp_size=min_tp_size,
            long_gen_token_threshold=long_gen_token_threshold,
            long_gen_min_tp_size=long_gen_min_tp_size,
        )
        params_b = model_cfg.get("params_B")
        if params_b is None:
            schedule_tuple = (1, len(available_gpus), "dp")
        else:
            schedule_tuple = vllm_loader.compute_schedule(
                params_b,
                vram_per_gpu,
                len(available_gpus),
                dp_threshold=dp_threshold,
                min_tp_size=effective_min_tp,
            )

        grouped_specs.append(
            {
                "model": model_name,
                "model_variant": model_variant,
                "generation_mode": generation_mode,
                "task_df": task_df.copy(),
                "results_path": get_results_path(output_root, model_variant),
                "expected_count": len(task_df) * num_items,
                "schedule": schedule_tuple,
            }
        )

    total_groups = len(grouped_specs)
    tracking_entries = []
    completed_groups = 0
    skipped_groups = 0
    permanently_failed = set()
    failure_counts = {}

    header = (
        f"{'=' * 70}\n"
        f"  Experiment : {experiment} ({output_root})\n"
        f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  GPU        : {len(available_gpus)}x {gpu_name} ({vram_per_gpu:.0f} GB each)\n"
        f"  Items      : {num_items}\n"
        f"  Groups     : {total_groups}\n"
        f"  Answer     : mcq\n"
        f"  Schedule   : dp<{dp_threshold:.2f}, min_tp={min_tp_size}, long_gen>={long_gen_token_threshold}->min_tp={long_gen_min_tp_size}\n"
        f"  vLLM mem   : util={gpu_memory_utilization:.2f}, max_len={max_model_len or 'model default'}, eager={enforce_eager}\n"
        f"  Retries    : {max_item_retries}\n"
        f"  Slim JSONL : save_prompts={save_prompts}, save_input_text={save_input_text}\n"
        f"{'=' * 70}\n"
    )

    def _update_tracking(footer_extra=""):
        elapsed = time.time() - start_time
        done = completed_groups + skipped_groups
        remaining = total_groups - done
        eta = f"~{(elapsed / done) * remaining:.0f}s" if done else "calculating..."
        footer = (
            f"\n{'-' * 70}\n"
            f"  Progress : {done}/{total_groups} "
            f"({completed_groups} done, {skipped_groups} skipped, {remaining} remaining)\n"
            f"  Elapsed  : {elapsed:.1f}s\n"
            f"  ETA      : {eta}\n"
            f"  Updated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'-' * 70}"
            + (f"\n{footer_extra}" if footer_extra else "")
        )
        vllm_loader.write_tracking_log(progress_path, header, tracking_entries, footer)

    _update_tracking()

    resume_done = 0
    resume_partial = 0
    resume_lines = []
    for spec in grouped_specs:
        success_count = count_successes(spec["results_path"])
        expected_count = spec["expected_count"]
        if success_count >= expected_count:
            resume_done += 1
            resume_lines.append(f"ALREADY_DONE {spec['model_variant']}: {success_count}/{expected_count}")
        elif success_count > 0:
            resume_partial += 1
            resume_lines.append(f"ALREADY_PARTIAL {spec['model_variant']}: {success_count}/{expected_count}")
        else:
            resume_lines.append(f"PENDING {spec['model_variant']}: 0/{expected_count}")

    tracking_entries.append(
        f"RESUME_SCAN done={resume_done}, partial={resume_partial}, total={total_groups}"
    )
    tracking_entries.extend(resume_lines)
    _update_tracking()

    manager = mp.Manager()
    try:
        while time.time() - start_time < MAX_RUN_TIME_SECONDS:
            remaining_specs = []
            for spec in grouped_specs:
                if spec["model_variant"] in permanently_failed:
                    continue
                success_count = count_successes(spec["results_path"])
                if success_count < spec["expected_count"]:
                    remaining_specs.append(spec)

            completed_groups = total_groups - len(remaining_specs) - len(permanently_failed)
            skipped_groups = len(permanently_failed)

            if not remaining_specs:
                break

            print(
                f"Pass: {len(remaining_specs)} model groups remaining "
                f"({len(permanently_failed)} permanently failed)"
            )

            for spec in remaining_specs:
                if time.time() - start_time > MAX_RUN_TIME_SECONDS:
                    break

                variant = spec["model_variant"]
                pre_count = count_successes(spec["results_path"])
                schedule_tuple = spec["schedule"]
                if schedule_tuple is None:
                    permanently_failed.add(variant)
                    tracking_entries.append(f"SKIP {variant}: model too large for available GPUs")
                    _update_tracking()
                    continue

                tp_size, num_instances, strategy = schedule_tuple
                max_instances = max(1, len(available_gpus) // tp_size)
                num_instances = max(1, min(num_instances, max_instances))
                gpu_groups = [
                    available_gpus[idx * tp_size : (idx + 1) * tp_size]
                    for idx in range(num_instances)
                ]
                task_shards = split_task_df(spec["task_df"], len(gpu_groups))

                tracking_entries.append(
                    f"START {variant}: strategy={strategy}, tp={tp_size}, instances={len(task_shards)}, "
                    f"pre={pre_count}/{spec['expected_count']}"
                )
                _update_tracking()

                stop_event = manager.Event()
                write_lock = manager.Lock()
                processes = []
                try:
                    for shard_df, gpu_ids in zip(task_shards, gpu_groups):
                        process = mp.Process(
                            target=worker_process,
                            args=(
                                gpu_ids,
                                spec["model"],
                                variant,
                                spec["generation_mode"],
                                shard_df,
                                model_config,
                                experiment,
                                classify_batch_size,
                                spec["results_path"],
                                write_lock,
                                stop_event,
                                max_item_retries,
                                gpu_memory_utilization,
                                max_model_len,
                                enforce_eager,
                                save_prompts,
                                save_input_text,
                                max_items,
                            ),
                        )
                        process.start()
                        processes.append(process)

                    for process in processes:
                        process.join()
                except KeyboardInterrupt:
                    stop_event.set()
                    for process in processes:
                        process.join(timeout=30)
                    raise

                post_count = count_successes(spec["results_path"])
                if post_count == pre_count:
                    failure_counts[variant] = failure_counts.get(variant, 0) + 1
                    if failure_counts[variant] >= 2:
                        permanently_failed.add(variant)
                        tracking_entries.append(f"SKIP {variant}: zero progress after 2 attempts")
                        print(f"PERMANENTLY SKIPPING {variant} after repeated zero-progress runs")
                else:
                    failure_counts[variant] = 0
                    tracking_entries.append(
                        f"DONE {variant}: +{post_count - pre_count} successful items "
                        f"({post_count}/{spec['expected_count']})"
                    )
                _update_tracking()

    except KeyboardInterrupt:
        pass

    completed_groups = 0
    for spec in grouped_specs:
        if spec["model_variant"] in permanently_failed:
            continue
        if count_successes(spec["results_path"]) >= spec["expected_count"]:
            completed_groups += 1
    skipped_groups = len(permanently_failed)

    final_footer = (
        f"\n{'=' * 70}\n"
        f"  COMPLETE\n"
        f"  Done     : {completed_groups}/{total_groups} ({skipped_groups} skipped)\n"
        f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Results  : {output_root}/\n"
        f"{'=' * 70}"
    )
    _update_tracking(final_footer)


def compact_results(output_path, model_filter=None):
    output_root = Path(output_path)
    if not output_root.exists():
        print(f"ERROR: Output directory does not exist: {output_root}")
        sys.exit(1)

    target_models = model_filter or PRIMARY_BASE_MODELS
    compacted = 0
    total_records = 0
    for model_name in target_models:
        results_path = get_results_path(output_root, model_name)
        if not results_path.exists():
            continue

        latest = {}
        for record in iter_result_records(results_path):
            item_uid = record.get("item_uid")
            if item_uid is not None:
                latest[int(item_uid)] = record

        tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for _, record in sorted(latest.items()):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(results_path)
        compacted += 1
        total_records += len(latest)
        print(f"Compacted {results_path} -> {len(latest)} records")

    print(f"Compacted {compacted} files with {total_records} deduplicated records")


def main():
    parser = argparse.ArgumentParser(
        description="IBM topic-sentiment MCQ experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate-design", help="Generate an LHS design")
    gen_parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_BLOCK)
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--models", type=str, default=None)
    gen_parser.add_argument("--output", type=str, default=None)

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--gpus", type=str, required=True, help='Comma-separated GPU IDs, e.g. "0,1"')
    run_parser.add_argument("--models", type=str, default=None)
    run_parser.add_argument("--classify-batch-size", type=int, default=vllm_loader.DEFAULT_CLASSIFY_BATCH_SIZE)
    run_parser.add_argument("--dp-threshold", type=float, default=vllm_loader.DP_THRESHOLD)
    run_parser.add_argument("--min-tp-size", type=int, default=vllm_loader.DEFAULT_MIN_TP_SIZE)
    run_parser.add_argument("--long-gen-token-threshold", type=int, default=vllm_loader.DEFAULT_LONG_GEN_TOKEN_THRESHOLD)
    run_parser.add_argument("--long-gen-min-tp-size", type=int, default=vllm_loader.DEFAULT_LONG_GEN_MIN_TP_SIZE)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=vllm_loader.DEFAULT_GPU_MEMORY_UTILIZATION)
    run_parser.add_argument("--max-model-len", type=int, default=vllm_loader.DEFAULT_VLLM_MAX_MODEL_LEN)
    run_parser.add_argument("--enforce-eager", action="store_true")
    run_parser.add_argument("--output", type=str, default=None)
    run_parser.add_argument("--progress-log", type=str, default="_progress_mcq_v1.log")
    run_parser.add_argument("--max-item-retries", type=int, default=DEFAULT_ITEM_RETRIES)
    run_parser.add_argument("--save-prompts", action="store_true")
    run_parser.add_argument("--save-input-text", action="store_true")
    run_parser.add_argument("--max-configs", type=int, help="limit pending configs per model (smoke tests)")
    run_parser.add_argument("--max-items", type=int, help="limit topic items per config (smoke tests)")

    compact_parser = subparsers.add_parser("compact", help="Compact result JSONLs by latest item_uid")
    compact_parser.add_argument("--output", type=str, required=True)
    compact_parser.add_argument("--models", type=str, default=None)

    args = parser.parse_args()

    if args.command == "generate-design":
        model_filter = parse_csv_arg(args.models)
        output_root = resolve_output_root(args.output)
        output_root.mkdir(parents=True, exist_ok=True)
        df = generate_experimental_design(
            samples_per_block=args.samples,
            seed=args.seed,
            model_filter=model_filter,
        )
        design_path = output_root / "experimental_design.csv"
        df.to_csv(design_path, index=False)
        item_count = len(load_experiment_items())
        print(f"Saved design to {design_path}")
        print(f"  Experiment    : {EXPERIMENT}")
        print(f"  Models        : {df['model'].nunique()}")
        print(f"  Ideologies    : {df['ideology'].nunique()}")
        print(f"  Samples/block : {args.samples}")
        print(f"  Configs       : {len(df)}")
        print(f"  Items/config  : {item_count}")
        print(f"  Expected rows : {len(df) * item_count}")
        return

    if args.command == "run":
        gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        model_filter = parse_csv_arg(args.models)
        run_experiments(
            gpu_ids,
            EXPERIMENT,
            model_filter=model_filter,
            dp_threshold=args.dp_threshold,
            min_tp_size=args.min_tp_size,
            long_gen_token_threshold=args.long_gen_token_threshold,
            long_gen_min_tp_size=args.long_gen_min_tp_size,
            classify_batch_size=args.classify_batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            output_path=args.output,
            progress_log_name=args.progress_log,
            max_item_retries=args.max_item_retries,
            save_prompts=args.save_prompts,
            save_input_text=args.save_input_text,
            max_configs=args.max_configs,
            max_items=args.max_items,
        )
        return

    if args.command == "compact":
        model_filter = parse_csv_arg(args.models)
        compact_results(args.output, model_filter=model_filter)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
