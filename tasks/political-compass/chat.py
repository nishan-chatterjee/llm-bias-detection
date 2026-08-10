#!/usr/bin/env python3
"""
Political Compass chat experiment orchestrator for the PoliLean release.

This script owns:
  - Language-aware LHS design generation for chat experiments
  - Expansion of Qwen think/no_think into separate model variants
  - Per-question JSONL checkpointing with resume support
  - Scheduling model-variant workers across explicit GPU IDs

The heavy lifting of loading vLLM models and running chat / classification is
delegated to `model_loader.py`.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from datetime import datetime

import pandas as pd
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_loader as vllm_loader

PROJECT_ROOT = SCRIPT_DIR
MODEL_DIR = PROJECT_ROOT.parent.parent / "models" / "serve"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "chat"
CHAT_PROMPTS_DIR = PROJECT_ROOT / "data" / "chat-prompts"
LEGACY_PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"

DEFAULT_LANGUAGE = "english"
CHAT_SAMPLES_PER_BLOCK = 300
MAX_RUN_TIME_SECONDS = 3600 * 95
DEFAULT_ITEM_RETRIES = 1
DEFAULT_REASONING_MODE = "neutral"
REASONING_MODES = ["short", "neutral", "think"]
CHAT_INSTRUCTION_VARIANTS = 5
DEFAULT_CHAT_SUFFIXES = [
    "\nFinal answer key:",
    "\nAnswer using one key only:",
    "\nChosen option key:",
    "\nReturn only the matching key:",
    "\nSelected answer token:",
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


def _jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def get_available_languages():
    prompt_languages = {path.stem for path in CHAT_PROMPTS_DIR.glob("*.json")}
    prompt_languages |= {path.stem for path in LEGACY_PROMPTS_DIR.glob("*.json")}
    question_languages = {path.stem for path in (PROJECT_ROOT / "data" / "questions").glob("*.json")}
    return sorted(prompt_languages & question_languages)


def resolve_prompts_path(language):
    preferred = CHAT_PROMPTS_DIR / f"{language}.json"
    if preferred.exists():
        return preferred
    fallback = LEGACY_PROMPTS_DIR / f"{language}.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No prompt file found for language='{language}' in {CHAT_PROMPTS_DIR} or {LEGACY_PROMPTS_DIR}")


def parse_languages(language_arg):
    if not language_arg:
        return [DEFAULT_LANGUAGE]

    values = [value.strip().lower() for value in language_arg.split(",") if value.strip()]
    if not values:
        return [DEFAULT_LANGUAGE]
    if "all" in values:
        return get_available_languages()

    available = set(get_available_languages())
    invalid = [value for value in values if value not in available]
    if invalid:
        raise ValueError(
            f"Unsupported language(s): {', '.join(invalid)}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return values


def load_prompts_and_questions(language):
    prompts_path = resolve_prompts_path(language)
    questions_path = PROJECT_ROOT / "data" / "questions" / f"{language}.json"

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_list = json.load(f)

    return prompts_dict, questions_list


def load_selected_language_resources(languages):
    return {
        language: {
            "prompts": prompts_dict,
            "questions": questions_list,
        }
        for language, (prompts_dict, questions_list) in (
            (language, load_prompts_and_questions(language)) for language in languages
        )
    }


def expand_model_variants(model_filter=None):
    model_config = vllm_loader.load_model_config(MODEL_DIR)
    requested = set(model_filter or [])
    variants = []

    if requested:
        requested_bases = {
            name.removesuffix("_no_think").removesuffix("_think") for name in requested
        }
        unknown = sorted(requested_bases - set(vllm_loader.SUPPORTED_MODELS))
        if unknown:
            raise ValueError("Unsupported model aliases: " + ", ".join(unknown))
        model_names = [name for name in vllm_loader.SUPPORTED_MODELS if name in requested_bases]
    else:
        model_names = list(vllm_loader.PRIMARY_MODELS)

    for model_name in model_names:
        model_cfg = model_config.get(model_name, {})
        if model_cfg.get("thinking", False):
            candidate_variants = [
                (model_name, f"{model_name}_think", "think"),
                (model_name, f"{model_name}_no_think", "no_think"),
            ]
        else:
            candidate_variants = [(model_name, model_name, "standard")]

        for base_model, model_variant, generation_mode in candidate_variants:
            if requested and base_model not in requested and model_variant not in requested:
                continue
            variants.append(
                {
                    "model": base_model,
                    "model_variant": model_variant,
                    "generation_mode": generation_mode,
                }
            )

    return variants


def map_factors_chat(sample, language, n_suffixes, n_instr_variants):
    factors = {}
    factors["language"] = language
    factors["suffix_idx"] = min(int(sample[0] * n_suffixes), n_suffixes - 1)

    if sample[1] < 0.5:
        factors["context_id"] = None
    else:
        ctx_idx = int((sample[1] - 0.5) * 2 * 5)
        factors["context_id"] = min(ctx_idx, 4)

    factors["instr_type"] = "question_first" if sample[2] < 0.5 else "options_first"
    factors["reasoning_mode"] = REASONING_MODES[min(int(sample[3] * len(REASONING_MODES)), len(REASONING_MODES) - 1)]
    factors["instr_idx"] = min(int(sample[4] * n_instr_variants), n_instr_variants - 1)
    factors["persona_class"] = "short" if sample[5] < 0.5 else "long"
    factors["persona_idx"] = int(sample[6] * 5)
    factors["key_type"] = KEY_TYPES[int(sample[7] * len(KEY_TYPES))]
    factors["perm_id"] = int(sample[8] * 4)
    return factors


def _get_chat_suffixes(prompts_dict):
    suffixes = prompts_dict.get("experiment_settings", {}).get("chat_suffixes")
    if isinstance(suffixes, list) and suffixes:
        return suffixes
    return list(DEFAULT_CHAT_SUFFIXES)


def _infer_instruction_variant_count(prompts_dict, answer_mode):
    pool = _get_instruction_pool(prompts_dict, answer_mode)
    counts = []
    for values in pool.values():
        if isinstance(values, list):
            counts.append(len(values))
        elif isinstance(values, dict):
            for mode_values in values.values():
                if isinstance(mode_values, list):
                    counts.append(len(mode_values))
    if not counts:
        return CHAT_INSTRUCTION_VARIANTS
    return max(1, max(counts))


def generate_experimental_design(
    samples_per_block=CHAT_SAMPLES_PER_BLOCK,
    seed=42,
    model_filter=None,
    languages=None,
):
    selected_languages = languages or [DEFAULT_LANGUAGE]
    language_resources = load_selected_language_resources(selected_languages)
    model_variants = expand_model_variants(model_filter)

    try:
        sampler = qmc.LatinHypercube(d=9, rng=seed)  # type: ignore[call-arg]
    except TypeError:
        sampler = qmc.LatinHypercube(d=9, seed=seed)  # type: ignore[call-arg]
    base_block_samples = sampler.random(n=samples_per_block)

    records = []
    for variant in model_variants:
        for language in selected_languages:
            prompts_dict = language_resources[language]["prompts"]
            n_suffixes = len(_get_chat_suffixes(prompts_dict))
            n_instr_variants = _infer_instruction_variant_count(prompts_dict, answer_mode="chat-classify")
            for ideology in IDEOLOGIES:
                for sample in base_block_samples:
                    config = map_factors_chat(sample, language, n_suffixes, n_instr_variants)
                    config.update(variant)
                    config["ideology"] = ideology
                    if ideology == "base":
                        config["persona_class"] = None
                        config["persona_idx"] = None
                    records.append(config)

    df = pd.DataFrame(records)
    df["config_id"] = range(len(df))
    return df


def _get_instruction_pool(prompts_dict, answer_mode):
    if answer_mode in {"chat", "chat-classify"} and "chat_instructions" in prompts_dict:
        return prompts_dict["chat_instructions"]
    return prompts_dict["instructions"]


def _get_instruction_entries(instruction_pool, instr_type, reasoning_mode):
    bucket = instruction_pool[instr_type]
    if isinstance(bucket, list):
        return bucket
    if isinstance(bucket, dict):
        if reasoning_mode in bucket and isinstance(bucket[reasoning_mode], list):
            return bucket[reasoning_mode]
        if DEFAULT_REASONING_MODE in bucket and isinstance(bucket[DEFAULT_REASONING_MODE], list):
            return bucket[DEFAULT_REASONING_MODE]
        for mode in REASONING_MODES:
            values = bucket.get(mode)
            if isinstance(values, list) and values:
                return values
    raise KeyError(f"Unsupported instruction schema for instr_type='{instr_type}'")


def assemble_prompt_for_question(config_row, question_obj, prompts_dict, answer_mode):
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
            persona_text = template["text"].format(**insertions)
            segments.append(persona_text)
        except (KeyError, IndexError):
            pass

    p_id = int(config_row["perm_id"])
    rev_labels = p_id in [2, 3]
    rev_keys = p_id in [1, 3]

    instr_type = config_row["instr_type"]
    instr_idx = int(config_row["instr_idx"])
    reasoning_mode = str(config_row.get("reasoning_mode", DEFAULT_REASONING_MODE))
    instruction_pool = _get_instruction_pool(prompts_dict, answer_mode)
    instr_entries = _get_instruction_entries(instruction_pool, instr_type, reasoning_mode)
    instr_template = instr_entries[instr_idx % len(instr_entries)]["text"]

    labels = list(question_obj["choices"])
    keys = list(prompts_dict["experiment_settings"]["answer_keys"][config_row["key_type"]])

    if rev_labels:
        labels = labels[::-1]
    if rev_keys:
        keys = keys[::-1]

    options_formatted = "\n" + "\n".join(f"{key}. {label}" for key, label in zip(keys, labels))
    final_instr = instr_template.format(
        question=question_obj["statement"],
        options_formatted=options_formatted,
    )
    segments.append(final_instr)

    prompt_text = "\n\n".join(segments)
    candidates = [
        {"index": idx, "key": key, "text": label, "score_text": f" {key}"}
        for idx, (key, label) in enumerate(zip(keys, labels))
    ]
    return prompt_text, candidates


def build_items_for_config(config_row, prompts_dict, questions_list, answer_mode):
    suffixes = _get_chat_suffixes(prompts_dict)
    suffix_idx = int(config_row["suffix_idx"])
    suffix_text = suffixes[suffix_idx % len(suffixes)]

    items = []
    for question_obj in questions_list:
        prompt_text, candidates = assemble_prompt_for_question(
            config_row,
            question_obj,
            prompts_dict,
            answer_mode,
        )
        item_id = f"cfg{int(config_row['config_id'])}_q{int(question_obj['id'])}"
        items.append(
            {
                "item_id": item_id,
                "input_index": int(question_obj["id"]),
                "prompt": prompt_text,
                "classification_suffix": suffix_text,
                "candidates": candidates,
                "metadata": {
                    "config_id": int(config_row["config_id"]),
                    "question_id": int(question_obj["id"]),
                    "question_page": _jsonable(question_obj.get("page")),
                    "statement": question_obj.get("statement"),
                    "premise": question_obj.get("premise"),
                    "original_choices": list(question_obj.get("choices", [])),
                },
            }
        )
    return items


def resolve_output_root(output_path=None):
    return Path(output_path) if output_path else DEFAULT_OUTPUT_ROOT


def get_results_path(output_root, model_variant):
    return output_root / f"{model_variant}.jsonl"


def load_item_statuses(results_path):
    statuses = {}
    if not results_path.exists():
        return statuses

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = record.get("item_id")
            if item_id:
                statuses[item_id] = {"error": record.get("error")}
    return statuses


def load_completed_success_ids(results_path):
    statuses = load_item_statuses(results_path)
    return {item_id for item_id, status in statuses.items() if not status.get("error")}


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


def compact_results_file(results_path):
    if not results_path.exists():
        return 0

    latest_records = {}
    for record in iter_result_records(results_path):
        item_id = record.get("item_id")
        if item_id:
            latest_records[item_id] = record

    ordered_records = sorted(
        latest_records.values(),
        key=lambda record: (
            str(record.get("language", "")),
            int(record.get("config_id", -1) if record.get("config_id") is not None else -1),
            int(record.get("question_id", -1) if record.get("question_id") is not None else -1),
            str(record.get("item_id", "")),
        ),
    )

    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in ordered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(results_path)
    return len(ordered_records)


def append_records_to_jsonl(results_path, config_row, records, answer_mode):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    base_meta = {
        "config_id": int(config_row["config_id"]),
        "model": config_row["model"],
        "model_variant": config_row["model_variant"],
        "generation_mode": config_row["generation_mode"],
        "language": config_row["language"],
        "ideology": config_row["ideology"],
        "context_id": int(config_row["context_id"]) if pd.notna(config_row.get("context_id")) else -1,
        "instr_type": config_row["instr_type"],
        "reasoning_mode": str(config_row.get("reasoning_mode", DEFAULT_REASONING_MODE)),
        "instr_idx": int(config_row["instr_idx"]),
        "persona_class": config_row["persona_class"] if pd.notna(config_row.get("persona_class")) else "",
        "persona_idx": int(config_row["persona_idx"]) if pd.notna(config_row.get("persona_idx")) else -1,
        "key_type": config_row["key_type"],
        "perm_id": int(config_row["perm_id"]),
        "suffix_idx": int(config_row["suffix_idx"]),
        "answer_mode": answer_mode,
        "inference_backend": "vllm",
        "quantization": "model-native",
        "written_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(results_path, "a", encoding="utf-8") as f:
        for record in records:
            metadata = dict(record.get("metadata", {}))
            output_record = dict(base_meta)
            output_record.update(
                {
                    "item_id": record["item_id"],
                    "question_id": metadata.get("question_id"),
                    "question_page": metadata.get("question_page"),
                    "statement": metadata.get("statement"),
                    "premise": metadata.get("premise"),
                    "prompt": record.get("prompt"),
                    "classification_suffix": record.get("classification_suffix"),
                    "candidate_keys": record.get("candidate_keys"),
                    "candidate_texts": record.get("candidate_texts"),
                    "stage1_text": record.get("stage1_text"),
                    "stage1_tokens": record.get("stage1_tokens"),
                    "stage1_finish_reason": record.get("stage1_finish_reason"),
                    "classification_logprobs": record.get("classification_logprobs"),
                    "classification_probs": record.get("classification_probs"),
                    "classification_pred_key": record.get("classification_pred_key"),
                    "item_metadata": metadata,
                    "attempt": record.get("attempt", 1),
                    "error": record.get("error"),
                }
            )
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def build_expected_item_ids(task_df, language_resources):
    expected = set()
    for _, config_row in task_df.iterrows():
        question_ids = [
            int(question["id"])
            for question in language_resources[str(config_row["language"])]["questions"]
        ]
        expected.update(
            f"cfg{int(config_row['config_id'])}_q{int(question_id)}"
            for question_id in question_ids
        )
    return expected


def count_successes_for_group(results_path, expected_item_ids):
    statuses = load_item_statuses(results_path)
    return sum(
        1
        for item_id in expected_item_ids
        if item_id in statuses and not statuses[item_id].get("error")
    )


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
    answer_mode,
    batch_size,
    classify_batch_size,
    results_path,
    write_lock,
    stop_event,
    max_item_retries,
    gpu_memory_utilization,
    max_model_len,
    enforce_eager,
):
    print(
        f"[{datetime.now()}] START {model_variant} | GPUs {gpu_ids} | "
        f"configs={len(task_df)} | answer={answer_mode}"
    )

    def _make_runner():
        model_source, _revision = vllm_loader.resolve_model_source(
            model_name, MODEL_DIR, model_config
        )
        return vllm_loader.VLLMRunner(
            model_source,
            model_name,
            generation_mode,
            gpu_ids,
            model_config,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )

    def _reset_runner(current_runner):
        if current_runner is not None:
            current_runner.close()
        return _make_runner()

    runner = None
    try:
        runner = _make_runner()
        language_resources = load_selected_language_resources(task_df["language"].dropna().unique().tolist())
        completed_success_ids = load_completed_success_ids(results_path)

        for _, config_row in task_df.sort_values("config_id").iterrows():
            if stop_event.is_set():
                break

            language = config_row["language"]
            prompts_dict = language_resources[language]["prompts"]
            questions_list = language_resources[language]["questions"]
            items = build_items_for_config(config_row, prompts_dict, questions_list, answer_mode)
            item_lookup = {item["item_id"]: item for item in items}
            pending_items = [
                item for item in items if item["item_id"] not in completed_success_ids
            ]
            if not pending_items:
                continue

            for attempt_idx in range(1, max_item_retries + 2):
                try:
                    records = runner.run_items(
                        pending_items,
                        answer_mode=answer_mode,
                        batch_size=batch_size,
                        classify_batch_size=classify_batch_size,
                    )
                except Exception as exc:
                    print(
                        f"[{datetime.now()}] ERROR {model_variant} | config={int(config_row['config_id'])} "
                        f"| attempt={attempt_idx}: {exc}"
                    )
                    records = []
                    for item in pending_items:
                        records.append(
                            {
                                "item_id": item["item_id"],
                                "input_index": item["input_index"],
                                "prompt": item["prompt"],
                                "answer_mode": answer_mode,
                                "classification_suffix": item.get("classification_suffix", ""),
                                "candidate_keys": [candidate["key"] for candidate in item.get("candidates", [])],
                                "candidate_texts": [candidate.get("text", "") for candidate in item.get("candidates", [])],
                                "metadata": dict(item.get("metadata", {})),
                                "stage1_text": None,
                                "stage1_tokens": 0,
                                "stage1_finish_reason": None,
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
                        append_records_to_jsonl(results_path, config_row, success_records, answer_mode)
                    for record in success_records:
                        completed_success_ids.add(record["item_id"])

                if not failed_records:
                    break

                should_reset_runner = any(
                    "EngineCore encountered an issue" in str(record.get("error", ""))
                    or "worker_config_failed" in str(record.get("error", ""))
                    for record in failed_records
                )
                if attempt_idx <= max_item_retries:
                    if should_reset_runner:
                        print(
                            f"[{datetime.now()}] RESET {model_variant} | config={int(config_row['config_id'])} "
                            f"| reason=engine failure before attempt={attempt_idx + 1}"
                        )
                        runner = _reset_runner(runner)
                    print(
                        f"[{datetime.now()}] RETRY {model_variant} | config={int(config_row['config_id'])} "
                        f"| remaining_failed={len(failed_records)} | next_attempt={attempt_idx + 1}"
                    )
                    pending_items = [item_lookup[record["item_id"]] for record in failed_records]
                else:
                    with write_lock:
                        append_records_to_jsonl(results_path, config_row, failed_records, answer_mode)

    finally:
        if runner is not None:
            runner.close()
        print(f"[{datetime.now()}] DONE {model_variant}")


def run_experiments(
    available_gpus,
    answer_mode="chat-classify",
    model_filter=None,
    languages=None,
    dp_threshold=vllm_loader.DP_THRESHOLD,
    min_tp_size=vllm_loader.DEFAULT_MIN_TP_SIZE,
    long_gen_token_threshold=vllm_loader.DEFAULT_LONG_GEN_TOKEN_THRESHOLD,
    long_gen_min_tp_size=vllm_loader.DEFAULT_LONG_GEN_MIN_TP_SIZE,
    batch_size=vllm_loader.DEFAULT_CHAT_BATCH_SIZE,
    classify_batch_size=vllm_loader.DEFAULT_CLASSIFY_BATCH_SIZE,
    gpu_memory_utilization=vllm_loader.DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=vllm_loader.DEFAULT_VLLM_MAX_MODEL_LEN,
    enforce_eager=False,
    output_path=None,
    progress_log_name="_progress.log",
    max_item_retries=DEFAULT_ITEM_RETRIES,
):
    if not 0.0 < dp_threshold <= 1.0:
        raise ValueError(f"dp_threshold must be in (0, 1], got {dp_threshold}")
    if min_tp_size < 1:
        raise ValueError(f"min_tp_size must be >= 1, got {min_tp_size}")
    if long_gen_token_threshold < 1:
        raise ValueError(
            f"long_gen_token_threshold must be >= 1, got {long_gen_token_threshold}"
        )
    if long_gen_min_tp_size < 1:
        raise ValueError(f"long_gen_min_tp_size must be >= 1, got {long_gen_min_tp_size}")

    start_time = time.time()
    mp.set_start_method("spawn", force=True)

    selected_languages = languages or [DEFAULT_LANGUAGE]
    language_resources = load_selected_language_resources(selected_languages)
    model_config = vllm_loader.load_model_config(MODEL_DIR)

    output_root = resolve_output_root(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / progress_log_name

    design_path = output_root / "experimental_design.csv"
    if not design_path.exists():
        print(f"ERROR: No experimental design found at {design_path}")
        print("Run: python tasks/political-compass/chat.py generate-design")
        sys.exit(1)

    df_schedule = pd.read_csv(design_path)
    df_schedule["language"] = df_schedule["language"].astype(str)
    if selected_languages:
        requested_languages = set(selected_languages)
        df_schedule = df_schedule[df_schedule["language"].isin(requested_languages)]
        if df_schedule.empty:
            print("ERROR: No matching languages found in the schedule.")
            sys.exit(1)
    if model_filter:
        requested = set(model_filter)
        df_schedule = df_schedule[
            df_schedule["model"].isin(requested) | df_schedule["model_variant"].isin(requested)
        ]
        if df_schedule.empty:
            print("ERROR: No matching models/model variants found in the schedule.")
            sys.exit(1)

        # Respect the user-provided model order when filtering subsets.
        order_map = {name: idx for idx, name in enumerate(model_filter)}
        fallback_rank = len(order_map)
        model_rank = df_schedule["model"].map(order_map)
        variant_rank = df_schedule["model_variant"].map(order_map)
        df_schedule["_model_order"] = model_rank.fillna(variant_rank).fillna(fallback_rank).astype(int)
        df_schedule = (
            df_schedule
            .sort_values(["_model_order", "config_id"], kind="stable")
            .drop(columns=["_model_order"])
        )

    detected_gpus, vram_per_gpu, gpu_name = vllm_loader.detect_gpus()
    if detected_gpus == 0:
        print("ERROR: No GPUs detected.")
        sys.exit(1)

    max_visible_gpu = max(available_gpus) if available_gpus else -1
    if max_visible_gpu >= detected_gpus:
        print(
            f"ERROR: Requested GPU id {max_visible_gpu} but only {detected_gpus} GPUs are visible."
        )
        sys.exit(1)

    grouped_specs = []
    for model_variant, task_df in df_schedule.groupby("model_variant", sort=False):
        model_name = task_df["model"].iloc[0]
        generation_mode = task_df["generation_mode"].iloc[0]
        results_path = get_results_path(output_root, model_variant)
        expected_item_ids = build_expected_item_ids(task_df, language_resources)

        model_cfg = model_config.get(model_name, {})
        params_b = model_cfg.get("params_B")
        effective_min_tp = vllm_loader.compute_effective_min_tp_size(
            model_cfg,
            generation_mode,
            base_min_tp_size=min_tp_size,
            long_gen_token_threshold=long_gen_token_threshold,
            long_gen_min_tp_size=long_gen_min_tp_size,
        )
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
                "results_path": results_path,
                "expected_item_ids": expected_item_ids,
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
        f"  Experiment : {output_root}\n"
        f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  GPU        : {len(available_gpus)}× {gpu_name} ({vram_per_gpu:.0f} GB each)\n"
        f"  Language   : {', '.join(selected_languages)}\n"
        f"  Groups     : {total_groups}\n"
        f"  Answer     : {answer_mode}\n"
        f"  Schedule   : dp<{dp_threshold:.2f}, min_tp={min_tp_size}, long_gen>={long_gen_token_threshold}->min_tp={long_gen_min_tp_size}\n"
        f"  vLLM mem   : util={gpu_memory_utilization:.2f}, max_len={max_model_len or 'model default'}, eager={enforce_eager}\n"
        f"  Retries    : {max_item_retries}\n"
        f"{'=' * 70}\n"
    )

    def _update_tracking(footer_extra=""):
        elapsed = time.time() - start_time
        done = completed_groups + skipped_groups
        remaining = total_groups - done
        eta = f"~{(elapsed / done) * remaining:.0f}s" if done else "calculating..."
        footer = (
            f"\n{'─' * 70}\n"
            f"  Progress : {done}/{total_groups} "
            f"({completed_groups} done, {skipped_groups} skipped, {remaining} remaining)\n"
            f"  Elapsed  : {elapsed:.1f}s\n"
            f"  ETA      : {eta}\n"
            f"  Updated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'─' * 70}"
            + (f"\n{footer_extra}" if footer_extra else "")
        )
        vllm_loader.write_tracking_log(progress_path, header, tracking_entries, footer)

    _update_tracking()

    # Emit resume status so progress logs show what is already complete/partial.
    resume_done = 0
    resume_partial = 0
    resume_lines = []
    for spec in grouped_specs:
        success_count = count_successes_for_group(
            spec["results_path"],
            spec["expected_item_ids"],
        )
        expected_count = len(spec["expected_item_ids"])
        if success_count >= expected_count:
            resume_done += 1
            resume_lines.append(
                f"ALREADY_DONE {spec['model_variant']}: {success_count}/{expected_count}"
            )
        elif success_count > 0:
            resume_partial += 1
            resume_lines.append(
                f"ALREADY_PARTIAL {spec['model_variant']}: {success_count}/{expected_count}"
            )

    if resume_lines:
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
                success_count = count_successes_for_group(
                    spec["results_path"],
                    spec["expected_item_ids"],
                )
                if success_count < len(spec["expected_item_ids"]):
                    remaining_specs.append(spec)

            completed_groups = total_groups - len(remaining_specs) - len(permanently_failed)
            skipped_groups = len(permanently_failed)

            if not remaining_specs:
                break

            print(
                f"Pass: {len(remaining_specs)} model variants remaining "
                f"({len(permanently_failed)} permanently failed)"
            )
            for spec in remaining_specs:
                if time.time() - start_time > MAX_RUN_TIME_SECONDS:
                    break

                variant = spec["model_variant"]
                pre_count = count_successes_for_group(
                    spec["results_path"],
                    spec["expected_item_ids"],
                )

                schedule_tuple = spec["schedule"]
                if schedule_tuple is None:
                    permanently_failed.add(variant)
                    tracking_entries.append(
                        f"SKIP {variant}: model too large for available GPUs"
                    )
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
                    f"START {variant}: strategy={strategy}, tp={tp_size}, instances={len(task_shards)}"
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
                                answer_mode,
                                batch_size,
                                classify_batch_size,
                                spec["results_path"],
                                write_lock,
                                stop_event,
                                max_item_retries,
                                gpu_memory_utilization,
                                max_model_len,
                                enforce_eager,
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

                post_count = count_successes_for_group(
                    spec["results_path"],
                    spec["expected_item_ids"],
                )
                if post_count == pre_count:
                    failure_counts[variant] = failure_counts.get(variant, 0) + 1
                    if failure_counts[variant] >= 2:
                        permanently_failed.add(variant)
                        tracking_entries.append(
                            f"SKIP {variant}: zero progress after 2 attempts"
                        )
                        print(f"PERMANENTLY SKIPPING {variant} after repeated zero-progress runs")
                else:
                    failure_counts[variant] = 0
                    tracking_entries.append(
                        f"DONE {variant}: +{post_count - pre_count} successful items"
                    )
                _update_tracking()

    except KeyboardInterrupt:
        pass

    completed_groups = 0
    for spec in grouped_specs:
        if spec["model_variant"] in permanently_failed:
            continue
        success_count = count_successes_for_group(spec["results_path"], spec["expected_item_ids"])
        if success_count >= len(spec["expected_item_ids"]):
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


def compact_results(output_path=None, model_filter=None):
    output_root = resolve_output_root(output_path)
    if not output_root.exists():
        print(f"ERROR: Output directory does not exist: {output_root}")
        sys.exit(1)

    if model_filter:
        target_variants = [entry["model_variant"] for entry in expand_model_variants(model_filter)]
        target_paths = [get_results_path(output_root, model_variant) for model_variant in target_variants]
    else:
        target_paths = sorted(output_root.glob("*.jsonl"))

    compacted = 0
    total_records = 0
    for results_path in target_paths:
        if not results_path.exists():
            continue
        record_count = compact_results_file(results_path)
        compacted += 1
        total_records += record_count
        print(f"Compacted {results_path} -> {record_count} records")

    print(f"Compacted {compacted} files with {total_records} deduplicated records")


def main():
    parser = argparse.ArgumentParser(
        description="Political Compass chat experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate-design", help="Generate the chat LHS design")
    gen_parser.add_argument(
        "--samples",
        type=int,
        default=CHAT_SAMPLES_PER_BLOCK,
        help="Samples per ideology/model-variant block",
    )
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for LHS")
    gen_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated subset of base models or model variants",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory that will receive experimental_design.csv",
    )
    gen_parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help='Comma-separated languages or "all". Default: english',
    )

    run_parser = subparsers.add_parser("run", help="Run chat experiments")
    run_parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='Comma-separated GPU IDs, e.g. "0,1,2,3"',
    )
    run_parser.add_argument(
        "--answer",
        type=str,
        choices=["chat", "chat-classify"],
        default="chat-classify",
        help="Whether to save only generation or generation plus candidate classification.",
    )
    run_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated subset of base models or model variants",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=vllm_loader.DEFAULT_CHAT_BATCH_SIZE,
        help="Stage-1 generation batch size inside each worker.",
    )
    run_parser.add_argument(
        "--classify-batch-size",
        type=int,
        default=vllm_loader.DEFAULT_CLASSIFY_BATCH_SIZE,
        help="Classification batch size inside each worker.",
    )
    run_parser.add_argument(
        "--dp-threshold",
        type=float,
        default=vllm_loader.DP_THRESHOLD,
        help="Weight-to-VRAM fraction target under which DP is allowed.",
    )
    run_parser.add_argument(
        "--min-tp-size",
        type=int,
        default=vllm_loader.DEFAULT_MIN_TP_SIZE,
        help="Global minimum tensor parallel size for all model variants.",
    )
    run_parser.add_argument(
        "--long-gen-token-threshold",
        type=int,
        default=vllm_loader.DEFAULT_LONG_GEN_TOKEN_THRESHOLD,
        help="If max_new_tokens >= this value, enforce long-generation TP policy.",
    )
    run_parser.add_argument(
        "--long-gen-min-tp-size",
        type=int,
        default=vllm_loader.DEFAULT_LONG_GEN_MIN_TP_SIZE,
        help="Minimum tensor parallel size for long-generation model variants.",
    )
    run_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=vllm_loader.DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Target fraction of VRAM reserved by vLLM per worker.",
    )
    run_parser.add_argument(
        "--max-model-len",
        type=int,
        default=vllm_loader.DEFAULT_VLLM_MAX_MODEL_LEN,
        help="Optional max model context length override for vLLM.",
    )
    run_parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run vLLM in eager mode instead of CUDA graph mode.",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory containing experimental_design.csv and result JSONLs.",
    )
    run_parser.add_argument(
        "--progress-log",
        type=str,
        default="_progress.log",
        help="Progress log filename written inside the output directory.",
    )
    run_parser.add_argument(
        "--max-item-retries",
        type=int,
        default=DEFAULT_ITEM_RETRIES,
        help="Number of additional same-session retries for failed items.",
    )
    run_parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help='Comma-separated languages or "all". Default: english',
    )

    compact_parser = subparsers.add_parser("compact", help="Compact result JSONLs by keeping the latest record per item_id")
    compact_parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory containing result JSONLs.",
    )
    compact_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated subset of base models or model variants",
    )

    args = parser.parse_args()

    if args.command == "generate-design":
        model_filter = [value.strip() for value in args.models.split(",")] if args.models else None
        languages = parse_languages(args.language)
        output_root = resolve_output_root(args.output)
        output_root.mkdir(parents=True, exist_ok=True)
        design_path = output_root / "experimental_design.csv"
        df = generate_experimental_design(
            samples_per_block=args.samples,
            seed=args.seed,
            model_filter=model_filter,
            languages=languages,
        )
        df.to_csv(design_path, index=False)
        print(f"Saved design to {design_path}")
        print(f"  Model variants: {df['model_variant'].nunique()}")
        print(f"  Languages     : {df['language'].nunique()} ({', '.join(languages)})")
        print(f"  Ideologies    : {df['ideology'].nunique()}")
        print(f"  Samples/block : {args.samples}")
        print(f"  Total configs : {len(df)}")
        return

    if args.command == "run":
        gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        model_filter = [value.strip() for value in args.models.split(",")] if args.models else None
        languages = parse_languages(args.language)
        run_experiments(
            gpu_ids,
            answer_mode=args.answer,
            model_filter=model_filter,
            languages=languages,
            dp_threshold=args.dp_threshold,
            min_tp_size=args.min_tp_size,
            long_gen_token_threshold=args.long_gen_token_threshold,
            long_gen_min_tp_size=args.long_gen_min_tp_size,
            batch_size=args.batch_size,
            classify_batch_size=args.classify_batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            output_path=args.output,
            progress_log_name=args.progress_log,
            max_item_retries=args.max_item_retries,
        )
        return

    if args.command == "compact":
        model_filter = [value.strip() for value in args.models.split(",")] if args.models else None
        compact_results(output_path=args.output, model_filter=model_filter)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
