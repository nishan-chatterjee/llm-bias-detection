#!/usr/bin/env python3
"""
Reusable vLLM runner for the PoliLean release experiments.

Supports three answer modes:
  - chat: free-form generation only
  - chat-classify: generate first, then score candidate keys with logprobs
  - mcq: score candidate keys directly from the prompt without generation

The input protocol is normalized to "items":
  {
    "item_id": "q17",
    "prompt": "full prompt text",
    "classification_suffix": "In summary, I select: ",
    "candidates": [
      {"index": 0, "key": "A", "text": "option text"},
      {"index": 1, "key": "B", "text": "option text"}
    ],
    "metadata": {...}
  }

For debug / ad-hoc usage the loader also accepts:
  - plain prompt lists
  - the older structured JSON format used in earlier local tests
"""

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import queue
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

# We only import transformers / vllm inside child processes so the main process
# stays clean of CUDA contexts.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "serve"

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
SECONDARY_MODELS = ["gemma-3-27b-it-abliterated-normpreserve-v1"]
SUPPORTED_MODELS = PRIMARY_MODELS + SECONDARY_MODELS
# Backwards-compatible name used by chat.py. Defaults intentionally mean only
# the eight primary checkpoints.
MODELS = PRIMARY_MODELS

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Write a short Python function to reverse a string.",
    "Explain deep learning in two sentences.",
]

DEFAULT_CHAT_BATCH_SIZE = 32
DEFAULT_CLASSIFY_BATCH_SIZE = 128
DEFAULT_DEADLINE_SECONDS = 7200
DP_THRESHOLD = 0.60


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_optional_int(name):
    value = os.getenv(name)
    if not value:
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except ValueError:
        return None


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_GPU_MEMORY_UTILIZATION = _env_float("VV_GPU_MEMORY_UTILIZATION", 0.82)
DEFAULT_VLLM_MAX_MODEL_LEN = _env_optional_int("VV_MAX_MODEL_LEN")
DEFAULT_MIN_TP_SIZE = max(1, _env_int("VV_MIN_TP_SIZE", 1))
DEFAULT_LONG_GEN_TOKEN_THRESHOLD = max(1, _env_int("VV_LONG_GEN_TOKEN_THRESHOLD", 4096))
DEFAULT_LONG_GEN_MIN_TP_SIZE = max(1, _env_int("VV_LONG_GEN_MIN_TP_SIZE", 2))


def _tp_candidates(num_gpus):
    candidates = [1]
    tp_size = 2
    while tp_size <= num_gpus:
        candidates.append(tp_size)
        tp_size *= 2
    return candidates


def load_model_config(models_dir=DEFAULT_MODELS_DIR):
    config_path = Path(models_dir) / "model_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_source(model_name, models_dir=DEFAULT_MODELS_DIR, config_data=None):
    """Return ``(source, revision)`` for a local checkpoint or pinned Hub ID."""
    models_dir = Path(models_dir)
    local = models_dir / model_name
    if (local / "config.json").exists():
        return str(local), None

    config_data = config_data if config_data is not None else load_model_config(models_dir)
    model_cfg = config_data.get(model_name, {})
    repo_id = model_cfg.get("repo_id")
    if not repo_id:
        raise FileNotFoundError(
            f"No populated local checkpoint at {local} and no repo_id configured for {model_name}"
        )
    return str(repo_id), model_cfg.get("revision")


def compute_schedule(params_b, vram_per_gpu, num_gpus, dp_threshold=DP_THRESHOLD, min_tp_size=1):
    """
    Decide TP size and number of parallel instances for a model.

    Returns (tp_size, num_instances, strategy_name) or None if the model
    cannot fit.
    """
    weight_vram = params_b * 2
    min_tp_size = max(1, min(int(min_tp_size), int(num_gpus)))
    candidates = [tp for tp in _tp_candidates(num_gpus) if tp >= min_tp_size]
    if not candidates:
        return None

    # Soft fit target for headroom.
    for tp_size in candidates:
        if (weight_vram / tp_size) < dp_threshold * vram_per_gpu:
            num_instances = max(1, num_gpus // tp_size)
            strategy = "dp" if tp_size == 1 else "tp"
            if tp_size > 1 and weight_vram < dp_threshold * vram_per_gpu:
                strategy = "tp-headroom"
            return tp_size, num_instances, strategy

    # Hard fit fallback.
    for tp_size in candidates:
        if (weight_vram / tp_size) < vram_per_gpu * 0.90:
            num_instances = max(1, num_gpus // tp_size)
            return tp_size, num_instances, "tp"

    return None


def get_mode_max_new_tokens(model_cfg, mode):
    generation_cfg = _get_chat_generation_config(model_cfg or {}, mode)
    value = generation_cfg.get("max_new_tokens", 4096)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 4096


def compute_effective_min_tp_size(
    model_cfg,
    mode,
    base_min_tp_size=DEFAULT_MIN_TP_SIZE,
    long_gen_token_threshold=DEFAULT_LONG_GEN_TOKEN_THRESHOLD,
    long_gen_min_tp_size=DEFAULT_LONG_GEN_MIN_TP_SIZE,
):
    effective_min_tp = max(1, int(base_min_tp_size))
    max_new_tokens = get_mode_max_new_tokens(model_cfg, mode)
    if max_new_tokens >= int(long_gen_token_threshold):
        effective_min_tp = max(effective_min_tp, int(long_gen_min_tp_size))
    return effective_min_tp


def assign_gpu_groups(num_instances, tp_size):
    assignments = []
    gpu_idx = 0
    for _ in range(num_instances):
        gpu_ids = list(range(gpu_idx, gpu_idx + tp_size))
        assignments.append(gpu_ids)
        gpu_idx += tp_size
    return assignments


def detect_gpus():
    script = (
        "import json, torch; "
        "r = {'count': torch.cuda.device_count(), "
        "'vram': torch.cuda.get_device_properties(0).total_memory / (1024**3), "
        "'name': torch.cuda.get_device_properties(0).name} "
        "if torch.cuda.is_available() else "
        "{'count': 0, 'vram': 0, 'name': 'none'}; "
        "print(json.dumps(r))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            print(f"GPU detection stderr:\n{result.stderr}")
            return 0, 0.0, "error"
        info = json.loads(result.stdout.strip())
        return info["count"], info["vram"], info["name"]
    except Exception as exc:
        print(f"GPU detection failed: {exc}")
        return 0, 0.0, "error"


def make_model_variant_name(model_name, mode):
    if mode in {"think", "no_think"}:
        return f"{model_name}_{mode}"
    return model_name


def make_output_filename(model_name, mode, answer_mode):
    variant = make_model_variant_name(model_name, mode)
    safe_answer_mode = answer_mode.replace("/", "_")
    return f"{variant}_{safe_answer_mode}.json"


def write_tracking_log(log_path, header, entries, footer=None):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        for entry in entries:
            f.write(entry + "\n")
        if footer:
            f.write("\n" + footer)


def build_model_tasks(target_models, config_data, modes):
    tasks = []
    for model_name in target_models:
        model_cfg = config_data.get(model_name, {})
        is_thinking = model_cfg.get("thinking", False)
        if is_thinking:
            if modes == "all":
                tasks.append((model_name, "think"))
                tasks.append((model_name, "no_think"))
            else:
                tasks.append((model_name, "think"))
        else:
            tasks.append((model_name, "standard"))
    return tasks


def _normalize_plain_prompts(prompts):
    items = []
    for idx, prompt in enumerate(prompts):
        items.append(
            {
                "item_id": f"item_{idx:05d}",
                "input_index": idx,
                "prompt": prompt,
                "classification_suffix": "",
                "candidates": [],
                "metadata": {},
            }
        )
    return items


def _normalize_candidates(raw_candidates, raw_item):
    if raw_candidates is None:
        raw_keys = raw_item.get("answer_key") or raw_item.get("candidate_keys") or []
        return [
            {"index": idx, "key": str(key), "text": ""}
            for idx, key in enumerate(raw_keys)
        ]

    normalized = []
    for idx, candidate in enumerate(raw_candidates):
        if isinstance(candidate, dict):
            normalized.append(
                {
                    "index": int(candidate.get("index", idx)),
                    "key": str(candidate["key"]),
                    "text": str(candidate.get("text", "")),
                }
            )
        else:
            normalized.append({"index": idx, "key": str(candidate), "text": ""})
    return normalized


def _normalize_structured_items(raw_items):
    items = []
    for idx, raw_item in enumerate(raw_items):
        prompt = raw_item.get("prompt")
        if not prompt:
            statement = raw_item.get("statement", "").strip()
            choices = raw_item.get("choices", "").strip()
            prompt = "\n\n".join(part for part in [statement, choices] if part)

        metadata = dict(raw_item.get("metadata", {}))
        for key in [
            "statement",
            "question_id",
            "page",
            "question_page",
            "answer",
            "answered_option",
        ]:
            if key in raw_item and key not in metadata:
                metadata[key] = raw_item[key]

        item = {
            "item_id": str(raw_item.get("item_id", f"item_{idx:05d}")),
            "input_index": int(raw_item.get("input_index", idx)),
            "prompt": prompt,
            "classification_suffix": raw_item.get("classification_suffix", raw_item.get("suffix", "")),
            "candidates": _normalize_candidates(raw_item.get("candidates"), raw_item),
            "metadata": metadata,
        }
        items.append(item)
    return items


def load_input_items(input_path=None, input_format="auto", test_prompts=None):
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = [line.strip() for line in f if line.strip()]
    else:
        data = test_prompts if test_prompts is not None else DEFAULT_PROMPTS

    if input_format == "auto":
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            input_format = "plain"
        else:
            input_format = "nway"

    if input_format == "plain":
        if not (isinstance(data, list) and all(isinstance(x, str) for x in data)):
            raise ValueError("Plain input format expects a list of strings.")
        return _normalize_plain_prompts(data)

    if input_format == "nway":
        if not (isinstance(data, list) and all(isinstance(x, dict) for x in data)):
            raise ValueError("nway input format expects a JSON list of objects.")
        return _normalize_structured_items(data)

    raise ValueError(f"Unsupported input_format: {input_format}")


def validate_items_for_answer_mode(items, answer_mode):
    if answer_mode in {"chat-classify", "mcq"}:
        for item in items:
            if not item.get("candidates"):
                raise ValueError(
                    f"Item {item.get('item_id')} has no candidates but answer mode is {answer_mode}."
                )


def _get_chat_generation_config(model_cfg, mode):
    chat_cfg = model_cfg.get("chat_generation", {})
    if mode in {"think", "no_think"} and isinstance(chat_cfg.get("think"), dict):
        return dict(chat_cfg.get(mode, chat_cfg.get("no_think", {})))
    return dict(chat_cfg)


def _get_chat_template_kwargs(model_cfg, mode):
    if model_cfg.get("family") == "qwen3":
        if mode == "think":
            return {"enable_thinking": True}
        if mode == "no_think":
            return {"enable_thinking": False}
    return {}


def _build_sampling_params(sampling_cls, **kwargs):
    clean = {key: value for key, value in kwargs.items() if value is not None}
    try:
        return sampling_cls(**clean)
    except TypeError:
        # Keep the fallback narrow and boring so we stay compatible with older
        # installed versions on the cluster.
        for key in [
            "seed",
            "skip_special_tokens",
            "spaces_between_special_tokens",
        ]:
            clean.pop(key, None)
        return sampling_cls(**clean)


def _extract_logprob_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict):
        if "logprob" in value:
            return float(value["logprob"])
    return None


def _extract_prompt_token_logprob(prompt_logprobs, token_id, position=-1):
    if not prompt_logprobs:
        return None

    try:
        entry = prompt_logprobs[position]
    except (IndexError, TypeError):
        return None
    if entry is None:
        return None

    if isinstance(entry, dict):
        if token_id in entry:
            return _extract_logprob_value(entry[token_id])
        str_token_id = str(token_id)
        if str_token_id in entry:
            return _extract_logprob_value(entry[str_token_id])
        if len(entry) == 1:
            return _extract_logprob_value(next(iter(entry.values())))

    if isinstance(entry, list):
        for value in entry:
            if hasattr(value, "token_id") and getattr(value, "token_id") == token_id:
                return _extract_logprob_value(value)
            if isinstance(value, dict):
                candidate_id = value.get("token_id", value.get("token"))
                if candidate_id == token_id:
                    return _extract_logprob_value(value)

    return None


def _extract_prompt_span_logprob(prompt_logprobs, token_ids):
    if not token_ids:
        return None

    total = 0.0
    span_len = len(token_ids)
    for idx, token_id in enumerate(token_ids):
        position = -span_len + idx
        value = _extract_prompt_token_logprob(prompt_logprobs, token_id, position=position)
        if value is None:
            return None
        total += value
    return total


def _compute_divergent_suffix(base_ids, candidate_ids):
    max_prefix = min(len(base_ids), len(candidate_ids))
    prefix_len = 0
    while prefix_len < max_prefix and base_ids[prefix_len] == candidate_ids[prefix_len]:
        prefix_len += 1

    divergent_ids = candidate_ids[prefix_len:]
    if not divergent_ids:
        return None, None
    return prefix_len, divergent_ids


def _softmax_from_logprobs(logprob_map):
    if not logprob_map:
        return {}
    max_logprob = max(logprob_map.values())
    exp_values = {
        key: math.exp(value - max_logprob)
        for key, value in logprob_map.items()
    }
    denom = sum(exp_values.values())
    if denom == 0:
        return {key: 0.0 for key in logprob_map}
    return {key: value / denom for key, value in exp_values.items()}


class VLLMRunner:
    """
    Load one model once and reuse it across many item batches.
    """

    def __init__(
        self,
        model_path,
        model_name,
        mode,
        gpu_ids,
        config_data,
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len=DEFAULT_VLLM_MAX_MODEL_LEN,
        enforce_eager=True,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = str(model_path)
        self.model_name = model_name
        self.mode = mode
        self.gpu_ids = list(gpu_ids)
        self.tp_size = max(1, len(self.gpu_ids))
        self.model_cfg = config_data.get(model_name, {})
        self.chat_template_kwargs = _get_chat_template_kwargs(self.model_cfg, mode)
        self.SamplingParams = SamplingParams
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        self.enforce_eager = bool(enforce_eager)

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")

        t_load = time.time()
        revision = self.model_cfg.get("revision") if not Path(self.model_path).is_dir() else None
        tokenizer_kwargs = {"trust_remote_code": True}
        if revision:
            tokenizer_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": self.tp_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if revision:
            llm_kwargs["revision"] = revision

        self.llm = LLM(
            **llm_kwargs,
        )
        self.load_seconds = round(time.time() - t_load, 2)

    def close(self):
        if getattr(self, "llm", None) is not None:
            del self.llm
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
        gc.collect()

    def _token_ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _safe_apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if continue_final_message:
            kwargs["continue_final_message"] = True
        kwargs.update(self.chat_template_kwargs)

        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            if continue_final_message:
                return None
            retry_kwargs = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            retry_kwargs.update(self.chat_template_kwargs)
            try:
                return self.tokenizer.apply_chat_template(messages, **retry_kwargs)
            except Exception:
                return None
        except Exception:
            return None

    def _call_generate(self, prompts, sampling_params):
        try:
            return self.llm.generate(prompts, sampling_params, use_tqdm=False)
        except TypeError:
            return self.llm.generate(prompts, sampling_params)

    def _init_record(self, item, answer_mode):
        candidates = item.get("candidates", [])
        return {
            "item_id": item["item_id"],
            "input_index": item["input_index"],
            "prompt": item["prompt"],
            "answer_mode": answer_mode,
            "classification_suffix": item.get("classification_suffix", ""),
            "candidate_keys": [candidate["key"] for candidate in candidates],
            "candidate_texts": [candidate.get("text", "") for candidate in candidates],
            "metadata": dict(item.get("metadata", {})),
            "stage1_text": None,
            "stage1_tokens": 0,
            "stage1_finish_reason": None,
            "classification_logprobs": None,
            "classification_probs": None,
            "classification_pred_key": None,
            "error": None,
        }

    def _error_record(self, item, answer_mode, error_message):
        record = self._init_record(item, answer_mode)
        record["error"] = str(error_message)
        return record

    def _build_stage1_prompt(self, item):
        prompt = self._safe_apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            add_generation_prompt=True,
        )
        return prompt if prompt is not None else item["prompt"]

    def _build_stage2_base_prompt(self, item, stage1_text, stage1_prompt):
        suffix = item.get("classification_suffix", "")
        assistant_prefix = f"{stage1_text}{suffix}"
        rendered = self._safe_apply_chat_template(
            [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": assistant_prefix},
            ],
            continue_final_message=True,
        )
        if rendered is not None:
            return rendered
        return f"{stage1_prompt}{assistant_prefix}"

    def _run_stage1_generation(self, items, answer_mode, batch_size):
        generation_cfg = _get_chat_generation_config(self.model_cfg, self.mode)
        sampling_params = _build_sampling_params(
            self.SamplingParams,
            temperature=generation_cfg.get("temperature", 0.7),
            top_p=generation_cfg.get("top_p", 0.95),
            top_k=generation_cfg.get("top_k", -1),
            max_tokens=generation_cfg.get("max_new_tokens", 4096),
            seed=generation_cfg.get("seed"),
        )

        records = []
        stage1_prompts = {}

        for start_idx in range(0, len(items), batch_size):
            chunk_items = items[start_idx : start_idx + batch_size]
            chunk_prompts = [self._build_stage1_prompt(item) for item in chunk_items]
            try:
                outputs = self._call_generate(chunk_prompts, sampling_params)
            except Exception as exc:
                for item, prompt in zip(chunk_items, chunk_prompts):
                    stage1_prompts[item["item_id"]] = prompt
                    records.append(self._error_record(item, answer_mode, f"stage1_generation_failed: {exc}"))
                continue

            for item, prompt, output in zip(chunk_items, chunk_prompts, outputs):
                stage1_prompts[item["item_id"]] = prompt
                record = self._init_record(item, answer_mode)
                try:
                    sample = output.outputs[0]
                    record["stage1_text"] = sample.text
                    record["stage1_tokens"] = len(getattr(sample, "token_ids", []) or [])
                    record["stage1_finish_reason"] = (
                        getattr(sample, "finish_reason", None)
                        or getattr(sample, "stop_reason", None)
                    )
                except Exception as exc:
                    record["error"] = f"stage1_parse_failed: {exc}"
                records.append(record)

        return records, stage1_prompts

    def _score_candidates(self, item_bundle, batch_size):
        sampling_params = _build_sampling_params(
            self.SamplingParams,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            prompt_logprobs=1,
        )

        per_item = {}
        prompt_requests = []

        for item, base_prompt in item_bundle:
            item_id = item["item_id"]
            per_item[item_id] = {"logprobs": {}, "error": None}
            base_ids = self._token_ids(base_prompt)
            if not base_ids:
                per_item[item_id]["error"] = "classification_base_prompt_tokenized_to_empty_sequence"
                continue

            for candidate in item.get("candidates", []):
                candidate_key = str(candidate["key"])
                candidate_surface = str(candidate.get("score_text", candidate_key))
                candidate_prompt = f"{base_prompt}{candidate_surface}"
                candidate_ids = self._token_ids(candidate_prompt)
                score_start, score_token_ids = _compute_divergent_suffix(base_ids, candidate_ids)
                if not score_token_ids:
                    per_item[item_id]["error"] = (
                        f"candidate_key_tokenized_to_empty_suffix_in_context: {candidate_key}"
                    )
                    per_item[item_id]["logprobs"] = {}
                    prompt_requests = [
                        req for req in prompt_requests if req["item_id"] != item_id
                    ]
                    break

                prompt_requests.append(
                    {
                        "item_id": item_id,
                        "candidate_key": candidate_key,
                        "candidate_prompt": candidate_prompt,
                        "candidate_token_ids": score_token_ids,
                        "candidate_start": score_start,
                    }
                )

        for start_idx in range(0, len(prompt_requests), batch_size):
            chunk_requests = prompt_requests[start_idx : start_idx + batch_size]
            chunk_prompts = [request["candidate_prompt"] for request in chunk_requests]
            try:
                outputs = self._call_generate(chunk_prompts, sampling_params)
            except Exception as exc:
                for request in chunk_requests:
                    item_state = per_item[request["item_id"]]
                    if item_state["error"] is None:
                        item_state["error"] = f"classification_generate_failed: {exc}"
                continue

            for request, output in zip(chunk_requests, outputs):
                item_state = per_item[request["item_id"]]
                if item_state["error"] is not None:
                    continue
                prompt_logprobs = getattr(output, "prompt_logprobs", None)
                logprob = _extract_prompt_span_logprob(prompt_logprobs, request["candidate_token_ids"])
                if logprob is None and prompt_logprobs:
                    total = 0.0
                    for offset, token_id in enumerate(request["candidate_token_ids"]):
                        value = _extract_prompt_token_logprob(
                            prompt_logprobs,
                            token_id,
                            position=request["candidate_start"] + offset,
                        )
                        if value is None:
                            total = None
                            break
                        total += value
                    logprob = total
                if logprob is None:
                    item_state["error"] = (
                        f"missing_prompt_logprob_for_candidate: {request['candidate_key']}"
                    )
                    item_state["logprobs"] = {}
                    continue
                item_state["logprobs"][request["candidate_key"]] = logprob

        results = {}
        for item_id, item_state in per_item.items():
            if item_state["error"] is not None:
                results[item_id] = {
                    "classification_logprobs": None,
                    "classification_probs": None,
                    "classification_pred_key": None,
                    "error": item_state["error"],
                }
                continue

            probs = _softmax_from_logprobs(item_state["logprobs"])
            pred_key = max(probs, key=probs.get) if probs else None
            results[item_id] = {
                "classification_logprobs": item_state["logprobs"],
                "classification_probs": probs,
                "classification_pred_key": pred_key,
                "error": None,
            }

        return results

    def run_items(self, items, answer_mode, batch_size=DEFAULT_CHAT_BATCH_SIZE, classify_batch_size=DEFAULT_CLASSIFY_BATCH_SIZE):
        validate_items_for_answer_mode(items, answer_mode)

        if answer_mode == "chat":
            records, _ = self._run_stage1_generation(items, answer_mode, batch_size)
            return records

        if answer_mode == "chat-classify":
            records, stage1_prompts = self._run_stage1_generation(items, answer_mode, batch_size)
            record_map = {record["item_id"]: record for record in records}
            to_score = []
            for item in items:
                record = record_map[item["item_id"]]
                if record["error"] is not None:
                    continue
                base_prompt = self._build_stage2_base_prompt(
                    item,
                    record["stage1_text"] or "",
                    stage1_prompts[item["item_id"]],
                )
                to_score.append((item, base_prompt))

            scored = self._score_candidates(to_score, classify_batch_size)
            for item_id, score in scored.items():
                record = record_map[item_id]
                record["classification_logprobs"] = score["classification_logprobs"]
                record["classification_probs"] = score["classification_probs"]
                record["classification_pred_key"] = score["classification_pred_key"]
                if score["error"] is not None:
                    record["error"] = score["error"]
            return records

        if answer_mode == "mcq":
            records = [self._init_record(item, answer_mode) for item in items]
            record_map = {record["item_id"]: record for record in records}
            to_score = [(item, item["prompt"]) for item in items]
            scored = self._score_candidates(to_score, classify_batch_size)
            for item_id, score in scored.items():
                record = record_map[item_id]
                record["classification_logprobs"] = score["classification_logprobs"]
                record["classification_probs"] = score["classification_probs"]
                record["classification_pred_key"] = score["classification_pred_key"]
                if score["error"] is not None:
                    record["error"] = score["error"]
            return records

        raise ValueError(f"Unsupported answer mode: {answer_mode}")


def run_vllm_job(
    model_path,
    model_name,
    tp_size,
    gpu_ids,
    items,
    config_data,
    mode,
    answer_mode,
    batch_size,
    classify_batch_size,
    gpu_memory_utilization,
    max_model_len,
    enforce_eager,
    instance_id,
    result_queue,
):
    display_name = make_model_variant_name(model_name, mode)
    tag = f"[inst-{instance_id}]" if instance_id >= 0 else ""
    print(
        f"\n{'=' * 60}\n"
        f"Loading: {display_name} {tag}\n"
        f"  TP={tp_size} | GPUs={gpu_ids} | Items={len(items)} | Mode={answer_mode}\n"
        f"{'=' * 60}"
    )

    runner = None
    try:
        runner = VLLMRunner(
            model_path,
            model_name,
            mode,
            gpu_ids,
            config_data,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        t_run = time.time()
        records = runner.run_items(
            items,
            answer_mode=answer_mode,
            batch_size=batch_size,
            classify_batch_size=classify_batch_size,
        )
        run_seconds = round(time.time() - t_run, 2)
        total_stage1_tokens = sum(record.get("stage1_tokens", 0) for record in records)
        result_queue.put(
            {
                "instance_id": instance_id,
                "records": records,
                "timing": {
                    "model_load_seconds": runner.load_seconds,
                    "run_seconds": run_seconds,
                    "total_seconds": round(runner.load_seconds + run_seconds, 2),
                    "total_stage1_tokens": total_stage1_tokens,
                },
            }
        )
    except Exception as exc:
        import traceback

        print(f"\n  ✗ Error in {display_name} {tag}: {exc}")
        traceback.print_exc()
        result_queue.put(
            {
                "instance_id": instance_id,
                "records": [
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
                        "error": f"worker_failed: {exc}",
                    }
                    for item in items
                ],
                "timing": {"error": str(exc)},
            }
        )
    finally:
        if runner is not None:
            runner.close()


def _missing_record_for_item(item, answer_mode):
    return {
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
        "error": "instance_crashed",
    }


def run_cli_experiment(args):
    if not 0.0 < args.dp_threshold <= 1.0:
        raise ValueError(f"--dp-threshold must be in (0, 1], got {args.dp_threshold}")
    if args.min_tp_size < 1:
        raise ValueError(f"--min-tp-size must be >= 1, got {args.min_tp_size}")
    if args.long_gen_token_threshold < 1:
        raise ValueError(
            "--long-gen-token-threshold must be >= 1, "
            f"got {args.long_gen_token_threshold}"
        )
    if args.long_gen_min_tp_size < 1:
        raise ValueError(
            f"--long-gen-min-tp-size must be >= 1, got {args.long_gen_min_tp_size}"
        )

    if args.input:
        items = load_input_items(args.input, args.input_format)
    elif args.test is not None:
        test_prompts = DEFAULT_PROMPTS if len(args.test) == 0 else args.test
        items = load_input_items(None, args.input_format, test_prompts=test_prompts)
    else:
        print("Error: Provide --test or --input.")
        sys.exit(1)

    validate_items_for_answer_mode(items, args.answer)
    print(f"Loaded {len(items)} items")

    config_data = load_model_config(args.models_dir)
    target_models = PRIMARY_MODELS if "all" in [model.lower() for model in args.models] else args.models
    unsupported = sorted(set(target_models) - set(SUPPORTED_MODELS))
    if unsupported:
        raise ValueError("Unsupported model aliases: " + ", ".join(unsupported))

    detected_gpus, vram_per_gpu, gpu_name = detect_gpus()
    num_gpus = min(args.gpus, detected_gpus)
    if num_gpus == 0:
        print(
            f"Error: No GPUs detected (got count={detected_gpus}, name={gpu_name})."
        )
        print(
            "Hint: Make sure you're on a GPU node (srun --gres=gpu:N) and the environment has torch+CUDA+vLLM."
        )
        sys.exit(1)

    mp.set_start_method("spawn", force=True)

    print(
        f"GPU: {gpu_name} | {vram_per_gpu:.1f} GB/GPU | "
        f"{detected_gpus} detected, using {num_gpus}"
    )
    print(
        f"Scheduling threshold: weights < {args.dp_threshold * 100:.0f}% VRAM → DP, else → TP"
        f" | min_tp={args.min_tp_size} | long_gen>={args.long_gen_token_threshold} => min_tp={args.long_gen_min_tp_size}\n"
    )

    tasks = build_model_tasks(target_models, config_data, args.modes)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = len(tasks)
    run_start = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tracking_path = output_dir / "_progress.log"
    tracking_header = (
        f"{'=' * 70}\n"
        f"  Experiment : {output_dir.name}\n"
        f"  Started    : {start_ts}\n"
        f"  GPU        : {num_gpus}× {gpu_name} ({vram_per_gpu:.0f} GB each)\n"
        f"  Items      : {len(items)}\n"
        f"  Tasks      : {total_tasks}\n"
        f"  Answer     : {args.answer}\n"
        f"  Threshold  : weights < {DP_THRESHOLD * 100:.0f}% GPU VRAM → DP, else → TP\n"
        f"{'=' * 70}\n"
    )
    tracking_entries = []
    completed_count = 0
    skipped_count = 0

    def _update_tracking(footer_extra=""):
        elapsed = time.time() - run_start
        done = completed_count + skipped_count
        remaining = total_tasks - done
        eta_str = f"~{(elapsed / done) * remaining:.0f}s" if done else "calculating..."
        footer = (
            f"\n{'─' * 70}\n"
            f"  Progress : {done}/{total_tasks} "
            f"({completed_count} done, {skipped_count} skipped, {remaining} remaining)\n"
            f"  Elapsed  : {elapsed:.1f}s\n"
            f"  ETA      : {eta_str}\n"
            f"  Updated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'─' * 70}"
            + (f"\n{footer_extra}" if footer_extra else "")
        )
        write_tracking_log(tracking_path, tracking_header, tracking_entries, footer)

    _update_tracking()

    for task_idx, (model_name, mode) in enumerate(tasks, 1):
        display_name = make_model_variant_name(model_name, mode)
        try:
            model_path, _revision = resolve_model_source(model_name, args.models_dir, config_data)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            msg = f"[{task_idx}/{total_tasks}] Skip {display_name}: {exc}"
            print(msg)
            tracking_entries.append(msg)
            skipped_count += 1
            _update_tracking()
            continue
        model_cfg = config_data.get(model_name, {})
        params_b = model_cfg.get("params_B")
        if params_b is None:
            tp_size, num_instances, strategy = 1, 1, "default"
        else:
            effective_min_tp = compute_effective_min_tp_size(
                model_cfg,
                mode,
                base_min_tp_size=args.min_tp_size,
                long_gen_token_threshold=args.long_gen_token_threshold,
                long_gen_min_tp_size=args.long_gen_min_tp_size,
            )
            schedule = compute_schedule(
                params_b,
                vram_per_gpu,
                num_gpus,
                dp_threshold=args.dp_threshold,
                min_tp_size=effective_min_tp,
            )
            if schedule is None:
                msg = (
                    f"[{task_idx}/{total_tasks}] Skip {display_name}: "
                    f"too large ({params_b}B) for {num_gpus}× {vram_per_gpu:.0f}GB GPUs"
                )
                print(msg)
                tracking_entries.append(msg)
                skipped_count += 1
                err_path = output_dir / make_output_filename(model_name, mode, args.answer)
                with open(err_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "_meta": {
                                "model_name": model_name,
                                "mode": mode,
                                "answer_mode": args.answer,
                                "error": "Model too large for available GPUs",
                            }
                        },
                        f,
                        indent=2,
                    )
                _update_tracking()
                continue
            tp_size, num_instances, strategy = schedule

        num_instances = max(1, min(num_instances, len(items)))
        gpu_groups = assign_gpu_groups(num_instances, tp_size)
        item_chunks = [items[idx::num_instances] for idx in range(num_instances)]

        print(f"[{task_idx}/{total_tasks}] Running {display_name}")
        print(
            f"    {strategy.upper()} | tp={tp_size} | instances={num_instances} | answer={args.answer}"
        )

        t_wall = time.time()
        result_queue = mp.Queue()
        processes = []
        for inst_idx, gpu_group in enumerate(gpu_groups):
            process = mp.Process(
                target=run_vllm_job,
                args=(
                    model_path,
                    model_name,
                    tp_size,
                    gpu_group,
                    item_chunks[inst_idx],
                    config_data,
                    mode,
                    args.answer,
                    args.batch_size,
                    args.classify_batch_size,
                    args.gpu_memory_utilization,
                    args.max_model_len,
                    args.enforce_eager,
                    inst_idx,
                    result_queue,
                ),
            )
            processes.append(process)
            process.start()

        instance_results = []
        deadline = time.time() + args.deadline_seconds
        while len(instance_results) < num_instances and time.time() < deadline:
            try:
                instance_results.append(result_queue.get(timeout=5))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    while True:
                        try:
                            instance_results.append(result_queue.get_nowait())
                        except queue.Empty:
                            break
                    break

        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                print(f"    Terminating hung process {process.pid}")
                process.terminate()
                process.join(timeout=10)

        wall_seconds = round(time.time() - t_wall, 2)
        merged_records = {}
        timing_per_instance = []
        for instance_result in instance_results:
            timing_per_instance.append(instance_result.get("timing", {}))
            for record in instance_result.get("records", []):
                merged_records[record["item_id"]] = record

        for item in items:
            if item["item_id"] not in merged_records:
                merged_records[item["item_id"]] = _missing_record_for_item(item, args.answer)

        ordered_records = sorted(merged_records.values(), key=lambda record: record["input_index"])
        success_count = sum(1 for record in ordered_records if record.get("error") is None)
        total_stage1_tokens = sum(record.get("stage1_tokens", 0) for record in ordered_records)

        output_data = {
            "_meta": {
                "model_name": model_name,
                "mode": mode,
                "model_variant": make_model_variant_name(model_name, mode),
                "answer_mode": args.answer,
                "input_format": args.input_format,
                "params_B": params_b,
                "strategy": strategy,
                "tp_size": tp_size,
                "num_instances": num_instances,
                "gpu_name": gpu_name,
                "gpu_vram_gb": round(vram_per_gpu, 1),
                "num_gpus_used": num_instances * tp_size,
                "num_items": len(items),
                "num_items_completed": success_count,
                "total_stage1_tokens": total_stage1_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timing": {
                    "wall_seconds": wall_seconds,
                    "per_instance": timing_per_instance,
                },
            },
            "records": ordered_records,
        }

        out_path = output_dir / make_output_filename(model_name, mode, args.answer)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        summary = (
            f"[{task_idx}/{total_tasks}] {display_name:40s} "
            f"| {strategy.upper():2s} tp={tp_size} ×{num_instances} "
            f"| {wall_seconds:7.1f}s "
            f"| {success_count}/{len(items)} items "
            f"| {total_stage1_tokens:>7} tok "
            f"| → {out_path.name}"
        )
        print(summary)
        tracking_entries.append(summary)
        completed_count += 1
        _update_tracking()

    total_seconds = time.time() - run_start
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_footer = (
        f"\n{'=' * 70}\n"
        f"  COMPLETE\n"
        f"  Finished : {end_ts}\n"
        f"  Total    : {total_seconds:.1f}s\n"
        f"  Done     : {completed_count}/{total_tasks} ({skipped_count} skipped)\n"
        f"  Results  : {output_dir}/\n"
        f"{'=' * 70}"
    )
    _update_tracking(final_footer)

    print(f"\n{'=' * 60}")
    print(f"  All {total_tasks} tasks complete in {total_seconds:.1f}s")
    print(f"  Results: {output_dir}/")
    print(f"  Tracking: {tracking_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Reusable vLLM runner for PoliLean release experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model names to run, or 'all'.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Local checkpoint root and model_config.json location; Hub IDs are the fallback.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        help="Number of GPUs available. Auto-schedules TP/DP per model.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        choices=["default", "all"],
        default="default",
        help="'default' = one run mode per model, 'all' = think + no_think for reasoning models.",
    )
    parser.add_argument(
        "--answer",
        type=str,
        choices=["chat", "chat-classify", "mcq"],
        default="chat",
        help="How to answer each item.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["auto", "plain", "nway"],
        default="auto",
        help="Interpretation of the input file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CHAT_BATCH_SIZE,
        help="Generation batch size inside each vLLM worker.",
    )
    parser.add_argument(
        "--classify-batch-size",
        type=int,
        default=DEFAULT_CLASSIFY_BATCH_SIZE,
        help="Classification prompt batch size inside each vLLM worker.",
    )
    parser.add_argument(
        "--dp-threshold",
        type=float,
        default=DP_THRESHOLD,
        help="Weight-to-VRAM fraction target under which DP is allowed.",
    )
    parser.add_argument(
        "--min-tp-size",
        type=int,
        default=DEFAULT_MIN_TP_SIZE,
        help="Global minimum tensor parallel size for all models.",
    )
    parser.add_argument(
        "--long-gen-token-threshold",
        type=int,
        default=DEFAULT_LONG_GEN_TOKEN_THRESHOLD,
        help="If max_new_tokens >= this value, enforce long-generation TP policy.",
    )
    parser.add_argument(
        "--long-gen-min-tp-size",
        type=int,
        default=DEFAULT_LONG_GEN_MIN_TP_SIZE,
        help="Minimum tensor parallel size for long-generation models.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=(
            "Target fraction of GPU memory reserved by vLLM (default: "
            f"{DEFAULT_GPU_MEMORY_UTILIZATION})."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_VLLM_MAX_MODEL_LEN,
        help=(
            "Optional vLLM max model context length override. "
            "Lower values reduce KV-cache pressure."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run vLLM in eager mode instead of CUDA graph mode.",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=int,
        default=DEFAULT_DEADLINE_SECONDS,
        help="Hard deadline per model task while waiting on worker processes.",
    )
    parser.add_argument(
        "--test",
        nargs="*",
        default=None,
        help="Test mode. Blank = default prompts, or pass inline prompts.",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to a JSON/TXT input file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")

    args = parser.parse_args()
    run_cli_experiment(args)


if __name__ == "__main__":
    main()
