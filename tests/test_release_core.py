from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_model_resolver_uses_pinned_hub_fallback():
    resolver = load_module("resolve_model", ROOT / "scripts" / "resolve_model.py")
    result = resolver.resolve("Qwen3-8B", ROOT / "models" / "serve")
    assert result["alias"] == "Qwen3-8B"
    assert result["source"] == "Qwen/Qwen3-8B"
    assert result["revision"]
    assert result["primary"] is True


def test_download_helpers_select_only_primary_models():
    downloader = load_module("download_models", ROOT / "scripts" / "download_models.py")
    config = json.loads(
        (ROOT / "models" / "serve" / "model_config.json").read_text(encoding="utf-8")
    )
    selected = downloader.parse_models("all", config)
    assert len(selected) == 8
    assert all(config[name]["primary"] for name in selected)
    assert not any(config[name].get("secondary_ablation") for name in selected)


def test_model_config_contains_only_primary_plus_named_ablation():
    config = json.loads((ROOT / "models" / "serve" / "model_config.json").read_text())
    primary = [name for name, value in config.items() if not name.startswith("_") and value.get("primary")]
    secondary = [name for name, value in config.items() if isinstance(value, dict) and value.get("secondary_ablation")]
    assert primary == [
        "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
        "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
    ]
    assert secondary == ["gemma-3-27b-it-abliterated-normpreserve-v1"]
    assert all(value.get("repo_id") and value.get("revision") for name, value in config.items() if not name.startswith("_"))


def test_pct_mcq_lhs_is_deterministic_and_primary_scoped():
    module = load_module("pct_mcq", ROOT / "tasks" / "political-compass" / "mcq.py")
    models = ["gemma-3-1b-it", "Qwen3-4B"]
    left = module.generate_experimental_design(3, 42, models, ["english"])
    right = module.generate_experimental_design(3, 42, models, ["english"])
    assert left.equals(right)
    assert len(left) == 2 * 6 * 3
    assert left["config_id"].is_unique
    assert set(left["model"]) == set(models)


def test_pct_mcq_release_seed_and_global_id_blocks():
    module = load_module(
        "pct_mcq_release_design", ROOT / "tasks" / "political-compass" / "mcq.py"
    )
    design = module.generate_experimental_design(
        300, 42, ["gemma-3-1b-it", "Qwen3-4B"], module.LANGUAGES
    )
    first = design.groupby("model", sort=False).first()
    assert int(first.loc["gemma-3-1b-it", "config_id"]) == 14_400
    assert int(first.loc["Qwen3-4B", "config_id"]) == 18_000
    assert first.loc["gemma-3-1b-it", "language"] == "german"
    assert int(first.loc["gemma-3-1b-it", "instr_idx"]) == 4
    assert first.loc["gemma-3-1b-it", "key_type"] == "numeric_zero_index"


def test_pct_prompt_permutation_reverses_labels_and_keys():
    module = load_module("pct_mcq_prompt", ROOT / "tasks" / "political-compass" / "mcq.py")
    prompts = {
        "contexts": [{"text": "context"}],
        "experiment_settings": {"answer_keys": {"numeric": ["1", "2", "3", "4"]}},
        "instructions": {"question_first": [{"text": "{question}{options_formatted}"}]},
        "persona_templates": {"short": [{"id": "p", "text": "{name}"}]},
        "ideology_insertions": {"libertarian_left": {"p": {"name": "persona"}}},
    }
    row = {
        "context_id": None, "ideology": "base", "perm_id": 3,
        "key_type": "numeric", "instr_type": "question_first", "instr_idx": 0,
    }
    text, keys = module.assemble_prompt_for_question(
        row, {"statement": "S", "choices": ["SA", "A", "D", "SD"]}, prompts
    )
    assert keys == ["4", "3", "2", "1"]
    assert "4. SD" in text and "1. SA" in text


def test_candidate_softmax_sums_to_one():
    import torch

    module = load_module("pct_mcq_softmax", ROOT / "tasks" / "political-compass" / "mcq.py")
    values = module.candidate_softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert np.isclose(values.sum().item(), 1.0)
    assert values.argmax().item() == 3


def test_ibm_design_and_topic_inventory():
    module = load_module("ibm_sentiment", ROOT / "tasks" / "sentiment" / "run_ibm_sentiment.py")
    first = module.generate_experimental_design(2, 42, ["gemma-3-1b-it"])
    second = module.generate_experimental_design(2, 42, ["gemma-3-1b-it"])
    assert first.equals(second)
    assert len(first) == 12
    items = module.load_experiment_items()
    assert len(items) == 30
    assert sum(item["gold_label"] == "POSITIVE" for item in items) == 19
    assert sum(item["gold_label"] == "NEGATIVE" for item in items) == 11


def hate_prompt_fixture() -> dict:
    templates = {
        kind: [{"id": f"{kind}_{idx}", "text": "{label}"} for idx in range(5)]
        for kind, label in (("short", "{name}"), ("long", "{name}"))
    }
    insertions = {
        ideology: {
            template["id"]: {"name": ideology}
            for values in templates.values() for template in values
        }
        for ideology in [
            "libertarian_right", "authoritarian_left", "authoritarian_right",
            "libertarian_left", "centrism",
        ]
    }
    return {
        "contexts": [{"text": f"context {idx}"} for idx in range(5)],
        "instructions": [{"text": f"instruction {idx}: {{text}}"} for idx in range(10)],
        "persona_templates": templates,
        "ideology_insertions": insertions,
    }


def test_hate_design_and_input_validation(tmp_path):
    module = load_module("hate_speech", ROOT / "tasks" / "hate-speech" / "run_hate_speech.py")
    left = module.generate_design(3, 42, ["gemma-3-1b-it"])
    right = module.generate_design(3, 42, ["gemma-3-1b-it"])
    assert left.equals(right)
    assert len(left) == 18

    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(hate_prompt_fixture()), encoding="utf-8")
    module.load_prompts(prompt_path)

    corpus_path = tmp_path / "corpus.jsonl"
    records = []
    for target in module.TARGETS:
        records.append(
            {
                "text": f"fixture for {target}", "hate": False,
                "target_groups": [target], "dataset": "fixture", "grouping": "test",
            }
        )
    corpus_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    items, grouped = module.load_corpus(corpus_path)
    assert len(items) == 10
    assert all(len(values) == 1 for values in grouped.values())


def test_hate_default_design_matches_completed_handoff():
    module = load_module("hate_speech_exact_design", ROOT / "tasks" / "hate-speech" / "run_hate_speech.py")
    generated = module.generate_design(300, 42)
    supplied = pd.read_csv(ROOT / "dataset" / "generated" / "hate_speech_design.csv")
    supplied = supplied[
        [
            "config_id", "language", "target", "context_id", "instr_idx",
            "persona_class", "persona_idx", "model_alias", "model", "ideology",
            "quantization",
        ]
    ]
    for frame in (generated, supplied):
        frame["context_id"] = pd.to_numeric(frame["context_id"], errors="coerce").fillna(-1).astype(int)
        frame["persona_idx"] = pd.to_numeric(frame["persona_idx"], errors="coerce").fillna(-1).astype(int)
        frame["persona_class"] = frame["persona_class"].fillna("").astype(str)
    assert generated.reset_index(drop=True).equals(supplied.reset_index(drop=True))


def test_hate_design_includes_released_model_identity():
    module = load_module(
        "hate_design_model_identity", ROOT / "tasks" / "hate-speech" / "run_hate_speech.py"
    )
    design = module.generate_design(1, 42, ["gemma-3-12b-it"])
    assert design.columns.tolist() == [
        "config_id", "language", "target", "context_id", "instr_idx",
        "persona_class", "persona_idx", "model_alias", "model", "ideology",
        "quantization",
    ]
    assert design.iloc[0]["model"] == "google/gemma-3-12b-it"
    assert design.iloc[0]["persona_idx"] == -1


def test_historical_hate_conversion_uses_positional_item_index():
    module = load_module(
        "hate_historical_converter",
        ROOT / "tasks" / "hate-speech" / "analysis" / "convert_historical_results.py",
    )
    prefixes = [f"q{index}" for index in range(1, 1_333)]
    base = {
        "config_id": [0, 1],
        "model": ["google/gemma-3-12b-it"] * 2,
        "quantization": ["bf16"] * 2,
        "language": ["english"] * 2,
        "ideology": ["base"] * 2,
        "context_id": [-1, 0],
        "instr_idx": [1, 2],
        "persona_class": ["", ""],
        "persona_idx": [-1, -1],
    }
    for index, prefix in enumerate(prefixes, 1):
        base[prefix + "_is_hate_speech"] = [index % 2 == 0, index % 2 == 0]
        base[prefix + "_target"] = ["women", "men"]
        base[prefix + "_meta"] = ["fixture;a", "fixture;b"]
        base[prefix + "_prob_ans_0"] = [0.25, 0.75]
        base[prefix + "_prob_ans_1"] = [0.75, 0.25]
    chunk = pd.DataFrame(base)
    expected = pd.DataFrame(
        {
            "config_id": [0, 1],
            "model_alias": ["gemma-3-12b-it"] * 2,
            "language": ["english"] * 2,
            "target": ["women", "men"],
            "context_id": [-1, 0],
            "instr_idx": [1, 2],
            "persona_class": ["", ""],
            "persona_idx": [-1, -1],
            "ideology": ["base"] * 2,
        }
    ).set_index("config_id", drop=False)
    long = module._long_chunk(chunk, prefixes, expected)
    assert len(long) == 2 * 1_332
    assert long.groupby("config_id")["item_index"].agg(["min", "max"]).to_dict("index") == {
        0: {"min": 1, "max": 1_332},
        1: {"min": 1, "max": 1_332},
    }
    assert set(long.loc[long["config_id"].eq(0), "target"]) == {"women"}
    assert set(long.loc[long["config_id"].eq(1), "target"]) == {"men"}
    assert np.allclose(long["p_hate"] + long["p_not_hate"], 1.0)


def test_tracked_analysis_shapes():
    sentiment = ROOT / "tasks" / "sentiment" / "analysis" / "data"
    hate = ROOT / "tasks" / "hate-speech" / "analysis" / "data"
    coverage = pd.read_csv(sentiment / "coverage_by_model.csv")
    assert len(coverage) == 8
    assert coverage["successful_rows"].sum() == 432_000

    hate_item = pd.read_csv(hate / "hs_item_metrics_by_persona_target.csv")
    hate_calibration = pd.read_csv(hate / "hs_calibration_by_model.csv")
    assert len(hate_item) == 8 * 6 * 10
    assert len(hate_calibration) == 8 * 10
    assert hate_item.groupby(["model_alias", "ideology"])["target"].nunique().eq(10).all()
    assert hate_calibration.groupby("model_alias")["probability_bin"].nunique().eq(10).all()
    assert coverage["error_rows"].sum() == 0
    assert len(pd.read_csv(sentiment / "topic_descriptive_metrics.csv")) == 8 * 6 * 30
    assert len(pd.read_csv(hate / "hs_configuration_scores.csv")) == 8 * 6 * 300
