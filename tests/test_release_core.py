from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_tracked_analysis_shapes():
    sentiment = ROOT / "tasks" / "sentiment" / "analysis" / "data"
    hate = ROOT / "tasks" / "hate-speech" / "analysis" / "data"
    import pandas as pd

    coverage = pd.read_csv(sentiment / "coverage_by_model.csv")
    assert len(coverage) == 8
    assert coverage["successful_rows"].sum() == 432_000
    assert coverage["error_rows"].sum() == 0
    assert len(pd.read_csv(sentiment / "topic_descriptive_metrics.csv")) == 8 * 6 * 30
    assert len(pd.read_csv(hate / "hs_configuration_scores.csv")) == 8 * 6 * 300
