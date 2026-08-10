"""Shared configuration for the Political Compass qualitative analyses."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
POLITICAL_COMPASS_ROOT = HERE.parent
REPO_ROOT = POLITICAL_COMPASS_ROOT.parent.parent

RAW_CHAT_DIR = POLITICAL_COMPASS_ROOT / "output" / "chat"
DESIGN_PATH = RAW_CHAT_DIR / "experimental_design.csv"
QUESTIONS_PATH = POLITICAL_COMPASS_ROOT / "data" / "questions" / "english.json"
PROMPTS_PATH = POLITICAL_COMPASS_ROOT / "data" / "chat-prompts" / "english.json"
SCORER_PATH = (
    POLITICAL_COMPASS_ROOT
    / "instrument"
    / "reconstructed_model_parameters.npz"
)

ARTIFACT_ROOT = HERE / "artifacts"
CORPUS_DIR = ARTIFACT_ROOT / "corpus"
FEATURE_DIR = ARTIFACT_ROOT / "features"
EMBEDDING_DIR = ARTIFACT_ROOT / "embeddings"
TABLE_DIR = ARTIFACT_ROOT / "tables"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
CASEBOOK_DIR = ARTIFACT_ROOT / "casebooks"
LOG_DIR = ARTIFACT_ROOT / "logs"


GENERAL_MODEL_ORDER_1 = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B_no_think",
    "Qwen3-8B_no_think",
    "Qwen3-14B_no_think",
    "Qwen3-32B_no_think",
    "Qwen3-4B_think",
    "Qwen3-8B_think",
    "Qwen3-14B_think",
    "Qwen3-32B_think",
]

GENERAL_MODEL_ORDER_2 = [
    "gemma-3-27b-it",
    "gemma-3-27b-it-abliterated-normpreserve-v1",
]

# Compute duplicated Gemma rows once; duplicate them only when rendering group 2.
ALL_MODELS = list(dict.fromkeys(GENERAL_MODEL_ORDER_1 + GENERAL_MODEL_ORDER_2))

IDEOLOGY_ORDER = [
    "base",
    "libertarian_left",
    "libertarian_right",
    "authoritarian_left",
    "authoritarian_right",
    "centrism",
]

QUADRANT_IDEOLOGIES = [
    "libertarian_left",
    "libertarian_right",
    "authoritarian_left",
    "authoritarian_right",
]

REASONING_MODE_ORDER = ["short", "neutral", "think"]

MODEL_METADATA = {
    "gemma-3-1b-it": ("Gemma-3", "standard", 1.0),
    "gemma-3-4b-it": ("Gemma-3", "standard", 4.0),
    "gemma-3-12b-it": ("Gemma-3", "standard", 12.0),
    "gemma-3-27b-it": ("Gemma-3", "standard", 27.0),
    "gemma-3-27b-it-abliterated-normpreserve-v1": (
        "Gemma-3-abliterated",
        "standard",
        27.0,
    ),
    "Qwen3-4B_no_think": ("Qwen3", "no_think", 4.0),
    "Qwen3-8B_no_think": ("Qwen3", "no_think", 8.0),
    "Qwen3-14B_no_think": ("Qwen3", "no_think", 14.0),
    "Qwen3-32B_no_think": ("Qwen3", "no_think", 32.0),
    "Qwen3-4B_think": ("Qwen3", "think", 4.0),
    "Qwen3-8B_think": ("Qwen3", "think", 8.0),
    "Qwen3-14B_think": ("Qwen3", "think", 14.0),
    "Qwen3-32B_think": ("Qwen3", "think", 32.0),
}


def ensure_artifact_directories() -> None:
    for path in [
        ARTIFACT_ROOT,
        CORPUS_DIR,
        FEATURE_DIR,
        EMBEDDING_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        CASEBOOK_DIR,
        LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def model_metadata(model_variant: str) -> tuple[str, str, float]:
    if model_variant not in MODEL_METADATA:
        raise KeyError(f"Missing model metadata for {model_variant}")
    return MODEL_METADATA[model_variant]


def qwen_base_model(model_variant: str) -> str:
    for suffix in ("_no_think", "_think"):
        if model_variant.endswith(suffix):
            return model_variant[: -len(suffix)]
    return model_variant
