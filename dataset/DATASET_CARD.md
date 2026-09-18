---
license: cc-by-4.0
language:
- en
- bg
- cs
- de
- es
- fa
- fr
- it
- pl
- pt
- ro
- ru
- sl
- tr
task_categories:
- text-classification
pretty_name: LLM Bias Detection Evaluation Traces
configs:
- config_name: political_compass_mcq
  data_files: data/political_compass_mcq/*.parquet
- config_name: political_compass_chat_numeric
  data_files: data/political_compass_chat_numeric/*.parquet
- config_name: political_compass_chat_ablation_numeric
  data_files: data/political_compass_chat_ablation_numeric/*.parquet
- config_name: ibm_sentiment
  data_files: data/ibm_sentiment/*.parquet
- config_name: hate_speech
  data_files: data/hate_speech/*.parquet
- config_name: hate_speech_inputs
  data_files: data/hate_speech_inputs/english.parquet
- config_name: hate_speech_aggregate
  data_files: data/analysis_ready/hate_speech/hs_configuration_scores.parquet
- config_name: hate_speech_factor_sensitivity
  data_files: data/analysis_ready/hate_speech/hs_factor_sensitivity.parquet
- config_name: analysis_pct_mcq_configurations
  data_files: data/analysis_ready/political_compass/pct_configuration_scores.parquet
- config_name: analysis_pct_chat_configurations
  data_files: data/analysis_ready/political_compass/chat_configuration_scores.parquet
- config_name: analysis_pct_stage_agreement
  data_files: data/analysis_ready/political_compass/chat_stage_agreement_summary.parquet
- config_name: analysis_pct_factor_sensitivity
  data_files: data/analysis_ready/political_compass/pct_factor_sensitivity.parquet
- config_name: analysis_ibm_coverage
  data_files: data/analysis_ready/sentiment/coverage_by_model.parquet
- config_name: analysis_ibm_config_metrics
  data_files: data/analysis_ready/sentiment/config_level_metrics.parquet
- config_name: analysis_ibm_model_metrics
  data_files: data/analysis_ready/sentiment/summary_by_model.parquet
- config_name: analysis_ibm_persona_metrics
  data_files: data/analysis_ready/sentiment/summary_by_model_persona.parquet
- config_name: analysis_ibm_factor_sensitivity
  data_files: data/analysis_ready/sentiment/factor_sensitivity_macro_f1.parquet
- config_name: analysis_ibm_topic_metrics
  data_files: data/analysis_ready/sentiment/topic_descriptive_metrics.parquet
- config_name: analysis_hate_configurations
  data_files: data/analysis_ready/hate_speech/hs_configuration_scores.parquet
- config_name: analysis_hate_factor_sensitivity
  data_files: data/analysis_ready/hate_speech/hs_factor_sensitivity.parquet
- config_name: analysis_hate_item_metrics
  data_files: data/analysis_ready/hate_speech/hs_item_metrics_by_persona_target.parquet
- config_name: analysis_hate_calibration
  data_files: data/analysis_ready/hate_speech/hs_calibration_by_model.parquet
---

# LLM Bias Detection Evaluation Traces

Evaluation data accompanying *Navigating the digital spectrum: Assessing
political bias, stability, and downstream fairness in Large Language Models*
([arXiv:2609.08637](https://arxiv.org/abs/2609.08637)).

**Licence scope:** CC BY 4.0 covers the authors' original documentation,
templates, selection/arrangement and author-generated tables. It does not
relicense source text or annotations. IBM retains CC BY-SA 3.0; hate-corpus
components retain CC BY 4.0, CC0 or MIT as documented in
`inputs/hate_speech/README.md` and `THIRD_PARTY_NOTICES.md`. Political Compass
propositions are expressly excluded from the CC grant pending rights review.
See `LICENSE-CC-BY-4.0.md` for scope and licence links. The repository-wide HF
badge cannot represent each component's terms; the notices remain binding.

The Dataset contains numerical outputs for Political Compass multiple
choice and chat experiments, IBM topic sentiment classification, and
identity-targeted hate-speech detection. Reproduction and analysis code is in
the [LLM Bias Detection GitHub repository](https://github.com/nishan-chatterjee/llm-bias-detection).

## Models and evaluation conditions

The primary model set is:

- instruction-tuned Gemma 3 1B, 4B, 12B and 27B from the
  [Gemma 3 release](https://huggingface.co/collections/google/gemma-3-release);
- Qwen3 4B, 8B, 14B and 32B from the
  [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3), including
  matched thinking and non-thinking chat protocols.

The optional ablation configuration uses
[YanLabs/gemma-3-27b-it-abliterated-normpreserve-v1](https://huggingface.co/YanLabs/gemma-3-27b-it-abliterated-normpreserve-v1).
Pinned repository revisions and decoding parameters are provided in
`metadata/model_config.json`. No model weights are included.

## Dataset configurations

### `political_compass_mcq`

Twenty-four Parquet files: eight primary models evaluated in bf16, 8-bit and
4-bit conditions. Each file has 1,800 sampled prompt configurations covering
six assigned-persona conditions. The experiment spans fourteen languages.

Rows contain experimental factors and the four candidate probabilities for
each of 62 propositions. Historical files did not record complete vocabulary
logits. The learned Political Compass scorer is not included; released
configuration-level coordinates and reconstruction code are available in the
GitHub repository.

### `political_compass_chat_numeric`

Twelve Parquet files: four Gemma chat variants and four Qwen checkpoints in
think/no-think modes. Each variant has 111,600 question rows: 1,800 prompt
configurations × 62 propositions.

Rows retain IDs, experimental factors, token counts, finish reasons,
candidate log-probabilities and candidate probabilities. An explicit allow-list
projection omits statement, premise, prompt, Stage-1 response, classification
suffix, candidate wording, item metadata and free-text errors; schema metadata
is stripped. `has_error` replaces the original error text. Every retained column
is copied unchanged from the raw files. These are numerical projections, not
the full chat traces. Stage 2 scores answer candidates and produces no rationale.

### `political_compass_chat_ablation_numeric`

A separate matched Gemma 3 27B abliterated-checkpoint diagnostic. It is not
part of the eight-model primary comparison. The same text-free projection applies.

### `ibm_sentiment`

Eight Parquet files with 54,000 rows each: 1,800 prompt configurations × 30
unique IBM topic-sentiment items. Rows include the topic, target phrase, gold
positive/negative label, model prediction, correctness, candidate scores,
assigned persona and prompt factors. This MCQ task has no generated rationale.

The release analysis uses gold labels and experiment fields only. It does not
include the discarded LLM-authored target taxonomy or downstream annotation
figures.

### `hate_speech`

Eight Parquet files with 2,397,600 rows per model (19,180,800 total): 1,800
prompt configurations × 1,332 target-specific item positions. Fields include
assigned persona, sampled identity target, gold hate label, `p_hate`,
`p_not_hate`, thresholded prediction and source metadata.

The historical output Parquets do not embed source statements or raw vocabulary
logits. Its wide CSV header
was written from the first target-specific list and later target rows were
appended positionally. Consequently, `item_index` is one-based within the
selected target subset and is intentionally not called a global question ID.
Candidate probabilities were stored at bf16 precision and may sum to one within
approximately 0.002 rather than exactly.

### `hate_speech_inputs`

The subsequently supplied original JSONL and prompt JSON are now available at
`inputs/hate_speech/corpus/english.jsonl` and
`inputs/hate_speech/prompts/english.json`. The Parquet input configuration has
13,320 records: ten targets × two classes × 666 items. Original file order,
source fields and the existing `fold` field are preserved. Added `question_id`
is the one-based input line; `target` is the first assigned target group and
`item_index` is its one-based within-target position. Join historical predictions
to these inputs on `(target, item_index)`, not on a global index.

All 19,180,800 archived gold-label/source-metadata rows match this input order.
That does not prove historical text-byte identity: original outputs have no
text/prompt hashes. Input checksums and licence attribution are included in
`inputs/hate_speech/README.md`. No old predictions were regenerated or changed.

### `hate_speech_aggregate`

The 14,400-row configuration aggregate used by the CPU analysis. It reproduces
exactly from the eight released hate-speech Parquets. The eight-row historical
factor table is exposed separately as `hate_speech_factor_sensitivity` because
the two tables have different schemas.

### Analysis-ready configurations

The fourteen `analysis_pct_*`, `analysis_ibm_*`, and `analysis_hate_*`
configurations contain the compact, derived tables consumed by the canonical
executed notebooks. Each configuration contains one table so that it has a
single coherent schema. They allow the analyses and figures to run from
the pinned Dataset snapshot without model weights. Political Compass
configuration coordinates are included here because rebuilding them from raw
candidate probabilities requires the locally reconstructed scorer, whose
generated parameters are not redistributed. `analysis_pct_stage_agreement`
includes a 12-row summary of explicit Stage-1 stance versus
Stage-2 candidate agreement; its counts were generated from all primary chat
traces with the released conservative parser.

## Loading the data

With `datasets`:

```python
from datasets import load_dataset

chat = load_dataset(
    "nishan-chatterjee/llm-bias-detection",
    "political_compass_chat_numeric",
)
```

For large configurations, select a file or use streaming:

```python
hate = load_dataset(
    "nishan-chatterjee/llm-bias-detection",
    "hate_speech",
    streaming=True,
)
```

Individual Parquets can also be read directly with pandas or PyArrow. The
`metadata/manifest.json` file records byte sizes, row counts where applicable,
and SHA-256 checksums for release files.

For example, the small derived chat-coordinate table can be loaded without
the proposition files or scorer:

```python
chat_coordinates = load_dataset(
    "nishan-chatterjee/llm-bias-detection",
    "analysis_pct_chat_configurations",
    split="train",
)
```

## Experimental metadata

The `metadata/` directory contains:

- model IDs, revisions and chat decoding parameters;
- experimental designs for Political Compass, IBM sentiment and hate speech;
- conversion provenance for the historical hate-speech files;
- a release manifest with checksums.

Prompt templates and the IBM 30-topic extract are under `inputs/`. The
`restricted_inputs/` directory containing Political Compass questionnaires was
removed from the current public tree on 2026-09-18. Its former name did not
restrict access. The 13 raw chat/ablation Parquets are also withheld and replaced
by text-free numerical projections. After these removals, main history was
squashed using the Hub's history-cleanup operation. Existing downloads, forks
or host caches are outside this operation; no universal erasure is claimed.
Do not redistribute the propositions without permission from the rights holder.

Raw originals are retained privately for audit. Public numerical outputs and
`analysis_pct_*` tables support aggregate reproduction, but text-dependent
qualitative diagnostics and reconstruction of the withheld scorer require
authorized local material. This limitation is explicit; public files do not
provide an out-of-the-box fresh-inference run for Political Compass.

## Intended use

The Dataset supports reproduction of the paper's aggregate results, evaluation
of prompt/persona sensitivity, matched Qwen think/no-think comparisons,
question- or topic-level diagnostics, calibration checks, and method development
for robust behavioral evaluation.

Assigned personas are prompt conditions. They must not be described as a
model's inherent political identity. Factor-sensitivity components describe
variation within this experimental design and are not causal estimates. The
four checkpoints per family are insufficient for universal scaling claims.

## Sensitive content and responsible use

Political prompts and model responses may discuss death, punishment, race,
religion, disability, segregation, sexuality, violence and other sensitive
subjects. Hate-speech inputs contain the source statements and identify target
groups. These materials are evaluation stimuli or model
outputs; their inclusion is not endorsement.

Do not use the Dataset to profile people, infer an individual's politics, or
rank protected groups. Applications should avoid displaying arbitrary
sensitive examples by default and should preserve the distinction between
assigned persona, model output and gold dataset labels.

## Source and licence notes

- Political Compass propositions and scoring remain subject to the upstream
  Political Compass terms. The generated scorer parameters are not included.
- The IBM 30-topic extract comes from `ibm-research/claim_stance`, pinned at
  revision `ec4e2c2ec3e0c70087c67a28a7bce58b682b8109`. Cite Bar-Haim et al.
  (EACL 2017). The upstream card's prose states CC BY-SA 3.0.
- Hate-speech inputs are associated with Yoder et al. (CoNLL 2022). The
  supplied corpus is included separately with source attribution and its
  component licence notices. Archived logits and prompt/text hashes remain
  unavailable.
- Model outputs may remain subject to the upstream model licences and terms.

The HF metadata now displays `cc-by-4.0` for the authors' original contributions.
It is not a blanket licence over all borrowed material. See the licence scope
at the top, `LICENSE-CC-BY-4.0.md`, and `THIRD_PARTY_NOTICES.md`.

## Cite the preprint

```bibtex
@misc{debevc2026navigatingdigitalspectrumassessing,
  title={Navigating the digital spectrum: Assessing political bias, stability, and downstream fairness in Large Language Models},
  author={Luka Debevc and Nishan Chatterjee and Antoine Doucet and Senja Pollak and Matej Martinc},
  year={2026},
  eprint={2609.08637},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2609.08637},
}
```
