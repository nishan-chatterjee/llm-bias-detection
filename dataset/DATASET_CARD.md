---
license: other
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
- config_name: political_compass_chat
  data_files: data/political_compass_chat/*.parquet
- config_name: political_compass_chat_ablation
  data_files: data/political_compass_chat_ablation/*.parquet
- config_name: ibm_sentiment
  data_files: data/ibm_sentiment/*.parquet
- config_name: hate_speech
  data_files: data/hate_speech/*.parquet
- config_name: hate_speech_aggregate
  data_files: data/hate_speech_aggregate/*.csv
---

# LLM Bias Detection Evaluation Traces

Evaluation data accompanying *Navigating the Digital Spectrum: Assessing
Political Bias, Moral Values, and Toxicity in LLMs*.

The Dataset contains the model outputs used for Political Compass multiple
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

### `political_compass_chat`

Twelve Parquet files: four Gemma chat variants and four Qwen checkpoints in
think/no-think modes. Each variant has 111,600 question rows: 1,800 prompt
configurations × 62 propositions.

Rows include the prompt, visible Stage-1 response, token count, finish reason,
candidate log-probabilities and candidate probabilities. Stage 2 scores the
answer candidates and does not produce a second rationale. Qwen thinking traces
contain the visible model output returned by the experiment; they should not be
treated as privileged hidden reasoning.

### `political_compass_chat_ablation`

A separate matched Gemma 3 27B abliterated-checkpoint diagnostic. It is not
part of the eight-model primary comparison.

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

The historical handoff does not contain source statements, the exact prompt
JSON, raw vocabulary logits, or the assembled corpus file. Its wide CSV header
was written from the first target-specific list and later target rows were
appended positionally. Consequently, `item_index` is one-based within the
selected target subset and is intentionally not called a global question ID.
Candidate probabilities were stored at bf16 precision and may sum to one within
approximately 0.002 rather than exactly.

### `hate_speech_aggregate`

The 14,400-row configuration aggregate and the factor-sensitivity table used
by the CPU analysis. The aggregate reproduces exactly from the eight released
hate-speech Parquets.

## Loading the data

With `datasets`:

```python
from datasets import load_dataset

chat = load_dataset(
    "nishan-chatterjee/llm-bias-detection",
    "political_compass_chat",
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

## Experimental metadata

The `metadata/` directory contains:

- model IDs, revisions and chat decoding parameters;
- experimental designs for Political Compass, IBM sentiment and hate speech;
- conversion provenance for the historical hate-speech files;
- a release manifest with checksums.

Prompt templates and the IBM 30-topic extract are under `inputs/`. Political
Compass proposition files are currently kept under `restricted_inputs/` while
redistribution terms are resolved; they must not be mirrored independently.

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
subjects. Hate-speech metadata identifies target groups, although source
statements are not released. These materials are evaluation stimuli or model
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
- The hate-speech source documentation associates the material with Yoder et
  al. (CoNLL 2022). The public release excludes its source statements; the
  exact assembled corpus and component licences remain limitations requiring
  final verification.
- Model outputs may remain subject to the upstream model licences and terms.

Because components have different or unresolved terms, the combined Dataset
uses `license: other`. See the GitHub repository's `THIRD_PARTY_NOTICES.md` for
the complete release notes.
