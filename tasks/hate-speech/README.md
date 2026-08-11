# Hate-speech experiment

This task evaluates binary hate-speech classification under the same six
persona conditions used elsewhere in the project. It is English-only, bf16,
and covers the eight primary Gemma 3 and Qwen3 checkpoints.

> **Sensitive-content warning:** the required corpus contains identity-targeted
> hate speech and offensive language. Treat raw text and generated artifacts as
> sensitive research data; do not display examples by default.

## Current release status

The executable runner, historical provenance runners, supplied 14,400-row
design, complete configuration aggregate, item-output converter, and
reproducible factor-sensitivity analysis are included. The colleague handoff
contains eight complete wide model CSVs: 1,800 configurations per model and
1,332 target-specific item predictions per configuration. Those large CSVs
remain outside Git and are converted to long Parquet for the private companion
Dataset repository.

The handoff does **not** contain the source statements, exact prompt JSON, raw
vocabulary logits, or assembled corpus file. The historical results contain
only candidate-normalized probabilities for literal `True` and `False`, gold
labels, target labels, and source metadata. These limitations are preserved in
the released schema rather than inferred or filled in. Because the historical
softmax values were computed/stored at bf16 precision, candidate-pair sums can
differ from one by at most 0.001953125 in the supplied files.

The historical anonymized package also says that its hate prompt JSON is
shipped, but the referenced file is actually a Political Compass prompt file:
its `instructions` field is split into `question_first`/`options_first`, whereas
the hate runner indexes a list of ten `{text}` templates. The exact hate prompt
file must therefore also be supplied or confirmed before the raw experiment is
called fully reproducible. The clean runner fails loudly on this mismatch.

Expected local files:

```text
tasks/hate-speech/data/
├── corpus/english.jsonl   # 13,320 target-specific rows: 1,332 per target
└── prompts/english.json   # five contexts, ten instructions, persona templates
```

The corpus schema is one JSON object per line with `text`, `hate`,
`target_groups`, `dataset`, and `grouping`. The first target group must be one
of the ten groups named in `run_hate_speech.py`. The completed run selected one
target per configuration and scored its 1,332 corresponding rows.

## Run

```bash
python tasks/hate-speech/run_hate_speech.py validate-inputs
python tasks/hate-speech/run_hate_speech.py generate-design
python tasks/hate-speech/run_hate_speech.py run \
  --model gemma-3-1b-it --gpus 0
```

Run one model per process/job. The release runner preserves the archived 8-D
Latin-hypercube schedule (including its three unused dimensions), plain-text
prompting, and exact last-token scoring of `True` and `False`. It writes long,
resumable JSONL with both selected-token logits and candidate-only softmax
probabilities. The untouched historical wide-CSV runner is retained at
`legacy/hate-speech/run_hate_speech_original.py`.

The completed historical CSVs can be normalized with:

```bash
python tasks/hate-speech/analysis/convert_historical_results.py \
  --raw /path/to/output_hate_speech/results_hs \
  --design /path/to/output_hate_speech/experimental_design_hs.csv \
  --output /tmp/hate-speech-parquet
```

The converter uses a one-based `item_index`. It does not expose the historical
header's `q...` prefixes as question IDs: the header came from the first
target-specific row, while subsequent targets were appended positionally.

## Data source

The archived code describes the corpus as target-specific HATE-IDENTITY data associated
with Yoder et al., *How Hate Speech Varies by Target Identity: A Computational
Analysis* (CoNLL 2022): <https://aclanthology.org/2022.conll-1.3/>. Before a
public data upload, verify the exact assembled split, its component-dataset
licenses, and the colleague-provided checksum. The initial Hugging Face release
therefore remains private.

## Analysis

The committed aggregate contains one row per model × persona × LHS
configuration: 8 × 6 × 300 = 14,400 rows. Each row stores mean `P(hate)` across
the items for its sampled target group and mean `p(1-p)` item dispersion.

```bash
cd tasks/hate-speech/analysis
python factor_sensitivity.py --figure
```

`target` is treated as a blocking variable, not a prompt factor. Prompt-factor
components are fitted within persona conditions; the persona-condition term is
tested separately in one model-level fit. These values describe score
sensitivity under this design and do not establish that a model family, size,
or political group has an intrinsic hate-speech tendency.

Regenerating the aggregate from the supplied eight CSVs reproduces all 14,400
rows and both stored metrics exactly (maximum absolute difference 0).
