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
remain outside Git and are converted to long Parquet for the public companion
Dataset repository.

The original handoff did not contain source text or prompts. The subsequently
supplied `english.jsonl` and `english.json` are now installed under `data/`,
with input checksums and component notices in `data/README.md`, and published
in the companion HF input layer. Their within-target ordering, gold labels and
source metadata match all 19,180,800 archived rows across eight checkpoints.
Historical outputs contain no text/prompt hashes, so this is metadata alignment,
not cryptographic proof of historical text identity. The historical results contain
only candidate-normalized probabilities for literal `True` and `False`, gold
labels, target labels, and source metadata. These limitations are preserved in
the released schema rather than inferred or filled in. Because the historical
softmax values were computed/stored at bf16 precision, candidate-pair sums can
differ from one by at most 0.001953125 in the supplied files.

The historical anonymized package also says that its hate prompt JSON is
shipped, but the referenced file is actually a Political Compass prompt file:
its `instructions` field is split into `question_first`/`options_first`, whereas
the hate runner indexes a list of ten `{text}` templates. The newly supplied
file has the correct list schema and passes preflight; it is not that mismatched
Political Compass file.

Expected local files:

```text
tasks/hate-speech/data/
├── corpus/english.jsonl   # 13,320 target-specific rows: 1,332 per target
└── prompts/english.json   # five contexts, ten instructions, persona templates
```

This uses the same task-local `data/` organisation as Political Compass and
sentiment, with prompt templates separated from the dataset. We deliberately
use `corpus/` for the corpus rather than the historical runner's `hate_speech/`
subfolder. The Python runner resolves both defaults from its own file location,
and the shell wrapper passes absolute paths; no particular working directory
is required. The preserved original runner under `legacy/` retains its original
relative paths and is not the recommended release entry point.

The corpus schema is one JSON object per line with `text`, `hate`,
`target_groups`, `dataset`, and `grouping`. The first target group must be one
of the ten groups named in `run_hate_speech.py`. The completed run selected one
target per configuration and scored its 1,332 corresponding rows.

## Run

The release wrapper defaults to four GPUs, preserves the historical model-block
order used to assign configuration IDs, validates both included inputs, and
runs the eight checkpoints sequentially with resumable outputs:

```bash
GPU_IDS=0,1,2,3 bash tasks/hate-speech/mcq.sh design
GPU_IDS=0,1,2,3 bash tasks/hate-speech/mcq.sh preflight
GPU_IDS=0,1,2,3 bash tasks/hate-speech/mcq.sh run
```

`GPU_IDS=0,1,2,3 bash tasks/hate-speech/run_experiments.sh all` combines those
steps. `DRY_RUN=1` prints the exact Python invocations. `preflight` now passes
from a clean checkout because both input files are included. The input corpus
and prompt JSON can also be obtained from HF with:

```bash
python scripts/download_dataset.py --component hate-inputs --install-hate-inputs
```

Installation checks file hashes and refuses to overwrite different local files.

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

The inference runner uses Transformers directly rather than an HTTP server so
it can retain the selected candidate logits. No separately launched vLLM or
llama.cpp service is part of this experiment.

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

The complete release notebook is
`analysis/notebooks/01_hate_speech_analysis_primary_models.ipynb`. Generate its
compact item-level metrics from the companion Dataset Parquets with:

```bash
python tasks/hate-speech/analysis/summarize_item_predictions.py \
  --input /path/to/hf-download/data/hate_speech \
  --output tasks/hate-speech/analysis/data
```

For notebook reproduction without downloading all 19.2 million item rows:

```bash
python scripts/download_dataset.py --component analysis
python scripts/verify_setup.py --require-analysis-data
jupyter nbconvert --to notebook --execute --inplace \
  tasks/hate-speech/analysis/notebooks/01_hate_speech_analysis_primary_models.ipynb
```

The notebook reads the compact Hugging Face analysis configuration and
regenerates its visualizations in place.

The historical notebook combined hate speech with a separate offensive-speech
experiment. It is preserved under `legacy/hate-speech/` with its rendered
outputs, but the offensive-speech cells are not canonical release evidence.

## Data source

The archived code describes the corpus as target-specific HATE-IDENTITY data associated
with Yoder et al., *How Hate Speech Varies by Target Identity: A Computational
Analysis* (CoNLL 2022): <https://aclanthology.org/2022.conll-1.3/>. Before a
public data upload, verify the exact assembled split, its component-dataset
licenses, and the colleague-provided checksum. The public Hugging Face release
therefore contains the item-level predictions but not the source statements or
exact prompt file.

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

## Preprocessing summary

The historical schedule selects one of ten identity targets. Its corresponding
1,332 items are prompted without text normalization and scored using the final
token logits for literal `True` and `False`. The collaborator CSV handoff was
converted to long Parquet by configuration and one-based within-target item
position. The conversion preserves supplied labels, source metadata, targets,
and bf16-rounded probabilities. Source text is supplied separately and joined
by `(target, item_index)`; original raw vocabulary logits and text/prompt hashes
were not recorded and cannot be reconstructed.
