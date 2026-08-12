# PoliLean

Code and data release for *Navigating the Digital Spectrum: Assessing
Political Bias, Moral Values, and Toxicity in LLMs*.

The release evaluates how political-persona prompts and prompt-format choices
affect open-weight language models on three tasks:

- Political Compass multiple choice and chat;
- IBM topic sentiment classification;
- identity-targeted hate-speech detection.

Large evaluation traces are hosted in the companion Hugging Face Dataset:
[nishan-chatterjee/polilean-evaluation-traces](https://huggingface.co/datasets/nishan-chatterjee/polilean-evaluation-traces).
The companion repository currently requires access permission.

## Models

The primary model set contains four instruction-tuned Gemma 3 checkpoints and
four standard Qwen3 checkpoints:

| Family | Released checkpoints | Upstream collection |
|---|---|---|
| Gemma 3 IT | 1B, 4B, 12B, 27B | [Gemma 3 release](https://huggingface.co/collections/google/gemma-3-release) |
| Qwen3 | 4B, 8B, 14B, 32B | [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3) |

For example, the largest Qwen checkpoint is
[Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B). Qwen chat evaluation
includes matched thinking and non-thinking protocols. A secondary matched
Gemma 27B diagnostic uses
[YanLabs/gemma-3-27b-it-abliterated-normpreserve-v1](https://huggingface.co/YanLabs/gemma-3-27b-it-abliterated-normpreserve-v1).

Exact repository IDs, pinned revisions, and the temperature/top-p/top-k/token
settings used in chat are recorded in
[`models/serve/model_config.json`](models/serve/model_config.json). Model
weights are not redistributed.

## Repository map

```text
models/serve/                 Checkpoint IDs, revisions, and decoding settings
tasks/political-compass/      MCQ/chat runners, scorer reconstruction, and analyses
tasks/sentiment/              IBM topic-sentiment runner and analyses
tasks/hate-speech/            Hate-speech runner, converter, and analyses
dataset/                      Dataset packaging, schema validation, and card
tests/                        Deterministic release and schema tests
legacy/                       Original research notebooks/scripts with provenance notes
```

The canonical executed analysis notebooks are:

- [`01_mcq_analysis_primary_models.ipynb`](tasks/political-compass/analysis/notebooks/01_mcq_analysis_primary_models.ipynb)
- [`02_chat_analysis_primary_models.ipynb`](tasks/political-compass/analysis/notebooks/02_chat_analysis_primary_models.ipynb)
- [`03_mcq_chat_comparison_primary_models.ipynb`](tasks/political-compass/analysis/notebooks/03_mcq_chat_comparison_primary_models.ipynb)
- [`01_ibm_sentiment_core.ipynb`](tasks/sentiment/analysis/notebooks/01_ibm_sentiment_core.ipynb)
- [`01_hate_speech_analysis_primary_models.ipynb`](tasks/hate-speech/analysis/notebooks/01_hate_speech_analysis_primary_models.ipynb)

These notebooks contain the complete release-facing coverage checks,
distributions, factor-sensitivity diagnostics, matched protocol comparisons,
item/topic analyses, and limitations. The original output-rich research
notebooks are retained under `legacy/`; they may depend on extra models,
withheld scorer parameters, discarded annotations, or the separate
offensive-speech experiment and are therefore not the canonical runnable
analysis.

The Political Compass qualitative-analysis directory contains the localized
chat diagnostics used in the paper. Predefined cue matches are reported as
qualifying-language diagnostics rather than human-validated hedging labels.
Experimental multi-LLM judge annotations are not used as headline evidence.

## Data

The Hugging Face release has six configurations:

| Configuration | Contents |
|---|---|
| `political_compass_mcq` | 8 models × bf16/8-bit/4-bit MCQ outputs |
| `political_compass_chat` | 4 Gemma chat and 8 Qwen think/no-think traces |
| `political_compass_chat_ablation` | matched Gemma 27B abliterated diagnostic |
| `ibm_sentiment` | 432,000 topic-sentiment predictions |
| `hate_speech` | 19,180,800 item predictions in 8 Parquets |
| `hate_speech_aggregate` | 14,400 configuration summaries and sensitivity table |

The Political Compass chat traces include prompts, visible Stage-1 answers,
token counts, finish reasons, and Stage-2 candidate scores. Stage 2 scores the
answer candidates; it does not generate another rationale. The IBM task is
multiple choice and contains no generated rationale. Historical MCQ and hate
files contain candidate-normalized probabilities, not complete vocabulary
logits.

The hate-speech handoff does not include source statements, the exact prompt
JSON, or the assembled corpus. `item_index` is therefore a stable position
within the selected target subset and must not be interpreted as a global
question identifier.

## Reproducing the analyses

Create the analysis environment:

```bash
conda env create -f environment-analysis.yml
conda activate polilean-analysis
pytest
```

Download the companion Dataset into a local directory, or use the compact
derived tables already tracked for CPU-only notebook reproduction. Execute a
notebook from the repository root, for example:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  tasks/political-compass/analysis/notebooks/01_mcq_analysis_primary_models.ipynb
```

Task-specific READMEs document full inference commands and data placement.
`environment-inference.yml` is intended for GPU generation; inference runners
resolve either a local `models/serve/<alias>/` checkpoint or the pinned
Hugging Face repository.

Political Compass coordinates require the locally reconstructed scorer. The
repository includes the reconstruction notebook and verification code, but
does not redistribute the generated `.npz` parameters. The released derived
coordinates allow the analysis notebooks to run without that file.

## Responsible use and known limitations

The release contains political claims and metadata concerning death,
discrimination, race, religion, disability, sexuality, violence, and hate
speech. Their presence is part of the evaluation and is not an endorsement.

- Assigned personas are experimental prompt conditions, not intrinsic model
  identities.
- Factor-sensitivity magnitudes describe variation under the sampled design;
  they are not causal effects.
- Four checkpoints per family do not establish general model-size laws.
- The IBM analysis excludes the discarded LLM-authored target taxonomy.
- Hate-speech source-text redistribution and component licences require final
  verification before the Dataset becomes public.
- Political Compass propositions remain subject to their upstream terms. The
  scorer parameters are not redistributed.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for data/model
attribution and rights notes.

## Licence

Code is currently released under MIT, subject to final joint-author review.
Datasets, model outputs, and upstream models retain their own licences and
terms.
