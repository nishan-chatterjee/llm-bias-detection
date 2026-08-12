# Political Compass analysis

The release keeps the large per-item outputs in the companion Hugging Face
dataset and includes compact configuration-level tables here:

- `data/pct_configuration_scores.csv`: 43,200 MCQ configurations across eight
  primary models and three quantizations.
- `data/chat_configuration_scores.parquet`: 21,600 English chat configurations
  across four Gemma variants and eight Qwen protocol variants.

The chat table contains derived coordinates and diagnostics, not generated
answer text. The raw JSONL files remain necessary for trace-level qualitative
analysis.

`factor_sensitivity.py` reproduces the MCQ factor-sensitivity table and optional
heatmap from the compact CSV. Run:

```bash
cd tasks/political-compass/analysis
python factor_sensitivity.py --figure
```

The three notebooks are complete primary-model release analyses. They include
coverage and design checks, configuration distributions, persona centroids,
factor sensitivity, protocol shifts, and explicit interpretation boundaries:

1. `01_mcq_analysis_primary_models.ipynb`
2. `02_chat_analysis_primary_models.ipynb`
3. `03_mcq_chat_comparison_primary_models.ipynb`

They intentionally exclude GaMS, Qwen 0.6B/1.7B, and stale post-hoc
temperature transformations that are not part of the paper's primary model
experiment. The original output-rich notebooks are retained under `legacy/`
for provenance, including their rendered outputs.

For a combined figure, the presentation order is Gemma MCQ by increasing size,
Gemma chat by increasing size, Qwen MCQ by increasing size, Qwen no-think chat
by increasing size, and Qwen think chat by increasing size.

To regenerate the configuration tables from raw outputs, first reconstruct the
local scorer described in `../instrument/README.md`, then use the dataset
derivation tool documented in the repository-level `dataset/README.md`.
