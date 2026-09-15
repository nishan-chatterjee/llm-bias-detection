# IBM topic-sentiment experiment

This directory contains only the IBM **topic-sentiment** experiment used in the
paper. It does not contain the separate stance experiments or the discarded
target-taxonomy/LLM-annotation analysis.

## Source data

`data/questions/ibm-claim-stance/ibm_test_topics.csv` is a 30-row extract of
the unique test topics in
[`ibm-research/claim_stance`](https://huggingface.co/datasets/ibm-research/claim_stance),
pinned to revision `ec4e2c2ec3e0c70087c67a28a7bce58b682b8109`. The extract has
19 positive and 11 negative topic-sentiment labels. The upstream card credits
Wikipedia and IBM and states CC BY-SA 3.0 terms; retain the attribution and
cite Bar-Haim et al. (EACL 2017) when redistributing or using these rows. The
upstream card's YAML metadata currently says CC BY 3.0 while its prose says CC
BY-SA 3.0, so this release follows the more restrictive prose statement.

## Run

The runner uses the eight primary checkpoints in
[`models/serve/model_config.json`](../../models/serve/model_config.json). A
populated local `models/serve/<alias>/` takes precedence; otherwise the pinned
Hugging Face ID and revision are used.

The full four-GPU wrapper is:

```bash
bash tasks/sentiment/run_experiments.sh design
GPU_IDS=0,1,2,3 bash tasks/sentiment/run_experiments.sh run
```

The direct task entry point is equivalent:

```bash
GPU_IDS=0,1,2,3 bash tasks/sentiment/mcq.sh design
GPU_IDS=0,1,2,3 bash tasks/sentiment/mcq.sh run
```

Alternatively, `GPU_IDS=0,1,2,3 bash tasks/sentiment/run_experiments.sh all`
combines both steps. Set `DRY_RUN=1` to inspect the exact Python commands
without launching vLLM. The runner creates vLLM workers in-process; no separate
API server is required.

```bash
python tasks/sentiment/run_ibm_sentiment.py generate-design
python tasks/sentiment/run_ibm_sentiment.py run \
  --gpus 0,1,2,3 \
  --models all
```

For a small operational smoke test:

```bash
python tasks/sentiment/run_ibm_sentiment.py generate-design \
  --models gemma-3-1b-it --samples 1 --output /tmp/ibm-sentiment-smoke
python tasks/sentiment/run_ibm_sentiment.py run \
  --gpus 0 --models gemma-3-1b-it --output /tmp/ibm-sentiment-smoke
```

The full design has 300 prompt configurations for each of six persona
conditions and eight models: 14,400 configurations and 432,000 item rows. Each
row stores the two candidate log-probabilities and their candidate-only
softmax probabilities. There is no generated rationale or hidden
chain-of-thought field in this MCQ task.

### Preprocessing

The original claim-level test split is reduced deterministically to the 30
unique `(topicText, topicTarget, topicSentiment)` records used by the paper.
The 19 positive and 11 negative values become `POSITIVE`/`NEGATIVE`; no claim
text, LLM-authored target taxonomy, imputation, or text normalization is used.
The answer-key permutation is applied at prompt construction, and the two
selected next-token log probabilities are normalized over those candidates.

## Analysis

For the smallest notebook-only download:

```bash
python scripts/download_dataset.py --component analysis
python scripts/verify_setup.py --require-analysis-data
jupyter nbconvert --to notebook --execute --inplace \
  tasks/sentiment/analysis/notebooks/01_ibm_sentiment_core.ipynb
```

The notebook prints its resolved Hugging Face analysis-table path and builds
its visualizations directly from those tables.

Rebuild the annotation-free tables and figures from the runner JSONLs or the
downloaded release Parquets:

```bash
python tasks/sentiment/analysis/core.py \
  --raw data/release/data/ibm_sentiment
```

The core analysis reports:

- coverage and errors;
- accuracy and macro-F1 by model and assigned persona;
- configuration-level sensitivity to context, instruction template, answer-key
  type, answer-order permutation, and persona wording;
- descriptive per-topic accuracy and positive-prediction rates.

The complete executed notebook additionally shows configuration-level score
distributions, persona deltas from base, topic-local fragility, prediction
skew, and descriptive model/size summaries. It does not use the discarded
LLM-authored target taxonomy or any downstream taxonomy figure.

The sensitivity heatmap is descriptive, not a causal variance decomposition.
For a factor, its cell is the root-mean-square spread of configuration-level
macro-F1 between that factor's levels, calculated within persona conditions.
`Ideology SD` is the standard deviation of the six persona-condition means,
and `Residual RMSE` is the residual error from an additive dummy-coded model.

Across the archived complete run, all eight models have 54,000/54,000 valid
rows and no recorded errors. The base condition has the highest macro-F1 for
every model, but the size of the drop under assigned personas varies strongly
by model and persona. Instruction wording is the largest named prompt-factor
component for six of eight models; this is a descriptive robustness result,
not evidence that any persona or prompt wording causes an intrinsic model
property.
