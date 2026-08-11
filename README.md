# PoliLean evaluation release

This branch is the reproducibility release for the Political Compass, IBM
sentiment, and hate-speech experiments reported in *Navigating the Digital
Spectrum: Assessing Political Bias, Moral Values, and Toxicity in LLMs*.

The release is deliberately separated into two parts:

- **This Git repository** contains redistributable prompt/input resources, runnable
  experiment code, instrument-reconstruction code, analysis notebooks and
  scripts, validation tests, and compact manifests.
- **A companion Hugging Face Dataset repository** contains the large model
  outputs. It is created as a private dataset first and is made public only
  after the final joint review.

The repository does **not** contain model weights, local model caches, the
generated Political Compass scorer (`.npz`), manuscript sources, stance
experiments, or LLM-generated research annotations. Political Compass
proposition files are also excluded from Git pending redistribution clearance.
Hate-speech item predictions from the collaborator handoff are packaged in the
private companion dataset. The exact prompt file, source statements, assembled
corpus checksum, and component-licence review remain outstanding.

## Layout

```text
models/serve/                 Model aliases, checkpoint revisions, and decoding settings
tasks/political-compass/      Instrument, MCQ/chat runners, analyses, and trace diagnostics
tasks/sentiment/              IBM two-way sentiment runner and analysis
tasks/hate-speech/            Hate-speech runner, analysis, and pending-data contract
dataset/                      Schema checks and Hugging Face packaging/upload tools
tests/                        Fast deterministic and schema tests
legacy/                       Historical source retained only for provenance
```

Each task has its own README with the exact run order. The short version is:

1. Create `environment-analysis.yml` for notebooks/validation and
   `environment-inference.yml` for GPU runners.
2. Download or mount the desired model checkpoints, or let Transformers/vLLM
   resolve the Hugging Face IDs in `models/serve/model_config.json`.
3. For Political Compass scoring, run the reconstruction notebook locally.
   This creates an ignored scorer file; it is not redistributed.
4. Run the task experiment scripts.
5. Download the companion dataset for analysis, or point the notebooks at an
   equivalent local result directory.
6. Run `python -m pytest` and `python dataset/validate_release.py ...` before
   using or extending the data.

## Models

The primary release covers four instruction-tuned Gemma 3 checkpoints and four
Qwen3 checkpoints. Qwen chat traces include both thinking and non-thinking
protocols. Exact repository IDs, snapshot revisions, aliases, maximum token
budgets, and decoding parameters are recorded in
`models/serve/model_config.json`. No weights are copied into this repository.

## Data and sensitive content

The released outputs preserve model answers needed for reproducibility. They
therefore contain political claims and, for some tasks, references to death,
discrimination, hate speech, disability, race, religion, sexuality, and other
sensitive subjects. These texts are research stimuli or model outputs; their
presence is not an endorsement.

Political Compass prompt templates are included, but the proposition files are
not tracked in Git. They may be present only in the initial private dataset
under `restricted_inputs/` while explicit redistribution clearance is sought.
The learned scoring parameters are reconstructed locally and are not
redistributed. Users remain responsible for the original Political Compass
terms and all upstream model licences.

The IBM sentiment task includes its 30-row test-topic extract with upstream
attribution. Hate-speech prediction records are bundled privately without
source text; the exact assembled corpus is not bundled until its checksum and
component licences have been confirmed.

## Scope of the qualitative diagnostics

The qualitative-analysis directory provides reproducible, primarily
rule-based exploratory diagnostics over the primary chat models. Cue matches
are measurements of predefined strings, not human-validated labels for
hedging, moral distancing, or other latent intentions. The release excludes
the experimental multi-LLM judge annotations because agreement was not strong
enough to support headline claims.

## Licence

Code is currently provided under the MIT License, subject to confirmation in
the final joint-author review. Data and model outputs may also be governed by
the licences or terms of their upstream sources and models; see
`THIRD_PARTY_NOTICES.md` and the companion dataset card.
