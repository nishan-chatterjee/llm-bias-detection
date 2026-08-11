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
pretty_name: PoliLean Evaluation Traces
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

# PoliLean evaluation traces

This private-first dataset accompanies the PoliLean code release. It contains
the primary-model Political Compass MCQ and chat traces, IBM topic-sentiment
traces, hate-speech item predictions, and a hate-speech configuration
aggregate.

## Visibility and rights status

The repository is intentionally private at initial upload.

- The Political Compass website asserts copyright and restricts unauthorized
  adoption/adaptation. Proposition files under `restricted_inputs/` must not be
  made public until explicit redistribution clearance is documented. The
  reconstructed scorer and its `.npz` parameters are not included.
- The 30 IBM test topics come from `ibm-research/claim_stance`, revision
  `ec4e2c2ec3e0c70087c67a28a7bce58b682b8109`. The upstream card credits
  Wikipedia and IBM and states CC BY-SA 3.0 in its licensing prose. Cite
  Bar-Haim et al. (EACL 2017).
- The hate-speech handoff supplies eight complete primary-model prediction
  files and the exact experimental design. It does not supply source statements,
  the exact prompt JSON, or the assembled corpus file. Component-dataset
  licences therefore still require verification before public release.
- Model weights are never included. `metadata/model_config.json` records pinned
  Hugging Face checkpoint IDs, revisions, and chat decoding parameters.

Because the components have different or unresolved terms, the combined
repository uses `license: other`; each component retains its own notices.

## Contents

- `political_compass_mcq`: eight primary models × bf16/8-bit/4-bit. Historical
  MCQ files store candidate-only probabilities; raw logits were not recorded.
- `political_compass_chat`: four Gemma chat variants and four Qwen checkpoints
  in think/no-think modes. Rows include prompts, Stage-1 visible text, token
  counts, finish reasons, candidate log-probabilities, and probabilities. There
  is no generated Stage-2 rationale: Stage 2 scores the answer candidates.
- `political_compass_chat_ablation`: optional matched Gemma 27B abliterated
  checkpoint, kept separate from the primary experiment.
- `ibm_sentiment`: eight primary-model MCQ outputs with candidate
  log-probabilities and probabilities. This task has no generated rationale.
- `hate_speech`: eight primary-model Parquets, each containing 1,800 prompt
  configurations × 1,332 target-specific item positions (2,397,600 rows per
  model; 19,180,800 total). The historical files store candidate-only
  probabilities for literal `True` and `False`; they do not store raw
  vocabulary logits or source text. `item_index` is the stable one-based
  position within the selected target's item list. It is intentionally not
  presented as a global question ID because the historical wide CSV header was
  written from the first target-specific row and later rows were appended
  positionally.
- `hate_speech_aggregate`: 14,400 configuration rows and factor-sensitivity
  values. It reproduces exactly from the included item predictions.

The `metadata/manifest.json` file records SHA-256 checksums and byte sizes.

## Sensitive content

Political propositions and hate-speech metadata may refer to death, race,
religion, disability, segregation, violence, and other sensitive subjects.
The hate-speech Parquets do not contain the source statements, but downstream
tools should still avoid displaying arbitrary sensitive records by default.
