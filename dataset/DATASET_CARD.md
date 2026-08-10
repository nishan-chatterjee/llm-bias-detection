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
- config_name: hate_speech_aggregate
  data_files: data/hate_speech_aggregate/*.csv
---

# PoliLean evaluation traces

This private-first dataset accompanies the PoliLean code release. It contains
the primary-model Political Compass MCQ and chat traces, IBM topic-sentiment
traces, and the currently available hate-speech configuration aggregate.

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
- Hate-speech raw text/results and the exact prompt file are pending a colleague
  handoff and component-license verification. The current aggregate contains
  no source text but does name identity target groups.
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
- `hate_speech_aggregate`: 14,400 configuration rows and factor-sensitivity
  values. Raw item-level outputs will be added after handoff and validation.

The `metadata/manifest.json` file records SHA-256 checksums and byte sizes.

## Sensitive content

Political propositions and hate-speech data may contain references to death,
race, religion, disability, segregation, violence, and other sensitive
subjects. Downstream tools should not display arbitrary examples by default.
