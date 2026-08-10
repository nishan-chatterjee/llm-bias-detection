# Model resolution

No model files are committed to this repository. An experiment runner resolves
an alias in this order:

1. a populated local directory at `models/serve/<alias>`;
2. the pinned Hugging Face `repo_id` and `revision` in `model_config.json`.

The eight entries marked `primary: true` are the paper’s primary models and are
the default experiment set. The abliterated Gemma entry is retained only for
the named secondary matched-checkpoint diagnostic and is never selected by
`all` or by a default run.

To use local weights, create a directory such as:

```text
models/serve/Qwen3-8B/
```

and place or symlink a normal Transformers/vLLM checkpoint there. Model files
are ignored by Git. Alternatively, authenticate with Hugging Face as required
by the upstream model and let the runner download the pinned snapshot.

The generation values were copied from the checkpoints’ official generation
configurations and are recorded here so that chat and Qwen think/no-think runs
do not depend on mutable library defaults.
