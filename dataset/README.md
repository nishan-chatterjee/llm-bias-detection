# Dataset release tooling

Build a new, non-overwriting staging directory:

```bash
python dataset/package_release.py \
  --source /path/to/veritas-vox \
  --stage /tmp/llm-bias-detection-hf-staging \
  --include-ablation \
  --hate-results-dir /path/to/output_hate_speech/results_hs \
  --hate-design /path/to/output_hate_speech/experimental_design_hs.csv
python dataset/validate_release.py /tmp/llm-bias-detection-hf-staging
```

The packager refuses the deprecated proposition-export flag. Provide any
authorized questionnaire files locally for inference; do not publish them in
this dataset or a public supplemental-code ZIP. The uploader also refuses a
public stage containing a `restricted_inputs/` directory. This guard does not
clear rights over text embedded in raw chat traces.

The hate conversion validates the supplied design, eight primary models,
1,800 configurations per model, 1,332 item positions per configuration, and
`True`/`False` probability sums before writing long Parquet. The resulting
dataset contains no hate-speech source statements or raw logits.

The new input layer is packaged separately: exact JSONL and prompt JSON under
`inputs/hate_speech/`, plus a 13,320-row Parquet under `data/hate_speech_inputs/`.
Its `(target, item_index)` keys join to historical predictions. Code remains
MIT; author-owned documentation/selection/tables are CC BY 4.0, with explicit
third-party exceptions and notices copied into the Dataset.

To add just this layer to an existing downloaded release tree:

```bash
python dataset/materialize_hate_inputs.py --dataset-root data/release
```

Upload with `dataset/upload_private.py`. The helper refuses to target an
organization namespace and creates/updates a **public Dataset** repository by
default; pass `--private` only for a private staging release.
