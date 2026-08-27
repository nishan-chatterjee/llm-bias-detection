# Dataset release tooling

Build a new, non-overwriting staging directory:

```bash
python dataset/package_release.py \
  --source /path/to/veritas-vox \
  --stage /tmp/polilean-hf-staging \
  --include-ablation \
  --include-restricted-pct-inputs \
  --hate-results-dir /path/to/output_hate_speech/results_hs \
  --hate-design /path/to/output_hate_speech/experimental_design_hs.csv
python dataset/validate_release.py /tmp/polilean-hf-staging
```

The proposition flag is suitable only when Political Compass redistribution
permission has been documented. Omit it for any public Dataset release unless
that documentation is available.

The hate conversion validates the supplied design, eight primary models,
1,800 configurations per model, 1,332 item positions per configuration, and
`True`/`False` probability sums before writing long Parquet. The resulting
private dataset contains no hate-speech source statements or raw logits.

Upload with `dataset/upload_private.py`. The helper refuses to target an
organization namespace and creates/updates a **public Dataset** repository by
default; pass `--private` only for a private staging release.
