# Dataset release tooling

Build a new, non-overwriting staging directory:

```bash
python dataset/package_release.py \
  --source /path/to/veritas-vox \
  --stage /tmp/polilean-hf-staging \
  --include-ablation \
  --include-restricted-pct-inputs
python dataset/validate_release.py /tmp/polilean-hf-staging
```

The proposition flag is suitable only for the initial private repository. Omit
it when preparing a public version unless Political Compass redistribution
permission has been documented.

Upload with `dataset/upload_private.py`. The helper refuses to target an
organization namespace and creates/updates a **private Dataset** repository.
