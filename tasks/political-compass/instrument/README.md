# Scorer reconstruction

`reconstruct_scoring_function.ipynb` documents how the two-axis scoring
function was recovered from randomized response profiles and how the learned
parameters were validated on held-out profiles. It defaults to a safe
documentation mode: it neither submits profiles to the website nor fits/writes
the scorer.

The generated `.npz` is intentionally ignored and is not redistributed. The
profile/coordinate table used to fit it is also kept out of the public Git
branch while the Political Compass rights question is unresolved. An
authorized copy must be named `political_compass_data.csv` in this directory.
To opt into the local fit, launch the notebook with
`POLILEAN_ENABLE_SCORER_RECONSTRUCTION=1`. The notebook does not print the
recovered weight arrays.

After running the notebook, validate it with:

```bash
python tasks/political-compass/instrument/verify_instrument.py
```

Expected reconstruction diagnostics are 372 profiles, a 322/50 train/held-out
split, 50 economic and 112 social reduced features, held-out RMSE 0.0034 and
0.0033, and held-out R² 0.999998 and 0.999993. These describe the recovered
offline approximation; they do not confer rights to redistribute the original
instrument.
