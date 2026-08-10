#!/usr/bin/env python3
"""Build the redistribution-safe scorer reconstruction notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "reconstruct_scoring_function.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Reconstructing the Political Compass scoring function

The Political Compass returns a two-dimensional coordinate but does not publish
its scoring function. For the experiments, randomized 62-answer profiles were
submitted to the live instrument and paired with the returned coordinates.
Those profiles were one-hot encoded (62 items × 4 answers), then separate
minimum-norm linear models were fitted for the economic and social axes.

The calibration table and fitted `.npz` are deliberately **not redistributed**.
This notebook documents and executes the fitting method only when an authorized
local copy is present and the opt-in environment variable is set. It does not
automate new submissions to the website.

The first 322 of 372 profiles form the fit set and the final 50 are held out.
The one-hot design is rank-deficient because the four columns for every item sum
to one. `np.linalg.lstsq(..., rcond=None)` selects the reproducible
minimum-norm solution before the detected sparse/symmetric structure is refit."""
        ),
        nbf.v4.new_code_cell(
            """import os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path.cwd()
if HERE.name != "instrument":
    candidate = HERE / "tasks" / "political-compass" / "instrument"
    if candidate.exists():
        HERE = candidate

CALIBRATION = HERE / "political_compass_data.csv"
OUTPUT = HERE / "reconstructed_model_parameters.npz"
ENABLED = os.environ.get("POLILEAN_ENABLE_SCORER_RECONSTRUCTION") == "1"
print("Authorized reconstruction enabled:", ENABLED)
print("Calibration table present:", CALIBRATION.exists())"""
        ),
        nbf.v4.new_markdown_cell(
            """## Why minimum-norm least squares is required

The small synthetic calculation below demonstrates the same structural null
space without revealing any Political Compass parameters."""
        ),
        nbf.v4.new_code_cell(
            """rng = np.random.default_rng(42)
answers = rng.integers(0, 4, size=(80, 62))
one_hot = np.zeros((len(answers), 62 * 4))
for row in range(len(answers)):
    for item in range(62):
        one_hot[row, 4 * item + answers[row, item]] = 1
centered = one_hot - one_hot.mean(axis=0)
print("Synthetic design shape:", centered.shape)
print("Synthetic numerical rank:", np.linalg.matrix_rank(centered))
print("Columns minus rank:", centered.shape[1] - np.linalg.matrix_rank(centered))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Authorized local reconstruction

To run this cell, place `political_compass_data.csv` beside the notebook and
launch Jupyter with `POLILEAN_ENABLE_SCORER_RECONSTRUCTION=1`. The helper module
contains the one-hot encoding, minimum-norm fit, structural reduction, held-out
checks, and expected paper diagnostics. No weight arrays are printed."""
        ),
        nbf.v4.new_code_cell(
            """if ENABLED:
    if not CALIBRATION.exists():
        raise FileNotFoundError(CALIBRATION)
    import sys
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import verify_instrument as instrument

    frame = pd.read_csv(CALIBRATION)
    categorical = frame.iloc[:, :instrument.N_ITEMS].to_numpy(dtype=int)
    targets = frame.iloc[:, instrument.N_ITEMS:].to_numpy(dtype=float)
    design = instrument.one_hot(categorical)
    parameters = {}
    rows = []
    for axis, column in (("economic", 0), ("social", 1)):
        weights, bias, features = instrument.reconstruct(design, targets, column)
        parameters[f"weights_{axis}"] = weights
        parameters[f"bias_{axis}"] = bias
        prediction = design[instrument.N_TRAIN:] @ weights.ravel() + bias
        truth = targets[instrument.N_TRAIN:, column]
        residual = prediction - truth
        rows.append({
            "axis": axis,
            "features": features,
            "held_out_rmse": np.sqrt(np.mean(residual ** 2)),
            "held_out_r2": 1 - np.sum(residual ** 2) / np.sum((truth - truth.mean()) ** 2),
        })
    np.savez(OUTPUT, **parameters)
    display(pd.DataFrame(rows))
    print("Wrote local ignored file:", OUTPUT)
else:
    print("Safe documentation mode: no scorer was fitted or written.")"""
        ),
        nbf.v4.new_markdown_cell(
            """Expected diagnostics for the archived authorized calibration are
50 economic and 112 social reduced features, held-out RMSE approximately
0.0034/0.0033, and held-out R² approximately 0.999998/0.999993. Run
`python verify_instrument.py` locally to compare the re-derived parameters with
an authorized local reference file. These diagnostics do not confer permission
to redistribute the instrument, calibration data, or fitted parameters."""
        ),
    ]
    nbf.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
