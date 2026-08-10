"""
Check the reconstruction of the Political Compass scoring function.

Two independent checks:

  1. **Re-derivation.** Refit the scoring function from `political_compass_data.csv`
     and compare against the local `reconstructed_model_parameters.npz`. This
     reproduces the local parameters to machine precision, so the notebook is
     a reproduction step and not merely a record.

  2. **Behaviour.** Score every collected profile with the local parameters
     and report the held-out error, which is what the paper quotes.

One numerical detail decides whether check 1 passes, and it is worth stating
plainly because it is easy to reintroduce. The one-hot design has 248 columns
that are linearly dependent by construction -- every profile answers every item
exactly once, so each item's four indicator columns sum to one -- and the centred
design therefore has exactly 62 zero singular values. Least squares does not have
a unique solution; it has an affine subspace of them. We take the minimum-norm
element, which *is* unique and machine-independent, via
`np.linalg.lstsq(..., rcond=None)`.

Several LAPACK drivers reached through `scipy.linalg.lstsq` and
`sklearn.linear_model.LinearRegression` do not truncate that null space. They
invert singular values of order 1e-15 and return coefficients of order 1e11.
Those coefficients fit the training data equally well, so nothing looks wrong,
but the reduction step reads the *pattern* of the unconstrained weights to decide
which items are scored symmetrically. Under the blown-up solution that pattern is
destroyed: the reduction selects 138 and 150 features instead of 50 and 112, and
the recovered parameters no longer match. `--solver` below demonstrates this.

Usage:  python verify_instrument.py              (exit 0 on success)
        python verify_instrument.py --solver     (show the solver comparison)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "political_compass_data.csv")
PARAMS = os.path.join(HERE, "reconstructed_model_parameters.npz")

N_ITEMS = 62
N_TRAIN = 322          # the fit used the first 322 profiles, holding out 50
TOLERANCE = 0.01

# Values reported in the paper, for the reduced model on the 50 held-out profiles.
EXPECTED = {
    'economic': {'rmse': 0.0034, 'r2': 0.999998, 'features': 50, 'items': 18},
    'social':   {'rmse': 0.0033, 'r2': 0.999993, 'features': 112, 'items': 43},
}


def one_hot(X_cat):
    X = np.zeros((len(X_cat), N_ITEMS * 4))
    for i in range(len(X_cat)):
        for j in range(N_ITEMS):
            if 0 <= X_cat[i, j] <= 3:
                X[i, 4 * j + X_cat[i, j]] = 1
    return X


def fit_least_squares(X, target, n_train):
    """Minimum-norm least squares with intercept. See module docstring."""
    X_tr, y_tr = X[:n_train], target[:n_train]
    X_mean, y_mean = X_tr.mean(axis=0), y_tr.mean()
    coef = np.linalg.lstsq(X_tr - X_mean, y_tr - y_mean, rcond=None)[0]
    return coef, float(y_mean - X_mean @ coef)


def build_reduced_design(X, weights):
    columns, structure = [], []
    for j in range(N_ITEMS):
        w0, w1, w2, w3 = weights[j]
        symmetric = (abs(w3 + w0 - w2 - w1) < 2 * TOLERANCE
                     and abs(w0) + abs(w1) + abs(w2) + abs(w3) > 4 * TOLERANCE)
        if symmetric:
            columns.append(X[:, 4 * j + 0] - X[:, 4 * j + 3])
            columns.append(X[:, 4 * j + 1] - X[:, 4 * j + 2])
            structure.append(('symmetric', None))
        else:
            kept = [k for k in range(4) if abs(weights[j, k]) >= TOLERANCE]
            columns.extend(X[:, 4 * j + k] for k in kept)
            structure.append(('tolerance', kept))
    return np.column_stack(columns), structure


def expand_weights(coef, structure):
    out = np.zeros((N_ITEMS, 4))
    i = 0
    for j, (kind, kept) in enumerate(structure):
        if kind == 'symmetric':
            a, b = coef[i], coef[i + 1]
            out[j] = [a, b, -b, -a]
            i += 2
        else:
            for k in kept:
                out[j, k] = coef[i]
                i += 1
    return out


def reconstruct(X_one_hot, y, col):
    unconstrained, _ = fit_least_squares(X_one_hot, y[:, col], N_TRAIN)
    X_red, structure = build_reduced_design(X_one_hot, unconstrained.reshape(N_ITEMS, 4))
    coef, bias = fit_least_squares(X_red, y[:, col], N_TRAIN)
    return expand_weights(coef, structure), bias, X_red.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solver', action='store_true',
                    help="show why the least-squares driver matters")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    X_cat = df.iloc[:, :N_ITEMS].values.astype(int)
    y = df.iloc[:, N_ITEMS:].values.astype(float)
    X_one_hot = one_hot(X_cat)
    local_params = np.load(PARAMS)

    print(f"{len(df)} profiles, {N_ITEMS} items, {N_TRAIN} used for fitting, "
          f"{len(df) - N_TRAIN} held out\n")

    ok = True
    for axis, col in [('economic', 0), ('social', 1)]:
        exp = EXPECTED[axis]
        W, b, n_feat = reconstruct(X_one_hot, y, col)
        W_ref = local_params[f'weights_{axis}']
        b_ref = float(local_params[f'bias_{axis}'])
        dW = float(np.abs(W - W_ref).max())
        n_items = int((np.abs(W_ref) > 1e-12).any(axis=1).sum())

        pred = X_one_hot[N_TRAIN:] @ W_ref.flatten() + b_ref
        truth = y[N_TRAIN:, col]
        resid = pred - truth
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        r2 = 1 - float(np.sum(resid ** 2)) / float(np.sum((truth - truth.mean()) ** 2))

        print(f"--- {axis} axis ---")
        print(f"  re-derived vs local      max|dW| {dW:.2e}   "
              f"|db| {abs(b - b_ref):.2e}")
        print(f"  reduced features         {n_feat:3d}   (paper: {exp['features']})")
        print(f"  items with weight        {n_items:3d}   (paper: {exp['items']})")
        print(f"  held-out RMSE            {rmse:.6f}   (paper: {exp['rmse']})")
        print(f"  held-out R^2             {r2:.6f}   (paper: {exp['r2']})")

        if dW > 1e-9:
            print("    MISMATCH: re-derivation does not reproduce the local weights")
            ok = False
        if n_feat != exp['features'] or n_items != exp['items']:
            print("    MISMATCH: feature/item counts differ from the paper")
            ok = False
        if abs(round(rmse, 4) - exp['rmse']) > 1e-4 or abs(round(r2, 6) - exp['r2']) > 2e-6:
            print("    MISMATCH: held-out error differs from the paper")
            ok = False
        print()

    if args.solver:
        import scipy.linalg as sla
        from sklearn.linear_model import LinearRegression
        X_tr = X_one_hot[:N_TRAIN]
        Xc = X_tr - X_tr.mean(axis=0)
        yc = y[:N_TRAIN, 0] - y[:N_TRAIN, 0].mean()
        s = np.linalg.svd(Xc, compute_uv=False)
        print("--- why the solver matters (economic axis) ---")
        print(f"  centred design: {Xc.shape[1]} columns, "
              f"{int((s < 1e-10).sum())} zero singular values "
              f"(one per item), numerical rank {np.linalg.matrix_rank(Xc)}\n")
        cands = {
            'np.linalg.lstsq(rcond=None)': np.linalg.lstsq(Xc, yc, rcond=None)[0],
            'np.linalg.pinv':              np.linalg.pinv(Xc) @ yc,
            'scipy.linalg.lstsq(gelsd)':   sla.lstsq(Xc, yc, lapack_driver='gelsd')[0],
            'sklearn LinearRegression':    LinearRegression().fit(X_tr, y[:N_TRAIN, 0]).coef_,
        }
        print(f"  {'solver':<30} {'max|w|':>11}   resulting reduced features")
        for name, w in cands.items():
            X_red, _ = build_reduced_design(X_one_hot, w.reshape(N_ITEMS, 4))
            print(f"  {name:<30} {np.abs(w).max():>11.3g}   {X_red.shape[1]}")
        print()

    if ok:
        print("Reconstruction reproduces the local parameters and the reported values.")
        return 0
    print("Reconstruction does NOT match.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
