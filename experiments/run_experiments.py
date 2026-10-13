"""
Bobathon - RMSE evaluation (local, no Skore required)
======================================================

Evaluates every model with RMSE using 5-fold GroupKFold by patient_id,
saves a summary to experiments/results_rmse.csv, and writes one Kaggle
submission CSV per experiment to submissions/.

Experiment keys (also used as CSV file names):
  01_dummy
  02_ridge
  03_hist_gradient_boosting
  04_hist_gradient_boosting_tuned

Usage (from repo root, inside the virtual environment):
  python experiments/run_experiments.py

Skore Hub integration: disabled in this version.
Re-enable by running experiments/push_to_skore.py (not yet created).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn

    def compute_rmse(y_true, y_pred) -> float:
        return float(_rmse_fn(y_true, y_pred))
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse_fn

    def compute_rmse(y_true, y_pred) -> float:  # type: ignore[misc]
        return float(_mse_fn(y_true, y_pred) ** 0.5)


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
DROP_COLS = ["patient_id", "Index"]


# ---------------------------------------------------------------------------
# Feature engineering  (same logic as Notebook.ipynb)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with engineered features; drops identifier columns."""
    d = df.copy()
    d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
    d["on_off_diff"] = d["off"] - d["on"]
    d["on_off_mean"] = (d["on"] + d["off"]) / 2.0
    return d.drop(columns=DROP_COLS)


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def cv_rmse_loop(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    gkf: GroupKFold,
    *,
    need_imputation: bool = False,
) -> np.ndarray:
    """Return per-fold RMSE array using GroupKFold (patients never split).

    Parameters
    ----------
    need_imputation : bool
        When True, fit a ``SimpleImputer(median)`` on each train fold and
        apply it to the val fold before passing data to the model.  Used
        for Ridge, which cannot handle NaN.  The imputer is never fitted on
        val data, so there is no leakage.
    """
    fold_rmse: list[float] = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        m = clone(model)
        if need_imputation:
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            X_val = imp.transform(X_val)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        fold_rmse.append(compute_rmse(y_val, preds))

    return np.array(fold_rmse)


# ---------------------------------------------------------------------------
# Per-experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    key: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_index: pd.Series,
    groups: pd.Series,
    gkf: GroupKFold,
    *,
    need_imputation: bool = False,
) -> dict:
    """Evaluate one model, print results, write submission CSV.

    Returns a metadata dict with fold_rmse / mean_rmse / std_rmse.
    """
    print(f"\n{'='*60}")
    print(f"Experiment : {key}")
    print(f"Model      : {model.__class__.__name__}")
    params = {k: v for k, v in model.get_params().items() if not callable(v)}
    print(f"Params     : {params}")
    print(f"{'='*60}")

    # -- Per-fold RMSE (GroupKFold, no patient leakage) -----------------------
    fold_rmse = cv_rmse_loop(
        model, X_train, y_train, groups, gkf,
        need_imputation=need_imputation,
    )
    for i, r in enumerate(fold_rmse, 1):
        print(f"  Fold {i}: RMSE = {r:.4f}")

    mean_r = float(fold_rmse.mean())
    std_r = float(fold_rmse.std())
    print(f"\n  Mean RMSE : {mean_r:.4f}")
    print(f"  Std  RMSE : {std_r:.4f}")

    # -- Final model: train on ALL data, predict test -------------------------
    if need_imputation:
        final = make_pipeline(SimpleImputer(strategy="median"), clone(model))
    else:
        final = clone(model)
    final.fit(X_train, y_train)
    test_preds = final.predict(X_test)

    # -- Submission CSV -------------------------------------------------------
    submission = pd.DataFrame({"Index": test_index, "target": test_preds})
    csv_path = SUBMISSIONS_DIR / f"{key}.csv"
    submission.to_csv(csv_path, index=False)
    print(f"  Submission: {csv_path.relative_to(REPO_ROOT)}")

    return {
        "experiment": key,
        "model": model.__class__.__name__,
        "fold_rmse": fold_rmse,
        "mean_rmse": mean_r,
        "std_rmse": std_r,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Load raw data --------------------------------------------------------
    print("Loading data ...")
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    assert (X_train_raw["Index"] == y_train_raw["Index"]).all(), \
        "Index mismatch between X_train and y_train!"
    y = y_train_raw["target"].copy()

    print(f"  Train rows : {len(X_train_raw):,}")
    print(f"  Test rows  : {len(X_test_raw):,}")
    print(f"  Patients   : {X_train_raw['patient_id'].nunique():,}")

    # -- Feature engineering --------------------------------------------------
    X_feat_train = engineer_features(X_train_raw)
    X_feat_test = engineer_features(X_test_raw)

    # Encode categoricals - fit on TRAIN only (no leakage to test/val)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_feat_train[CATEGORICAL_COLS] = encoder.fit_transform(
        X_feat_train[CATEGORICAL_COLS].astype(str)
    )
    X_feat_test[CATEGORICAL_COLS] = encoder.transform(
        X_feat_test[CATEGORICAL_COLS].astype(str)
    )

    test_index = X_test_raw["Index"]

    # -- GroupKFold by patient ------------------------------------------------
    groups = X_train_raw["patient_id"]
    gkf = GroupKFold(n_splits=N_SPLITS)
    print(f"\nValidation : GroupKFold(n_splits={N_SPLITS}) grouped by patient_id")

    # -- Experiments ----------------------------------------------------------
    common = dict(
        X_train=X_feat_train,
        y_train=y,
        X_test=X_feat_test,
        test_index=test_index,
        groups=groups,
        gkf=gkf,
    )
    results: list[dict] = []

    results.append(run_experiment(
        "01_dummy",
        DummyRegressor(strategy="mean"),
        **common,
    ))

    results.append(run_experiment(
        "02_ridge",
        Ridge(alpha=1.0),
        need_imputation=True,
        **common,
    ))

    results.append(run_experiment(
        "03_hist_gradient_boosting",
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        **common,
    ))

    best_params = dict(
        max_iter=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    results.append(run_experiment(
        "04_hist_gradient_boosting_tuned",
        HistGradientBoostingRegressor(**best_params),
        **common,
    ))

    # -- Summary --------------------------------------------------------------
    header = f"\n\n{'='*72}"
    print(header)
    print("EXPERIMENT SUMMARY - RMSE (5-fold GroupKFold by patient_id)")
    print("="*72)
    print(f"{'Experiment':<45} {'Mean RMSE':>10} {'Std RMSE':>10}")
    print("-"*72)
    for res in sorted(results, key=lambda x: x["mean_rmse"]):
        print(
            f"{res['experiment']:<45}"
            f"{res['mean_rmse']:>10.4f}"
            f"{res['std_rmse']:>10.4f}"
        )
    print("="*72)

    best = min(results, key=lambda x: x["mean_rmse"])
    print(f"\nBest model : {best['experiment']} ({best['model']})")
    print(f"Best RMSE  : {best['mean_rmse']:.4f} +/- {best['std_rmse']:.4f}")

    # -- Save results CSV -----------------------------------------------------
    rows = []
    for res in results:
        row = {
            "experiment": res["experiment"],
            "model": res["model"],
            "mean_rmse": round(res["mean_rmse"], 6),
            "std_rmse": round(res["std_rmse"], 6),
        }
        for i, r in enumerate(res["fold_rmse"], 1):
            row[f"fold_{i}_rmse"] = round(float(r), 6)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = EXPERIMENTS_DIR / "results_rmse.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved : {results_path.relative_to(REPO_ROOT)}")

    # -- Verify submission CSVs -----------------------------------------------
    print("\nSubmission file verification:")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    for res in results:
        csv_path = SUBMISSIONS_DIR / f"{res['experiment']}.csv"
        sub = pd.read_csv(csv_path)
        ok_cols = list(sub.columns) == ["Index", "target"]
        ok_rows = len(sub) == len(sample_sub)
        ok_no_nan = sub["target"].notna().all()
        ok_index = sub["Index"].equals(sample_sub["Index"])
        status = "PASS" if all([ok_cols, ok_rows, ok_no_nan, ok_index]) else "FAIL"
        print(
            f"  {res['experiment']}.csv : {status}"
            f" (rows={len(sub)}, no_nan={ok_no_nan}, index_match={ok_index})"
        )

    print("\nDone. Skore Hub push was skipped (run push_to_skore.py when ready).")


if __name__ == "__main__":
    main()
