"""
Day 2 — Full pipeline: skore.evaluate + Skore Hub push + Kaggle submissions
============================================================================

Steps covered (matches DAY2.md exactly):
  Step 1-2  : Ridge with grouped CV  →  02_ridge
  Step 3    : GroupKFold comparison  →  (already 01_dummy on Hub; adds 02_ridge)
  Step 4    : HistGBM NaN-native     →  03_hist_gradient_boosting
  Step 5    : skrub tabular_pipeline →  05_tabular_pipeline
  Step 6    : skrub DataOps          →  06_dataops
  Step 7    : best-model submission  →  best CSV already written; pushes Hub report

Run from repo root:
  python notebooks/day2_skore.py
"""

from __future__ import annotations

import json
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

PROJECT_NAME = "bobathon-esilv"
N_SPLITS = 5

# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------

def hub_project():
    """Return a connected skore.Project (hub mode). Raises on failure."""
    import skore
    cfg = json.loads((REPO_ROOT / ".skore").read_text())
    skore.login(mode="hub")
    return skore.Project(name=PROJECT_NAME, mode="hub", workspace=cfg["workspace"])


def push(project, key: str, report) -> str:
    """Put a report and return the public URL (printed by skore)."""
    project.put(key, report)
    print(f"  [Hub] '{key}' pushed.")
    return key


# ---------------------------------------------------------------------------
# Data loading — follows DAY2.md style exactly
# ---------------------------------------------------------------------------

def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

    # visits = merged table (DAY2.md pattern)
    visits = X_train.merge(y_train, on="Index")
    y = visits["target"]

    print(f"Loaded: {len(visits):,} train visits, "
          f"{visits['patient_id'].nunique():,} patients, "
          f"{len(X_test):,} test visits")
    return X_train, y_train, X_test, visits, y


# ---------------------------------------------------------------------------
# Feature sets (DAY2.md)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off",
]


def save_submission(X_test, preds, key: str):
    sub = X_test[["Index"]].copy()
    sub["target"] = preds
    path = SUBMISSIONS_DIR / f"{key}.csv"
    sub.to_csv(path, index=False)
    # Verify
    assert list(sub.columns) == ["Index", "target"]
    assert len(sub) == len(X_test)
    assert sub["target"].notna().all()
    print(f"  Submission saved: submissions/{key}.csv  ({len(sub):,} rows)")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X_train_raw, y_train_raw, X_test_raw, visits, y = load_data()

    # GroupKFold splits — precomputed once, reused everywhere (DAY2.md §3)
    groups = visits["patient_id"]
    cv_splits = list(GroupKFold(n_splits=N_SPLITS).split(
        visits[FEATURE_COLS], y, groups=groups
    ))
    X = visits[FEATURE_COLS]

    # Connect Hub
    try:
        project = hub_project()
        print("Skore Hub: connected\n")
    except Exception as exc:
        print(f"WARNING: Hub unavailable ({type(exc).__name__}). "
              "Reports will NOT be pushed.", file=sys.stderr)
        project = None

    import skore

    # -------------------------------------------------------------------------
    # Step 1-2: Ridge (DAY2.md §1-2)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("02_ridge — Ridge with GroupKFold CV")
    print("=" * 60)

    dummy = DummyRegressor(strategy="mean")
    ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))

    # Evaluate with GroupKFold splits
    report_ridge = skore.evaluate(ridge, X, y, splitter=cv_splits)
    rmse_ridge = report_ridge.metrics.rmse()
    print(f"  Ridge CV RMSE:\n{rmse_ridge}")

    if project:
        push(project, "02_ridge", report_ridge)

    # Final model → submission
    final_ridge = clone(ridge).fit(X, y)
    preds_ridge = final_ridge.predict(X_test_raw[FEATURE_COLS])
    save_submission(X_test_raw, preds_ridge, "02_ridge")

    # -------------------------------------------------------------------------
    # Step 4: HistGradientBoosting — missingness as signal (DAY2.md §4)
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("03_hist_gradient_boosting — HistGBM NaN-native, GroupKFold CV")
    print("=" * 60)

    hgbr = HistGradientBoostingRegressor(random_state=0)
    report_hgbr = skore.evaluate(hgbr, X, y, splitter=cv_splits)
    rmse_hgbr = report_hgbr.metrics.rmse()
    print(f"  HistGBM CV RMSE:\n{rmse_hgbr}")

    if project:
        push(project, "03_hist_gradient_boosting", report_hgbr)

    final_hgbr = clone(hgbr).fit(X, y)
    preds_hgbr = final_hgbr.predict(X_test_raw[FEATURE_COLS])
    save_submission(X_test_raw, preds_hgbr, "03_hist_gradient_boosting")

    # -------------------------------------------------------------------------
    # Step 5: skrub tabular_pipeline (DAY2.md §5)
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("05_tabular_pipeline — skrub tabular_pipeline, all columns")
    print("=" * 60)

    from skrub import tabular_pipeline

    # X_full: drop identifiers + target (DAY2.md pattern)
    X_full = visits.drop(columns=["Index", "patient_id", "target"])
    X_test_full = X_test_raw.drop(columns=["Index", "patient_id"], errors="ignore")

    # cv_splits_full uses same patient groups but with X_full row count
    cv_splits_full = list(GroupKFold(n_splits=N_SPLITS).split(
        X_full, y, groups=groups
    ))

    model_tabular = tabular_pipeline("regressor")
    report_tabular = skore.evaluate(model_tabular, X_full, y, splitter=cv_splits_full)
    rmse_tabular = report_tabular.metrics.rmse()
    print(f"  tabular_pipeline CV RMSE:\n{rmse_tabular}")

    if project:
        push(project, "05_tabular_pipeline", report_tabular)

    final_tabular = clone(model_tabular).fit(X_full, y)
    preds_tabular = final_tabular.predict(X_test_full)
    save_submission(X_test_raw, preds_tabular, "05_tabular_pipeline")

    # -------------------------------------------------------------------------
    # Step 6: skrub DataOps (DAY2.md §6)
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("06_dataops — skrub DataOps, GroupKFold baked in")
    print("=" * 60)

    import skrub

    data_op = skrub.var("visits", visits)
    groups_op = data_op["patient_id"]
    X_op = data_op.drop("target", axis=1).skb.mark_as_X(
        cv=GroupKFold(n_splits=N_SPLITS),
        split_kwargs={"groups": groups_op},
    )
    y_op = data_op["target"].skb.mark_as_y()
    pred_op = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=0),
        y=y_op,
    )

    # evaluate reads cv/groups from the DataOp
    report_dataops = skore.evaluate(pred_op)
    rmse_dataops = report_dataops.metrics.rmse()
    print(f"  DataOps CV RMSE:\n{rmse_dataops}")

    if project:
        push(project, "06_dataops", report_dataops)

    # Submission via SkrubLearner
    learner = pred_op.skb.make_learner()
    learner.fit({"visits": visits})
    preds_dataops = learner.predict({"visits": X_test_raw})
    save_submission(X_test_raw, preds_dataops, "06_dataops")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY — Mean RMSE (5-fold GroupKFold by patient_id)")
    print("=" * 70)

    def _mean_rmse(report):
        try:
            frame = report.metrics.rmse(aggregate=None)
            # frame is a DataFrame with splits as columns
            vals = frame.values.flatten()
            return float(np.mean(vals)), float(np.std(vals))
        except Exception:
            return float("nan"), float("nan")

    rows = [
        ("02_ridge",             report_ridge),
        ("03_hist_gradient_boosting", report_hgbr),
        ("05_tabular_pipeline",  report_tabular),
        ("06_dataops",           report_dataops),
    ]
    print(f"{'Key':<35} {'Mean RMSE':>10} {'Std':>8}")
    print("-" * 55)
    for key, rep in sorted(rows, key=lambda x: _mean_rmse(x[1])[0]):
        m, s = _mean_rmse(rep)
        print(f"{key:<35} {m:>10.4f} {s:>8.4f}")
    print("=" * 70)

    if project:
        print("\nAll reports pushed. Visit https://skore.probabl.ai")
    else:
        print("\nHub push skipped (not connected).")


if __name__ == "__main__":
    main()
