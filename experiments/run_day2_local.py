"""
Day 2 — Complete local pipeline (no Skore uploads)
====================================================

Covers all DAY2.md steps using only local evaluation.
Skore Hub uploads are intentionally disabled here.
Run from repo root:  python experiments/run_day2_local.py

Models evaluated:
  01_dummy                   — DummyRegressor(mean)        [already run]
  02_ridge                   — Ridge + SimpleImputer        [DAY2 step 1-2]
  03_hist_gradient_boosting  — HistGBM NaN-native           [DAY2 step 4]
  05_tabular_pipeline        — skrub tabular_pipeline        [DAY2 step 5]
  06_dataops                 — skrub DataOps                 [DAY2 step 6]

CSV safety: never overwrites an existing file; appends +1, +2 … until free.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

try:
    from sklearn.metrics import root_mean_squared_error as _sklearn_rmse
    def rmse(y_true, y_pred): return float(_sklearn_rmse(y_true, y_pred))
except ImportError:
    def rmse(y_true, y_pred): return float(mean_squared_error(y_true, y_pred) ** 0.5)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "data"
SUBMISSIONS = REPO_ROOT / "submissions"
EXPERIMENTS = REPO_ROOT / "experiments"
SUBMISSIONS.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

N_SPLITS    = 5
RANDOM_STATE = 0    # matches DAY2.md examples

# ---------------------------------------------------------------------------
# CSV safety helper
# ---------------------------------------------------------------------------
def safe_csv_path(key: str) -> Path:
    """Return SUBMISSIONS/key.csv; if taken, try key+1.csv, key+2.csv, …"""
    base = SUBMISSIONS / f"{key}.csv"
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = SUBMISSIONS / f"{key}+{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def save_submission(X_test_raw: pd.DataFrame, preds: np.ndarray, key: str) -> Path:
    """Build, validate, and save a submission CSV with collision-safe naming."""
    path = safe_csv_path(key)
    sub = X_test_raw[["Index"]].copy()
    sub["target"] = preds

    # --- Validation ---
    assert list(sub.columns) == ["Index", "target"], "Wrong columns"
    assert len(sub) == len(X_test_raw), "Row count mismatch"
    assert sub["target"].notna().all(), "NaN predictions"
    assert all(math.isfinite(v) for v in sub["target"]), "Non-finite predictions"

    # Index order must match sample_submission.csv
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert sub["Index"].equals(sample["Index"]), "Index order mismatch vs sample"

    sub.to_csv(path, index=False)
    suffix = " (NEW)" if "+" not in path.name else f" (collision -> {path.name})"
    print(f"  Saved: submissions/{path.name}{suffix}  [{len(sub):,} rows]")
    return path


# ---------------------------------------------------------------------------
# Cross-validation loop — RMSE + MAE per fold
# ---------------------------------------------------------------------------
def cv_eval(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    gkf: GroupKFold,
    *,
    need_imputation: bool = False,
) -> dict:
    """Return per-fold and aggregate RMSE + MAE. No patient leakage."""
    fold_rmse, fold_mae = [], []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = clone(model)
        if need_imputation:
            imp = SimpleImputer(strategy="median")
            X_tr  = imp.fit_transform(X_tr)
            X_val = imp.transform(X_val)

        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        fold_rmse.append(rmse(y_val, preds))
        fold_mae.append(float(mean_absolute_error(y_val, preds)))

    return {
        "fold_rmse": fold_rmse,
        "fold_mae":  fold_mae,
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_rmse":  float(np.std(fold_rmse)),
        "mean_mae":  float(np.mean(fold_mae)),
        "std_mae":   float(np.std(fold_mae)),
    }


def print_cv_result(key: str, res: dict):
    print(f"\n  Key        : {key}")
    for i, (r, m) in enumerate(zip(res["fold_rmse"], res["fold_mae"]), 1):
        print(f"  Fold {i}     : RMSE={r:.4f}  MAE={m:.4f}")
    print(f"  Mean RMSE  : {res['mean_rmse']:.4f} ± {res['std_rmse']:.4f}")
    print(f"  Mean MAE   : {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")


# ---------------------------------------------------------------------------
# DataOps cross-validation helper (uses skrub's own CV machinery)
# ---------------------------------------------------------------------------
def cv_eval_dataops(pred_op, visits: pd.DataFrame, y: pd.Series) -> dict:
    """Run skrub DataOps CV locally without touching the Hub."""
    from skore import evaluate as skore_evaluate

    # DataOp needs data= dict so the named var 'visits' can be bound
    report = skore_evaluate(pred_op, data={"visits": visits})

    # Extract per-fold metrics
    fold_rmse_df = report.metrics.rmse(aggregate=None)
    fold_mae_df  = report.metrics.mae(aggregate=None)

    fold_rmse = [float(v) for v in fold_rmse_df.values.flatten()]
    fold_mae  = [float(v) for v in fold_mae_df.values.flatten()]

    return {
        "fold_rmse": fold_rmse,
        "fold_mae":  fold_mae,
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_rmse":  float(np.std(fold_rmse)),
        "mean_mae":  float(np.mean(fold_mae)),
        "std_mae":   float(np.std(fold_mae)),
        "_report":   report,          # kept for later Hub push
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # -- Load data (DAY2.md pattern) -----------------------------------------
    print("Loading data ...")
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    y      = visits["target"]

    print(f"  {len(visits):,} train visits | {visits['patient_id'].nunique():,} patients "
          f"| {len(X_test_raw):,} test visits")

    # Shared GroupKFold grouped by patient
    groups = visits["patient_id"]
    gkf    = GroupKFold(n_splits=N_SPLITS)

    # -- Feature sets ---------------------------------------------------------
    # DAY2 numeric features (Steps 1-4)
    FEATURE_COLS = [
        "sexM", "age_at_diagnosis", "age", "ledd",
        "time_since_intake_on", "time_since_intake_off",
        "on", "off",
    ]
    X      = visits[FEATURE_COLS]
    X_test = X_test_raw[FEATURE_COLS]

    # DAY2 full features (Steps 5-6 — drop ids + target)
    X_full      = visits.drop(columns=["Index", "patient_id", "target"])
    X_test_full = X_test_raw.drop(columns=["Index", "patient_id"], errors="ignore")

    # cv_splits_full: same groups, different X shape
    cv_splits_full = list(gkf.split(X_full, y, groups))

    all_results: list[dict] = []

    # =========================================================================
    # Step 1-2: Ridge (DAY2.md §1-2)
    # =========================================================================
    print("\n" + "=" * 62)
    print("02_ridge  |  Ridge(alpha=1.0) + SimpleImputer  |  GroupKFold CV")
    print("=" * 62)
    ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
    res_ridge = cv_eval(ridge, X, y, groups, gkf)
    print_cv_result("02_ridge", res_ridge)

    # Final model on all train data → submission
    final_ridge = clone(ridge).fit(X, y)
    csv_ridge = save_submission(X_test_raw, final_ridge.predict(X_test), "02_ridge")
    all_results.append({"key": "02_ridge", **res_ridge, "csv": csv_ridge.name})

    # =========================================================================
    # Step 4: HistGBM NaN-native (DAY2.md §4)
    # =========================================================================
    print("\n" + "=" * 62)
    print("03_hist_gradient_boosting  |  HistGBM(random_state=0)  |  NaN-native")
    print("=" * 62)
    hgbr = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    res_hgbr = cv_eval(hgbr, X, y, groups, gkf)
    print_cv_result("03_hist_gradient_boosting", res_hgbr)

    final_hgbr = clone(hgbr).fit(X, y)
    csv_hgbr = save_submission(X_test_raw, final_hgbr.predict(X_test),
                                "03_hist_gradient_boosting")
    all_results.append({"key": "03_hist_gradient_boosting", **res_hgbr, "csv": csv_hgbr.name})

    # =========================================================================
    # Step 5: skrub tabular_pipeline (DAY2.md §5)
    # =========================================================================
    print("\n" + "=" * 62)
    print("05_tabular_pipeline  |  skrub tabular_pipeline  |  all columns")
    print("=" * 62)
    from skrub import tabular_pipeline as skrub_tabular_pipeline

    model_tabular = skrub_tabular_pipeline("regressor")
    res_tabular = cv_eval(model_tabular, X_full, y, groups, gkf)
    print_cv_result("05_tabular_pipeline", res_tabular)

    final_tabular = clone(model_tabular).fit(X_full, y)
    csv_tabular = save_submission(X_test_raw, final_tabular.predict(X_test_full),
                                   "05_tabular_pipeline")
    all_results.append({"key": "05_tabular_pipeline", **res_tabular, "csv": csv_tabular.name})

    # =========================================================================
    # Step 6: skrub DataOps (DAY2.md §6)
    # =========================================================================
    print("\n" + "=" * 62)
    print("06_dataops  |  skrub DataOps + GroupKFold baked in")
    print("=" * 62)
    import skrub

    data_op   = skrub.var("visits", visits)
    groups_op = data_op["patient_id"]
    # errors="ignore" so the drop works on X_test_raw too (no "target" column there)
    X_op = (
        data_op.drop("target", axis=1, errors="ignore")
        .skb.mark_as_X(
            cv=GroupKFold(n_splits=N_SPLITS),
            split_kwargs={"groups": groups_op},
        )
    )
    y_op = data_op["target"].skb.mark_as_y()
    pred_op = (
        X_op
        .skb.apply(skrub.TableVectorizer())
        .skb.apply(HistGradientBoostingRegressor(random_state=RANDOM_STATE), y=y_op)
    )

    res_dataops = cv_eval_dataops(pred_op, visits, y)
    print_cv_result("06_dataops", res_dataops)

    # Final predictions via SkrubLearner
    learner = pred_op.skb.make_learner()
    learner.fit({"visits": visits})
    # X_test_raw has no "target" column; errors="ignore" in drop handles this
    preds_dataops = learner.predict({"visits": X_test_raw})
    csv_dataops = save_submission(X_test_raw, preds_dataops, "06_dataops")
    all_results.append({"key": "06_dataops", **{k: v for k, v in res_dataops.items()
                                                  if k != "_report"},
                         "csv": csv_dataops.name})

    # =========================================================================
    # Summary — sorted by mean RMSE (primary Kaggle metric)
    # =========================================================================
    # Include previously-computed results from results_rmse.csv
    prev = pd.read_csv(EXPERIMENTS / "results_rmse.csv")
    prev_rows = []
    for _, row in prev.iterrows():
        key = row["experiment"]
        if key not in {r["key"] for r in all_results}:
            prev_rows.append({
                "key": key,
                "mean_rmse": row["mean_rmse"],
                "std_rmse":  row["std_rmse"],
                "mean_mae":  float("nan"),   # not stored in that file
                "std_mae":   float("nan"),
                "csv": f"{key}.csv",
            })

    combined = prev_rows + all_results
    combined_sorted = sorted(combined, key=lambda r: r["mean_rmse"])

    print("\n\n" + "=" * 78)
    print("FINAL SUMMARY — all experiments, sorted by mean RMSE")
    print("=" * 78)
    print(f"{'Key':<40} {'Mean RMSE':>10} {'Std':>7}  {'Mean MAE':>9} {'Std':>7}")
    print("-" * 78)
    for r in combined_sorted:
        mae_str = f"{r['mean_mae']:>9.4f} {r['std_mae']:>7.4f}" \
                  if not math.isnan(r["mean_mae"]) else f"{'—':>9}   —   "
        print(f"{r['key']:<40} {r['mean_rmse']:>10.4f} {r['std_rmse']:>7.4f}  {mae_str}")
    print("=" * 78)

    best = combined_sorted[0]
    print(f"\nBest model by RMSE: {best['key']}")
    print(f"  Mean RMSE = {best['mean_rmse']:.4f} +/- {best['std_rmse']:.4f}")
    if not math.isnan(best["mean_mae"]):
        print(f"  Mean MAE  = {best['mean_mae']:.4f} +/- {best['std_mae']:.4f}")

    # =========================================================================
    # Save updated results
    # =========================================================================
    results_rows = []
    for r in combined:
        row = {
            "experiment": r["key"],
            "mean_rmse":  round(r["mean_rmse"], 6),
            "std_rmse":   round(r["std_rmse"], 6),
        }
        if "fold_rmse" in r:
            for i, v in enumerate(r["fold_rmse"], 1):
                row[f"fold_{i}_rmse"] = round(v, 6)
        if not math.isnan(r.get("mean_mae", float("nan"))):
            row["mean_mae"] = round(r["mean_mae"], 6)
            row["std_mae"]  = round(r["std_mae"], 6)
            if "fold_mae" in r:
                for i, v in enumerate(r["fold_mae"], 1):
                    row[f"fold_{i}_mae"] = round(v, 6)
        row["csv_file"] = r.get("csv", "")
        results_rows.append(row)

    out_path = EXPERIMENTS / "results_all.csv"
    pd.DataFrame(results_rows).to_csv(out_path, index=False)
    print(f"\nFull results saved: experiments/results_all.csv")

    # =========================================================================
    # CSV verification report
    # =========================================================================
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print("\nCSV verification:")
    print(f"  {'File':<50} {'Rows':>5}  {'Cols OK':>7}  {'No NaN':>6}  {'Idx OK':>6}")
    print("  " + "-" * 74)

    verified_keys = {r["key"] for r in all_results}
    verified_keys |= {r["key"] for r in prev_rows}

    for r in combined:
        csv_name = r.get("csv", f"{r['key']}.csv")
        csv_path = SUBMISSIONS / csv_name
        if not csv_path.exists():
            print(f"  {csv_name:<50}  MISSING")
            continue
        sub = pd.read_csv(csv_path)
        cols_ok  = list(sub.columns) == ["Index", "target"]
        no_nan   = sub["target"].notna().all()
        finite   = all(math.isfinite(v) for v in sub["target"])
        idx_ok   = sub["Index"].equals(sample["Index"])
        rows_ok  = len(sub) == len(sample)
        status   = "PASS" if all([cols_ok, no_nan, finite, idx_ok, rows_ok]) else "FAIL"
        print(f"  {csv_name:<50} {len(sub):>5}  {str(cols_ok):>7}  {str(no_nan and finite):>6}  "
              f"{str(idx_ok):>6}  {status}")

    print("\nDone. No Skore uploads were performed.")


if __name__ == "__main__":
    main()
