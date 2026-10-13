"""
Execute all the logic of notebooks/bobathon_complete.ipynb as a plain Python
script, so it can run without nbconvert installed.

Run from repo root:
    python experiments/run_complete_notebook.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import skore
import skrub

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skrub import tabular_pipeline as skrub_tabular_pipeline

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR        = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_NAME = "bobathon-esilv"
N_SPLITS     = 5
RANDOM_STATE = 42

FEATURE_COLS     = ["sexM", "age_at_diagnosis", "age", "ledd",
                    "time_since_intake_on", "time_since_intake_off", "on", "off"]
CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
DROP_COLS        = ["patient_id", "Index"]

print(f"skore  {skore.__version__}")
print(f"skrub  {skrub.__version__}")
print(f"Repo   {REPO_ROOT}")

# ── Data loading ───────────────────────────────────────────────────────────────
print("\n[1] Loading data ...")
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub  = pd.read_csv(DATA_DIR / "sample_submission.csv")

assert (X_train_raw["Index"] == y_train_raw["Index"]).all()
visits = X_train_raw.merge(y_train_raw, on="Index")
y      = visits["target"]
overlap = set(X_train_raw["patient_id"]) & set(X_test_raw["patient_id"])
print(f"  Train visits: {len(X_train_raw):,}  |  Test visits: {len(X_test_raw):,}")
print(f"  Patients train/test: {X_train_raw['patient_id'].nunique():,} / "
      f"{X_test_raw['patient_id'].nunique():,}  |  Overlap: {len(overlap)}")

# ── Feature preparation ────────────────────────────────────────────────────────
print("\n[2] Preparing features ...")

X      = visits[FEATURE_COLS]
X_test = X_test_raw[FEATURE_COLS]

X_full      = visits.drop(columns=["Index", "patient_id", "target"])
X_test_full = X_test_raw.drop(columns=["Index", "patient_id"], errors="ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
    d["on_off_diff"]      = d["off"] - d["on"]
    d["on_off_mean"]      = (d["on"] + d["off"]) / 2.0
    return d.drop(columns=DROP_COLS)


X_feat_train = engineer_features(X_train_raw)
X_feat_test  = engineer_features(X_test_raw)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_feat_train[CATEGORICAL_COLS] = encoder.fit_transform(
    X_feat_train[CATEGORICAL_COLS].astype(str))
X_feat_test[CATEGORICAL_COLS] = encoder.transform(
    X_feat_test[CATEGORICAL_COLS].astype(str))

groups     = visits["patient_id"]
groups_raw = X_train_raw["patient_id"]
gkf        = GroupKFold(n_splits=N_SPLITS)

cv_splits_feat = list(gkf.split(X_feat_train, y, groups_raw))
cv_splits      = list(gkf.split(X, y, groups))
cv_splits_full = list(gkf.split(X_full, y, groups))

# Verify no leakage
for i, (tr, val) in enumerate(cv_splits):
    assert len(set(groups.iloc[tr]) & set(groups.iloc[val])) == 0
print(f"  GroupKFold({N_SPLITS}) by patient_id — no leakage confirmed.")

# ── Helper functions ───────────────────────────────────────────────────────────

def cv_eval(model, X_cv, y_cv, splits, *, need_imputation=False):
    fold_rmse, fold_mae = [], []
    t0 = time.perf_counter()
    for train_idx, val_idx in splits:
        X_tr  = X_cv.iloc[train_idx]
        X_val = X_cv.iloc[val_idx]
        y_tr  = y_cv.iloc[train_idx]
        y_val = y_cv.iloc[val_idx]
        m = clone(model)
        if need_imputation:
            imp   = SimpleImputer(strategy="median")
            X_tr  = imp.fit_transform(X_tr)
            X_val = imp.transform(X_val)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        fold_rmse.append(float(root_mean_squared_error(y_val, preds)))
        fold_mae.append(float(mean_absolute_error(y_val, preds)))
    elapsed = time.perf_counter() - t0
    return {
        "fold_rmse": fold_rmse, "fold_mae": fold_mae,
        "mean_rmse": float(np.mean(fold_rmse)), "std_rmse":  float(np.std(fold_rmse)),
        "mean_mae":  float(np.mean(fold_mae)),  "std_mae":   float(np.std(fold_mae)),
        "fit_time":  round(elapsed, 3),
    }


def print_cv(key, res):
    for i, (r, m) in enumerate(zip(res["fold_rmse"], res["fold_mae"]), 1):
        print(f"  Fold {i}: RMSE={r:.4f}  MAE={m:.4f}")
    print(f"  Mean RMSE : {res['mean_rmse']:.4f} ± {res['std_rmse']:.4f}")
    print(f"  Mean MAE  : {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")
    print(f"  Fit time  : {res['fit_time']:.1f}s")


def extract_skore_metrics(report):
    try:
        df = report.metrics.summarize().frame()
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        std_cols  = [c for c in df.columns if c.endswith("_std")]
        mc = mean_cols[0] if mean_cols else df.columns[0]
        sc = std_cols[0]  if std_cols  else df.columns[1]
        rmse_mean = float(df.loc["rmse", mc]) if "rmse" in df.index else float("nan")
        rmse_std  = float(df.loc["rmse", sc]) if "rmse" in df.index else float("nan")
        mae_mean  = float(df.loc["mae",  mc]) if "mae"  in df.index else float("nan")
        mae_std   = float(df.loc["mae",  sc]) if "mae"  in df.index else float("nan")
        fit_t     = float(df.loc["fit_time", mc]) if "fit_time" in df.index else float("nan")
        pred_t    = float(df.loc["predict_time", mc]) if "predict_time" in df.index else float("nan")
        return {"mean_rmse": rmse_mean, "std_rmse": rmse_std,
                "mean_mae": mae_mean, "std_mae": mae_std,
                "fit_time": fit_t, "pred_time": pred_t}
    except Exception as exc:
        print(f"  [Skore] metric extraction failed: {exc}")
        return {}


all_results: list[dict] = []
hub_report_urls: dict[str, str] = {}

# ── Skore Hub connection ────────────────────────────────────────────────────────
print("\n[3] Connecting to Skore Hub ...")
hub_project = None
workspace = None
try:
    cfg = json.loads((REPO_ROOT / ".skore").read_text())
    workspace = cfg["workspace"]
    skore.login(mode="hub")
    hub_project = skore.Project(name=PROJECT_NAME, mode="hub", workspace=workspace)
    print(f"  Hub connected: {hub_project}")
except Exception as exc:
    print(f"  WARNING: Hub unavailable ({type(exc).__name__}). Reports will NOT be pushed.",
          file=sys.stderr)


def push_report(key, report):
    if hub_project is None:
        return None
    try:
        hub_project.put(key, report)
        url = (f"https://skore.probabl.ai/{workspace}"
               f"/projects/{PROJECT_NAME}/reports/{key}")
        hub_report_urls[key] = url
        print(f"  [Hub] '{key}' pushed => {url}")
        return url
    except Exception as exc:
        print(f"  [Hub] '{key}' push FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


# ── 01_dummy ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("01_dummy  |  DummyRegressor(mean)  |  GroupKFold")
print("=" * 62)

dummy = DummyRegressor(strategy="mean")
res_dummy = cv_eval(dummy, X, y, cv_splits)
print_cv("01_dummy", res_dummy)

report_dummy = skore.evaluate(dummy, X, y, splitter=cv_splits)
# Already on Hub — skip re-push
if workspace:
    hub_report_urls["01_dummy"] = (
        f"https://skore.probabl.ai/{workspace}"
        f"/projects/{PROJECT_NAME}/reports/01_dummy"
    )
    print(f"  [Hub] '01_dummy' already on Hub => {hub_report_urls['01_dummy']}")

all_results.append({
    "key": "01_dummy", "model": "DummyRegressor(mean)",
    "feature_set": "A (numeric)", "validation": f"GroupKFold({N_SPLITS})",
    **res_dummy,
    "hub_url": hub_report_urls.get("01_dummy", ""),
    "notes": "Global mean baseline — already on Hub",
})

# ── 02_ridge ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("02_ridge  |  Ridge(alpha=1) + Imputer  |  GroupKFold")
print("=" * 62)

ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
res_ridge = cv_eval(ridge, X, y, cv_splits)
print_cv("02_ridge", res_ridge)

report_ridge = skore.evaluate(ridge, X, y, splitter=cv_splits)
push_report("02_ridge", report_ridge)

all_results.append({
    "key": "02_ridge", "model": "Ridge(alpha=1) + Imputer",
    "feature_set": "A (numeric)", "validation": f"GroupKFold({N_SPLITS})",
    **res_ridge,
    "hub_url": hub_report_urls.get("02_ridge", ""),
    "notes": "Linear baseline with median imputation",
})

# ── 03_hist_gradient_boosting ─────────────────────────────────────────────────
print("\n" + "=" * 62)
print("03_hist_gradient_boosting  |  HistGBM default  |  GroupKFold")
print("=" * 62)

hgbr = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
res_hgbr = cv_eval(hgbr, X, y, cv_splits)
print_cv("03_hist_gradient_boosting", res_hgbr)

report_hgbr = skore.evaluate(hgbr, X, y, splitter=cv_splits)
push_report("03_hist_gradient_boosting", report_hgbr)

all_results.append({
    "key": "03_hist_gradient_boosting", "model": "HistGBM(default, rs=42)",
    "feature_set": "A (numeric)", "validation": f"GroupKFold({N_SPLITS})",
    **res_hgbr,
    "hub_url": hub_report_urls.get("03_hist_gradient_boosting", ""),
    "notes": "NaN-native gradient boosting",
})

# ── 04_hist_gradient_boosting_tuned ──────────────────────────────────────────
print("\n" + "=" * 62)
print("04_hist_gradient_boosting_tuned  |  HistGBM tuned  |  GroupKFold")
print("=" * 62)

best_params = dict(max_iter=500, learning_rate=0.05, max_depth=5,
                   min_samples_leaf=20, l2_regularization=0.0,
                   random_state=RANDOM_STATE)
hgbr_tuned = HistGradientBoostingRegressor(**best_params)
res_tuned = cv_eval(hgbr_tuned, X_feat_train, y, cv_splits_feat)
print_cv("04_hist_gradient_boosting_tuned", res_tuned)

report_tuned = skore.evaluate(hgbr_tuned, X_feat_train, y, splitter=cv_splits_feat)
push_report("04_hist_gradient_boosting_tuned", report_tuned)

all_results.append({
    "key": "04_hist_gradient_boosting_tuned", "model": "HistGBM(tuned, rs=42)",
    "feature_set": "C (engineered + encoded)", "validation": f"GroupKFold({N_SPLITS})",
    **res_tuned,
    "hub_url": hub_report_urls.get("04_hist_gradient_boosting_tuned", ""),
    "notes": "Best HP from Day 1 random search; engineered features",
})

# ── 05_tabular_pipeline ───────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("05_tabular_pipeline  |  skrub tabular_pipeline  |  all cols  |  GroupKFold")
print("=" * 62)

model_tabular = skrub_tabular_pipeline("regressor")
res_tabular = cv_eval(model_tabular, X_full, y, cv_splits_full)
print_cv("05_tabular_pipeline", res_tabular)

report_tabular = skore.evaluate(model_tabular, X_full, y, splitter=cv_splits_full)
push_report("05_tabular_pipeline", report_tabular)

all_results.append({
    "key": "05_tabular_pipeline", "model": "skrub tabular_pipeline(regressor)",
    "feature_set": "B (all columns)", "validation": f"GroupKFold({N_SPLITS})",
    **res_tabular,
    "hub_url": hub_report_urls.get("05_tabular_pipeline", ""),
    "notes": "Auto-encoder + HistGBM via skrub; all non-id columns",
})

# ── 06_dataops ────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("06_dataops  |  skrub DataOps  |  GroupKFold baked in")
print("=" * 62)

data_op   = skrub.var("visits", visits)
groups_op = data_op["patient_id"]
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

report_dataops = skore.evaluate(pred_op)
sm = extract_skore_metrics(report_dataops)
print(f"  DataOps skore RMSE: {sm.get('mean_rmse', float('nan')):.4f} "
      f"± {sm.get('std_rmse', float('nan')):.4f}")
print(f"  DataOps skore MAE:  {sm.get('mean_mae', float('nan')):.4f} "
      f"± {sm.get('std_mae', float('nan')):.4f}")
push_report("06_dataops", report_dataops)

all_results.append({
    "key": "06_dataops", "model": "skrub DataOps + TableVectorizer + HistGBM",
    "feature_set": "B (all via DataOps)", "validation": f"GroupKFold({N_SPLITS}) baked in",
    "fold_rmse": [], "fold_mae": [],
    "mean_rmse": sm.get("mean_rmse", float("nan")),
    "std_rmse":  sm.get("std_rmse",  float("nan")),
    "mean_mae":  sm.get("mean_mae",  float("nan")),
    "std_mae":   sm.get("std_mae",   float("nan")),
    "fit_time":  sm.get("fit_time",  float("nan")),
    "hub_url":   hub_report_urls.get("06_dataops", ""),
    "notes": "DataOps graph with GroupKFold groups baked in",
})

# ── Summary ───────────────────────────────────────────────────────────────────
df_results = (pd.DataFrame(all_results)
              .sort_values("mean_rmse", ascending=True)
              .reset_index(drop=True))

print("\n\n" + "=" * 100)
print("MODEL COMPARISON — sorted by mean validation RMSE (ascending)")
print("=" * 100)
print(f"{'Key':<40} {'Mean RMSE':>10} {'Std':>7}  {'Mean MAE':>9} {'Std':>7}  {'Fit(s)':>7}")
print("-" * 100)
for _, r in df_results.iterrows():
    rmse_s = f"{r['mean_rmse']:.4f}" if not np.isnan(r['mean_rmse']) else "    —  "
    rstd_s = f"{r['std_rmse']:.4f}"  if not np.isnan(r['std_rmse'])  else "   —   "
    mae_s  = f"{r['mean_mae']:.4f}"  if not np.isnan(r['mean_mae'])  else "    —  "
    mstd_s = f"{r['std_mae']:.4f}"   if not np.isnan(r['std_mae'])   else "   —   "
    ft_s   = f"{r['fit_time']:.1f}"  if not np.isnan(r.get('fit_time', float('nan'))) else " —"
    print(f"{r['key']:<40} {rmse_s:>10} {rstd_s:>7}  {mae_s:>9} {mstd_s:>7}  {ft_s:>7}")
print("=" * 100)

best = df_results.iloc[0]
print(f"\nBest model: {best['key']}")
print(f"  RMSE: {best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}")
print(f"  MAE:  {best['mean_mae']:.4f} ± {best['std_mae']:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
rows = []
for r in all_results:
    row = {
        "experiment":  r["key"],
        "model":       r["model"],
        "feature_set": r["feature_set"],
        "mean_rmse":   round(r["mean_rmse"], 6),
        "std_rmse":    round(r["std_rmse"],  6),
        "mean_mae":    round(r["mean_mae"],  6),
        "std_mae":     round(r["std_mae"],   6),
        "fit_time":    r.get("fit_time", float("nan")),
        "hub_url":     r.get("hub_url", ""),
        "notes":       r.get("notes", ""),
    }
    for i, v in enumerate(r.get("fold_rmse", []), 1):
        row[f"fold_{i}_rmse"] = round(float(v), 6)
    for i, v in enumerate(r.get("fold_mae", []), 1):
        row[f"fold_{i}_mae"] = round(float(v), 6)
    rows.append(row)

df_out = pd.DataFrame(rows)
out_path = EXPERIMENTS_DIR / "results_complete.csv"
df_out.to_csv(out_path, index=False)
print(f"\nFull results saved: {out_path.relative_to(REPO_ROOT)}")

print("\nHub report URLs:")
print(f"  {'Key':<40} URL")
print("  " + "-" * 90)
for k, u in sorted(hub_report_urls.items()):
    print(f"  {k:<40} {u}")

print("\nDone.")
