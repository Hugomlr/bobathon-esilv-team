"""
Bobathon ESILV — Advanced Experiments
=======================================

Runs a systematic set of experiments to minimize RMSE for MDS-UPDRS prediction.
Uses GroupKFold(patient_id) to avoid patient leakage.

Experiments:
  exp_001_dummy               — DummyRegressor baseline
  exp_002_ridge               — Ridge regression
  exp_003_histgbm_default     — HistGradientBoosting default
  exp_004_histgbm_tuned       — HistGradientBoosting (existing best params)
  exp_005_histgbm_feateng     — HistGBM + temporal features
  exp_006_extratrees          — ExtraTreesRegressor
  exp_007_randomforest        — RandomForestRegressor
  exp_008_xgboost             — XGBoost (if available)
  exp_009_lightgbm            — LightGBM (if available)
  exp_010_svr                 — SVR (linear kernel)
  exp_011_histgbm_optuna      — HistGBM Optuna-tuned
  exp_012_ensemble            — Weighted ensemble of best models

Usage (from repo root, venv active):
  python experiments/run_advanced_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
PREDICTIONS_DIR = EXPERIMENTS_DIR / "predictions"
OOF_DIR = EXPERIMENTS_DIR / "oof_predictions"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

for d in [SUBMISSIONS_DIR, RESULTS_DIR, PREDICTIONS_DIR, OOF_DIR, CONFIGS_DIR]:
    d.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]

LOG_PATH = EXPERIMENTS_DIR / "experiment_log.csv"


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Build all engineered features. Works for train and test.

    IMPORTANT: Preserves original row order so that the returned DataFrame
    stays aligned with the `groups` and `y` arrays passed to GroupKFold.
    Longitudinal features are computed using the *sorted* order within each
    patient group but then reindexed back to the original row order.
    """
    d = df.copy()  # do NOT sort — keep original row order

    # Basic derived features (order-independent)
    d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
    d["on_off_diff"] = d["off"] - d["on"]
    d["on_off_mean"] = (d["on"] + d["off"]) / 2.0
    d["on_off_ratio"] = d["on"] / (d["off"] + 1e-6)

    # Longitudinal features: sort within patient, compute, then reindex
    orig_index = d.index
    d_sorted = d.sort_values(["patient_id", "age"])
    d_sorted["visit_num"] = d_sorted.groupby("patient_id").cumcount()
    d_sorted["n_visits"] = d_sorted.groupby("patient_id")["patient_id"].transform("count")
    d_sorted["age_first_visit"] = d_sorted.groupby("patient_id")["age"].transform("min")
    d_sorted["age_last_visit"] = d_sorted.groupby("patient_id")["age"].transform("max")
    d_sorted["time_from_first_visit"] = d_sorted["age"] - d_sorted["age_first_visit"]
    d_sorted["visit_frac"] = d_sorted["visit_num"] / (d_sorted["n_visits"] - 1).clip(lower=1)

    # Copy back to original order using the shared index
    for col in ["visit_num", "n_visits", "age_first_visit", "age_last_visit",
                "time_from_first_visit", "visit_frac"]:
        d[col] = d_sorted[col].reindex(orig_index)

    # Interaction features
    d["age_x_ledd"] = d["age"] * d["ledd"] / 1000.0
    d["duration_x_ledd"] = d["disease_duration"] * d["ledd"] / 1000.0

    # Encode categorical
    for col in CATEGORICAL_COLS:
        d[col] = d[col].astype("category").cat.codes

    # Drop identifiers (keep original Index for test submission)
    drop_cols = ["patient_id"]
    if "target" in d.columns:
        drop_cols.append("target")
    if is_train:
        drop_cols.append("Index")
    else:
        # Keep Index for submission alignment
        pass

    return d.drop(columns=drop_cols)


def engineer_features_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature set used in original experiments (for comparison)."""
    d = df.copy()
    d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
    d["on_off_diff"] = d["off"] - d["on"]
    d["on_off_mean"] = (d["on"] + d["off"]) / 2.0
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    d[CATEGORICAL_COLS] = enc.fit_transform(d[CATEGORICAL_COLS].astype(str))
    drop = ["patient_id", "Index"]
    if "target" in d.columns:
        drop.append("target")
    return d.drop(columns=drop)


# ── Cross-validation engine ────────────────────────────────────────────────────

def cv_evaluate(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    gkf: GroupKFold,
    *,
    need_imputation: bool = False,
) -> dict:
    """
    Run GroupKFold CV. Returns fold metrics + OOF predictions.
    """
    oof_pred = np.zeros(len(y))
    fold_rmse = []
    fold_mae = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        m = clone(model)
        if need_imputation:
            imp = SimpleImputer(strategy="median")
            X_tr_arr = imp.fit_transform(X_tr)
            X_val_arr = imp.transform(X_val)
            m.fit(X_tr_arr, y_tr)
            preds = m.predict(X_val_arr)
        else:
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)

        oof_pred[val_idx] = preds
        fold_rmse.append(float(root_mean_squared_error(y_val, preds)))
        fold_mae.append(float(mean_absolute_error(y_val, preds)))

    return {
        "fold_rmse": fold_rmse,
        "fold_mae": fold_mae,
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_rmse": float(np.std(fold_rmse)),
        "mean_mae": float(np.mean(fold_mae)),
        "std_mae": float(np.std(fold_mae)),
        "oof_pred": oof_pred,
    }


# ── Experiment runner ──────────────────────────────────────────────────────────

def run_experiment(
    key: str,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    test_index: pd.Series,
    groups: np.ndarray,
    gkf: GroupKFold,
    *,
    need_imputation: bool = False,
    description: str = "",
) -> dict:
    """Evaluate one model, save OOF + test predictions, return metadata."""
    t0 = time.perf_counter()
    print(f"\n{'='*65}")
    print(f"Experiment : {key}")
    print(f"Model      : {model.__class__.__name__}")
    if description:
        print(f"Desc       : {description}")
    print(f"{'='*65}")

    cv_results = cv_evaluate(
        model, X_train, y_train, groups, gkf,
        need_imputation=need_imputation,
    )

    for i, (r, m) in enumerate(zip(cv_results["fold_rmse"], cv_results["fold_mae"]), 1):
        print(f"  Fold {i}: RMSE={r:.4f}  MAE={m:.4f}")

    mean_r = cv_results["mean_rmse"]
    std_r = cv_results["std_rmse"]
    mean_m = cv_results["mean_mae"]
    print(f"\n  Mean RMSE : {mean_r:.4f} ± {std_r:.4f}")
    print(f"  Mean MAE  : {mean_m:.4f}")

    # Final model on all train data
    if need_imputation:
        final = make_pipeline(SimpleImputer(strategy="median"), clone(model))
    else:
        final = clone(model)
    final.fit(X_train, y_train)
    test_preds = final.predict(X_test)

    elapsed = time.perf_counter() - t0

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "oof_pred": cv_results["oof_pred"],
        "target": y_train,
    })
    oof_path = OOF_DIR / f"{key}_oof.csv"
    oof_df.to_csv(oof_path, index=False)

    # Save test predictions
    test_df = pd.DataFrame({"Index": test_index, "target": test_preds})
    pred_path = PREDICTIONS_DIR / f"{key}_test.csv"
    test_df.to_csv(pred_path, index=False)

    # Save submission (same as predictions)
    sub_path = SUBMISSIONS_DIR / f"{key}.csv"
    test_df.to_csv(sub_path, index=False)

    # Save config
    try:
        params = {k: v for k, v in model.get_params().items() if not callable(v)}
    except Exception:
        params = {}
    cfg = {
        "key": key,
        "model": model.__class__.__name__,
        "params": params,
        "description": description,
        "need_imputation": need_imputation,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
    }
    cfg_path = CONFIGS_DIR / f"{key}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    print(f"  Time      : {elapsed:.1f}s")
    print(f"  OOF saved : {oof_path.relative_to(REPO_ROOT)}")
    print(f"  Test saved: {pred_path.relative_to(REPO_ROOT)}")

    result = {
        "experiment": key,
        "model": model.__class__.__name__,
        "description": description,
        "mean_rmse": round(mean_r, 6),
        "std_rmse": round(std_r, 6),
        "mean_mae": round(mean_m, 6),
        "std_mae": round(cv_results["std_mae"], 6),
        "elapsed_s": round(elapsed, 1),
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "oof_path": str(oof_path.relative_to(REPO_ROOT)),
        "test_pred_path": str(pred_path.relative_to(REPO_ROOT)),
        "submission_path": str(sub_path.relative_to(REPO_ROOT)),
        "skore_status": "pending",
    }
    for i, r in enumerate(cv_results["fold_rmse"], 1):
        result[f"fold_{i}_rmse"] = round(r, 6)
    for i, m in enumerate(cv_results["fold_mae"], 1):
        result[f"fold_{i}_mae"] = round(m, 6)

    result["oof_pred"] = cv_results["oof_pred"]  # keep for ensemble

    return result


# ── Optuna tuning ──────────────────────────────────────────────────────────────

def tune_histgbm_optuna(X, y, groups, gkf, n_trials=50):
    """Tune HistGradientBoostingRegressor with Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [Optuna] Not installed, skipping tuning.")
        return None

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "random_state": RANDOM_STATE,
        }
        m = HistGradientBoostingRegressor(**params)
        rmse_list = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            mc = clone(m)
            mc.fit(X.iloc[tr_idx], y[tr_idx])
            pred = mc.predict(X.iloc[val_idx])
            rmse_list.append(float(root_mean_squared_error(y[val_idx], pred)))
        return np.mean(rmse_list)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best = study.best_params
    print(f"  [Optuna] Best RMSE: {study.best_value:.4f}")
    print(f"  [Optuna] Best params: {best}")
    return best


# ── Ensemble ───────────────────────────────────────────────────────────────────

def build_ensemble(results_with_oof: list[dict], y_train: np.ndarray,
                   test_preds_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build weighted ensemble from OOF predictions.
    Optimize weights to minimize OOF RMSE using scipy.
    """
    from scipy.optimize import minimize

    oof_list = [r["oof_pred"] for r in results_with_oof]
    keys = [r["experiment"] for r in results_with_oof]
    n = len(oof_list)

    def neg_rmse(w):
        w = np.array(w)
        w = np.clip(w, 0, None)
        w = w / w.sum()
        ensemble_oof = sum(wi * o for wi, o in zip(w, oof_list))
        return float(root_mean_squared_error(y_train, ensemble_oof))

    # Start with equal weights
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    res = minimize(neg_rmse, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 200})

    weights = np.clip(res.x, 0, None)
    weights /= weights.sum()
    print(f"  [Ensemble] OOF RMSE: {res.fun:.4f}")
    for k, w in zip(keys, weights):
        print(f"    {k}: {w:.3f}")

    # OOF ensemble
    oof_ensemble = sum(w * o for w, o in zip(weights, oof_list))

    # Test ensemble
    test_ensemble = sum(
        w * test_preds_map[k] for w, k in zip(weights, keys)
    )

    return oof_ensemble, test_ensemble, weights


# ── Skore push ─────────────────────────────────────────────────────────────────

def push_to_skore(key: str, model, X_train, y_train) -> str | None:
    """Attempt to push EstimatorReport to Skore Hub."""
    try:
        import skore
        cfg = json.loads((REPO_ROOT / ".skore").read_text())
        os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
        os.environ.setdefault("SKORE_HUB_URI", cfg.get("hub_url", ""))
        skore.login(mode="hub")
        project = skore.Project(
            name="bobathon-esilv", mode="hub", workspace=cfg["workspace"]
        )
        t0 = time.perf_counter()
        report = skore.evaluate(clone(model), X_train, y_train)
        project.put(key, report)
        url = (
            f"https://skore.probabl.ai/{cfg['workspace']}/projects/"
            f"bobathon-esilv/reports/{key}"
        )
        print(f"  [Skore] {key} pushed in {time.perf_counter()-t0:.0f}s → {url}")
        return url
    except Exception as exc:
        print(f"  [Skore] {key} failed: {type(exc).__name__}: {exc}")
        return None


# ── Log utils ──────────────────────────────────────────────────────────────────

def log_results(results: list[dict]) -> None:
    """Append / update experiment log CSV."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "oof_pred"}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(LOG_PATH, index=False)
    print(f"\nExperiment log saved: {LOG_PATH.relative_to(REPO_ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Bobathon ESILV — Advanced Experiments")
    print("=" * 65)

    # Load data
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    assert (X_train_raw["Index"] == y_train_raw["Index"]).all()

    y = y_train_raw["target"].values
    groups = X_train_raw["patient_id"].values
    test_index = X_test_raw["Index"]

    print(f"Train: {len(X_train_raw):,} rows | {X_train_raw['patient_id'].nunique():,} patients")
    print(f"Test : {len(X_test_raw):,} rows  | {X_test_raw['patient_id'].nunique():,} patients")

    gkf = GroupKFold(n_splits=N_SPLITS)

    # Feature sets
    print("\nEngineering features...")
    # engineer_features needs X merged with target to drop 'target' column;
    # we pass X_train_raw (no target col) — is_train=True drops Index+patient_id
    X_full = engineer_features(X_train_raw, is_train=True)

    # For test: is_train=False keeps Index column so we can extract it
    X_test_full_with_idx = engineer_features(X_test_raw, is_train=False)
    test_index_full = X_test_full_with_idx["Index"].reset_index(drop=True)
    X_test_full = X_test_full_with_idx.drop(columns=["Index"]).reset_index(drop=True)

    # Minimal feature set (replicates original)
    X_min = engineer_features_minimal(X_train_raw)
    X_test_min = engineer_features_minimal(X_test_raw)

    print(f"Full feature set: {X_full.shape[1]} columns")
    print(f"Minimal feature set: {X_min.shape[1]} columns")

    results: list[dict] = []
    common_full = dict(
        X_train=X_full,
        y_train=y,
        X_test=X_test_full,
        test_index=test_index_full,
        groups=groups,
        gkf=gkf,
    )
    common_min = dict(
        X_train=X_min,
        y_train=y,
        X_test=X_test_min,
        test_index=test_index,
        groups=groups,
        gkf=gkf,
    )

    # ── exp_001: Dummy baseline ──────────────────────────────────────────────
    results.append(run_experiment(
        "exp_001_dummy",
        DummyRegressor(strategy="mean"),
        description="Dummy baseline (predict mean)",
        **common_min,
    ))

    # ── exp_002: Ridge ────────────────────────────────────────────────────────
    results.append(run_experiment(
        "exp_002_ridge",
        Ridge(alpha=1.0),
        description="Ridge regression, minimal features",
        need_imputation=True,
        **common_min,
    ))

    # ── exp_003: HistGBM default ──────────────────────────────────────────────
    results.append(run_experiment(
        "exp_003_histgbm_default",
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        description="HistGBM default params, minimal features",
        **common_min,
    ))

    # ── exp_004: HistGBM tuned (existing best) ────────────────────────────────
    best_params_existing = dict(
        max_iter=500, learning_rate=0.05, max_depth=5,
        min_samples_leaf=20, l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    results.append(run_experiment(
        "exp_004_histgbm_tuned",
        HistGradientBoostingRegressor(**best_params_existing),
        description="HistGBM tuned (existing best params), minimal features",
        **common_min,
    ))

    # ── exp_005: HistGBM + full feature engineering ───────────────────────────
    results.append(run_experiment(
        "exp_005_histgbm_feateng",
        HistGradientBoostingRegressor(**best_params_existing),
        description="HistGBM tuned + full feature engineering (temporal features)",
        **common_full,
    ))

    # ── exp_006: ExtraTrees + full features ───────────────────────────────────
    results.append(run_experiment(
        "exp_006_extratrees",
        ExtraTreesRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        description="ExtraTreesRegressor + full feature engineering",
        need_imputation=True,
        **common_full,
    ))

    # ── exp_007: RandomForest + full features ─────────────────────────────────
    results.append(run_experiment(
        "exp_007_randomforest",
        RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        description="RandomForestRegressor + full feature engineering",
        need_imputation=True,
        **common_full,
    ))

    # ── exp_008: XGBoost (if available) ──────────────────────────────────────
    try:
        from xgboost import XGBRegressor
        results.append(run_experiment(
            "exp_008_xgboost",
            XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                tree_method="hist", device="cpu",
                n_jobs=-1, random_state=RANDOM_STATE,
                verbosity=0,
            ),
            description="XGBoost + full feature engineering",
            need_imputation=True,
            **common_full,
        ))
    except ImportError:
        print("\n[SKIP] XGBoost not available")

    # ── exp_009: LightGBM (if available) ─────────────────────────────────────
    try:
        from lightgbm import LGBMRegressor
        results.append(run_experiment(
            "exp_009_lightgbm",
            LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_STATE,
                verbose=-1,
            ),
            description="LightGBM + full feature engineering",
            **common_full,
        ))
    except ImportError:
        print("\n[SKIP] LightGBM not available")

    # ── exp_010: CatBoost (if available) ─────────────────────────────────────
    try:
        from catboost import CatBoostRegressor
        results.append(run_experiment(
            "exp_010_catboost",
            CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=6,
                l2_leaf_reg=3.0, random_seed=RANDOM_STATE, verbose=0,
            ),
            description="CatBoost + full feature engineering",
            need_imputation=True,
            **common_full,
        ))
    except ImportError:
        print("\n[SKIP] CatBoost not available")

    # ── exp_011: Optuna-tuned HistGBM ─────────────────────────────────────────
    print("\nRunning Optuna hyperparameter search (50 trials)...")
    best_optuna = tune_histgbm_optuna(X_full, y, groups, gkf, n_trials=50)
    if best_optuna is not None:
        results.append(run_experiment(
            "exp_011_histgbm_optuna",
            HistGradientBoostingRegressor(**best_optuna),
            description="HistGBM + Optuna 50-trial HPO + full feature engineering",
            **common_full,
        ))

    # ── exp_012: Weighted ensemble ────────────────────────────────────────────
    # Select experiments that used full feature set for ensemble
    ensemble_keys = [
        r for r in results
        if r["experiment"] in {
            "exp_005_histgbm_feateng", "exp_006_extratrees", "exp_007_randomforest",
            "exp_008_xgboost", "exp_009_lightgbm", "exp_010_catboost",
            "exp_011_histgbm_optuna",
        }
        and "oof_pred" in r
    ]
    if len(ensemble_keys) >= 2:
        print("\nBuilding weighted ensemble...")
        # Load test preds for each
        test_preds_map = {}
        for r in ensemble_keys:
            pred_df = pd.read_csv(PREDICTIONS_DIR / f"{r['experiment']}_test.csv")
            test_preds_map[r["experiment"]] = pred_df["target"].values

        oof_ens, test_ens, weights = build_ensemble(ensemble_keys, y, test_preds_map)
        ens_rmse = float(root_mean_squared_error(y, oof_ens))
        ens_mae = float(mean_absolute_error(y, oof_ens))

        print(f"  Ensemble OOF RMSE: {ens_rmse:.4f}")
        print(f"  Ensemble OOF MAE : {ens_mae:.4f}")

        # Save ensemble
        oof_df = pd.DataFrame({"oof_pred": oof_ens, "target": y})
        oof_df.to_csv(OOF_DIR / "exp_012_ensemble_oof.csv", index=False)

        test_df = pd.DataFrame({"Index": test_index_full, "target": test_ens})
        test_df.to_csv(PREDICTIONS_DIR / "exp_012_ensemble_test.csv", index=False)
        test_df.to_csv(SUBMISSIONS_DIR / "exp_012_ensemble.csv", index=False)

        ens_result = {
            "experiment": "exp_012_ensemble",
            "model": "WeightedEnsemble",
            "description": f"Weighted ensemble of {len(ensemble_keys)} models",
            "mean_rmse": round(ens_rmse, 6),
            "std_rmse": 0.0,
            "mean_mae": round(ens_mae, 6),
            "std_mae": 0.0,
            "elapsed_s": 0.0,
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "oof_path": str((OOF_DIR / "exp_012_ensemble_oof.csv").relative_to(REPO_ROOT)),
            "test_pred_path": str((PREDICTIONS_DIR / "exp_012_ensemble_test.csv").relative_to(REPO_ROOT)),
            "submission_path": str((SUBMISSIONS_DIR / "exp_012_ensemble.csv").relative_to(REPO_ROOT)),
            "skore_status": "skipped",
            "oof_pred": oof_ens,
        }
        results.append(ens_result)

    # ── Summary ───────────────────────────────────────────────────────────────
    header = "\n" + "=" * 72
    print(header)
    print("EXPERIMENT SUMMARY — RMSE (5-fold GroupKFold by patient_id)")
    print("=" * 72)
    print(f"{'Experiment':<42} {'Mean RMSE':>10} {'±':>3} {'Std':>8} {'MAE':>8}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["mean_rmse"]):
        print(
            f"{r['experiment']:<42}"
            f"{r['mean_rmse']:>10.4f}"
            f"  ±"
            f"{r['std_rmse']:>8.4f}"
            f"{r['mean_mae']:>8.4f}"
        )
    print("=" * 72)

    best = min(results, key=lambda x: x["mean_rmse"])
    print(f"\nBEST : {best['experiment']} ({best['model']})")
    print(f"RMSE : {best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}")
    print(f"MAE  : {best['mean_mae']:.4f}")

    # Save log
    log_results(results)

    # Try Skore push for best model
    print("\nAttempting Skore Hub push for best model...")
    try:
        if best.get("experiment") in {
            "exp_005_histgbm_feateng", "exp_011_histgbm_optuna"
        }:
            cfg_path = CONFIGS_DIR / f"{best['experiment']}.json"
            cfg = json.loads(cfg_path.read_text())
            from sklearn.base import clone as _clone
            mdl_cls = HistGradientBoostingRegressor
            url = push_to_skore(
                best["experiment"],
                mdl_cls(**{k: v for k, v in cfg["params"].items() if v is not None}),
                X_full,
                y,
            )
            if url:
                best["skore_status"] = url
    except Exception as exc:
        print(f"  [Skore] push failed: {exc}")

    # Verify submission files
    print("\nSubmission file verification:")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    for r in results:
        sub_path = SUBMISSIONS_DIR / f"{r['experiment']}.csv"
        if not sub_path.exists():
            print(f"  {r['experiment']}.csv : MISSING")
            continue
        sub = pd.read_csv(sub_path)
        ok_cols = list(sub.columns) == ["Index", "target"]
        ok_rows = len(sub) == len(sample_sub)
        ok_no_nan = sub["target"].notna().all()
        ok_index = set(sub["Index"]) == set(sample_sub["Index"])
        status = "PASS" if all([ok_cols, ok_rows, ok_no_nan, ok_index]) else "FAIL"
        print(
            f"  {r['experiment']}.csv : {status}"
            f" (rows={len(sub)}, no_nan={ok_no_nan}, index_ok={ok_index})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
