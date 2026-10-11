# %% [markdown]
# # Audit — 03_hgbr: TableVectorizer + HistGradientBoostingRegressor
#
# Read-only review of the skore report, plus feature-importance and
# residual-error analysis requested by the user.
# Never calls `skore.evaluate(...)` or `project.put(...)`.

# %%
import warnings

import numpy as np
import pandas as pd
import skrub
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.pipeline import Pipeline

import skore

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %% [markdown]
# ## Open the project

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project

# %% [markdown]
# ## List the available reports

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by `project.put()`:
# `https://skore.probabl.ai/chml/bobathon-esilv/cross-validations/61079`
# → id = `skore:report:cross-validation:61079`

# %%
REPORT_ID = "skore:report:cross-validation:61079"
report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## Prediction-error distribution (from stored report)
#
# `metrics.prediction_error()` assembles `y_true`, `y_pred`, and
# `residuals` across all CV folds — no refit needed.

# %%
report.metrics.prediction_error().frame()

# %% [markdown]
# ## Feature importance
#
# `inspection.permutation_importance` and `inspection.impurity_decrease`
# are not available for SkrubLearner-based reports in skore 0.26.
# We refit the equivalent plain sklearn Pipeline on the full training
# data to compute permutation importance on a 2000-row subsample.
# This is an analysis-only refit; nothing is pushed to the Project.

# %%
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
visits = X_train_raw.join(y_train_raw, how="left")

FEATURE_COLS = [c for c in visits.columns if c not in ("patient_id", "target")]
X_all = visits[FEATURE_COLS]
y_all = visits["target"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe = Pipeline([
        ("tv", skrub.TableVectorizer()),
        ("hgbr", HistGradientBoostingRegressor(random_state=0)),
    ])
    pipe.fit(X_all, y_all)

feature_names = pipe.named_steps["tv"].get_feature_names_out()
X_tv = pipe.named_steps["tv"].transform(X_all)
X_tv_arr = X_tv.values if hasattr(X_tv, "values") else X_tv

rng = np.random.default_rng(42)
idx = rng.choice(len(X_tv_arr), size=2000, replace=False)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    perm = sk_permutation_importance(
        pipe.named_steps["hgbr"],
        X_tv_arr[idx],
        y_all.values[idx],
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

importance_df = (
    pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)
importance_df.head(10)  # noqa: B018

# %% [markdown]
# ## Residual analysis
#
# Residuals = y_true − y_pred. Positive → underprediction, negative → overprediction.
# We segment by target quartile and cohort to identify where the largest errors occur.

# %%
y_pred_all = pipe.predict(X_all)
residuals = y_all.values - y_pred_all

analysis = visits[["cohort", "gene"]].copy()
analysis["y_true"] = y_all.values
analysis["y_pred"] = y_pred_all
analysis["residual"] = residuals
analysis["abs_residual"] = np.abs(residuals)
analysis["target_quartile"] = pd.qcut(
    analysis["y_true"], q=4, labels=["Q1 (0–26)", "Q2 (26–37)", "Q3 (37–49)", "Q4 (49+)"]
)

residual_by_quartile = (
    analysis.groupby("target_quartile", observed=True)[["residual", "abs_residual"]]
    .agg(["mean", "std"])
    .round(2)
)
residual_by_quartile  # noqa: B018

# %%
residual_by_cohort = (
    analysis.groupby("cohort")[["residual", "abs_residual"]]
    .agg(["mean", "std"])
    .round(2)
)
residual_by_cohort  # noqa: B018

# %%
# Top 1% worst errors
threshold = np.percentile(analysis["abs_residual"], 99)
worst = analysis[analysis["abs_residual"] > threshold]
{
    "n_worst_rows": int(len(worst)),
    "threshold_abs_residual": round(float(threshold), 1),
    "mean_y_true": round(float(worst["y_true"].mean()), 1),
    "mean_residual": round(float(worst["residual"].mean()), 1),
    "cohort_counts": worst["cohort"].value_counts().to_dict(),
    "gene_counts": worst["gene"].value_counts().to_dict(),
}  # noqa: B018

# %% [markdown]
# ## End of audit
#
# Feature importance and residual findings are summarised in the
# response to the user's analysis request.
