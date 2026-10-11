# %% [markdown]
# # 02 — Ridge regression with patient-grouped cross-validation
#
# Compares a median-imputed Ridge against the dummy baseline using
# GroupKFold(n_splits=5) on patient_id so no patient leaks across folds.

# %%
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import skore
from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
visits = X_train.join(y_train, how="left")

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[FEATURE_COLS]
y = visits["target"]

# ── Grouped splitter ──────────────────────────────────────────────────────────
# GroupKFold.split() requires groups; skore calls split(X, y) without groups,
# so pre-compute the indices and pass them as a reusable list.
groups = visits["patient_id"]
cv = GroupKFold(n_splits=5)
split_indices = list(cv.split(X, y, groups=groups))

# ── Models ────────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")

ridge = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(alpha=1.0)),
    ]
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
# skore.evaluate with a dict recurses once per estimator, consuming a shared
# iterator.  Build one CrossValidationReport per model (each gets its own fresh
# iterator) then combine them into a ComparisonReport.
report_dummy = skore.CrossValidationReport(dummy, X, y, splitter=iter(split_indices))
report_ridge = skore.CrossValidationReport(ridge, X, y, splitter=iter(split_indices))
report = skore.ComparisonReport({"dummy": report_dummy, "ridge": report_ridge})
print("Comparison report:", report)
print("\nRMSE per model:")
print(report.metrics.rmse())

# ── Push to Skore Hub ─────────────────────────────────────────────────────────
# project.put only accepts EstimatorReport or CrossValidationReport, not
# ComparisonReport.  Push the Ridge CV report under '02_ridge'; the dummy
# baseline is already on Hub as '01_dummy'.
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", report_ridge)

# ── Kaggle submission ─────────────────────────────────────────────────────────
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

ridge_full = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(alpha=1.0)),
    ]
)
ridge_full.fit(X[FEATURE_COLS], y)

predictions = ridge_full.predict(X_test[FEATURE_COLS])

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "02_ridge.csv")
print(f"\nSubmission written: {submissions_dir / '02_ridge.csv'}  shape={submission.shape}")
