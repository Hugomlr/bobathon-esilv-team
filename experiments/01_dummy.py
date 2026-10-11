# %% [markdown]
# # 01 — Dummy mean baseline
#
# Predicts the training-set mean of `target` for every visit.
# Establishes the floor RMSE that every subsequent model must beat.

# %%
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

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

# ── Model ─────────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")

# ── Evaluate ──────────────────────────────────────────────────────────────────
# Default splitter=0.2 (random 80/20 row holdout) — no splitter= argument.
report = skore.evaluate(dummy, X, y)
print("RMSE:", report.metrics.rmse())

# ── Push to Skore Hub ─────────────────────────────────────────────────────────
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)

# ── Kaggle submission ─────────────────────────────────────────────────────────
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

# Fit on all training data before predicting on test.
dummy_full = DummyRegressor(strategy="mean")
dummy_full.fit(X[FEATURE_COLS], y)

predictions = dummy_full.predict(X_test[FEATURE_COLS])

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "01_dummy.csv")
print(f"Submission written: {submissions_dir / '01_dummy.csv'}  shape={submission.shape}")
