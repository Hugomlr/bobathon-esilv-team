# %% [markdown]
# # 03 — HistGradientBoostingRegressor with skrub DataOps
#
# Uses the full feature set (including categoricals) via TableVectorizer.
# GroupKFold(n_splits=5) is baked into the graph via mark_as_X so patient
# groups are respected without any manual split pre-computation.

# %%
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skrub

import skore
from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
visits = X_train.join(y_train, how="left")

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

# ── DataOps graph ─────────────────────────────────────────────────────────────
# Root variable — the merged training table (or X_test at predict time).
data = skrub.var("visits", visits)

# groups must be a DataOp so it is evaluated at the same time as X and y.
groups = data["patient_id"]

# X: drop the id column, the row-index column, and the target.
# errors='ignore' lets the same graph work on X_test (no 'target' column).
X_op = (
    data.drop(columns=["patient_id", "target"], errors="ignore")
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)

# y: the regression target.
y_op = data["target"].skb.mark_as_y()

# Encoding: TableVectorizer handles categoricals, numerics, and NaNs in one pass.
# HGBR handles remaining NaN natively — no separate imputer needed.
encoded = X_op.skb.apply(skrub.TableVectorizer())
predictions = encoded.skb.apply(
    HistGradientBoostingRegressor(random_state=0), y=y_op
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
# skore.evaluate with a SkrubLearner takes data= (env-dict) not X/y.
# Omitting splitter= lets skore reuse the cv+split_kwargs baked into the graph.
learner = predictions.skb.make_learner()
report = skore.evaluate(learner, data={"visits": visits})
print("CV report:", report)
print("\nRMSE:")
print(report.metrics.rmse())

# ── Push to Skore Hub ─────────────────────────────────────────────────────────
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("03_hgbr", report)

# ── Kaggle submission ─────────────────────────────────────────────────────────
# Refit on all training data, then predict on X_test.
fitted_learner = predictions.skb.make_learner()
fitted_learner.fit({"visits": visits})
preds = fitted_learner.predict({"visits": X_test})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": preds}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "03_hgbr.csv")
print(f"\nSubmission written: {submissions_dir / '03_hgbr.csv'}  shape={submission.shape}")
