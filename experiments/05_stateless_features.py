# %% [markdown]
# # 05 — Stateless feature engineering + HistGradientBoostingRegressor
#
# Extends experiment 03 with four stateless engineered features:
#   - disease_duration  = age - age_at_diagnosis
#   - is_off_missing    = binary flag for missing off score
#   - is_ledd_missing   = binary flag for missing ledd
#   - log_time_since_intake_on  = log1p(time_since_intake_on)
#   - log_time_since_intake_off = log1p(time_since_intake_off)
#
# All features are stateless (no .groupby on patient_id) so they apply
# identically to unseen test patients. Row order is preserved throughout
# to keep DataOps CV split indices aligned.

# %%
from pathlib import Path

import numpy as np
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


# ── Feature function ──────────────────────────────────────────────────────────

def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stateless engineered features; row order is never changed."""
    out = df.copy()
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]
    out["is_off_missing"] = out["off"].isna().astype(int)
    out["is_ledd_missing"] = out["ledd"].isna().astype(int)
    out["log_time_since_intake_on"] = np.log1p(out["time_since_intake_on"].fillna(0))
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"].fillna(0))
    return out


# ── DataOps graph ─────────────────────────────────────────────────────────────
data = skrub.var("visits", visits)

groups = data["patient_id"]

# Stateless enrichment before the CV split boundary — no leakage.
enriched_op = data.skb.apply_func(add_stateless_features)

X_op = (
    enriched_op
    .drop(columns=["patient_id", "target"], errors="ignore")
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)

y_op = data["target"].skb.mark_as_y()

encoded = X_op.skb.apply(skrub.TableVectorizer())
predictions = encoded.skb.apply(
    HistGradientBoostingRegressor(random_state=0), y=y_op
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
learner = predictions.skb.make_learner()
report = skore.evaluate(learner, data={"visits": visits})
print("CV report:", report)
print("\nRMSE:")
print(report.metrics.rmse())

# ── Push to Skore Hub ─────────────────────────────────────────────────────────
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("05_stateless_features", report)

# ── Kaggle submission ─────────────────────────────────────────────────────────
fitted_learner = predictions.skb.make_learner()
fitted_learner.fit({"visits": visits})
preds = fitted_learner.predict({"visits": X_test})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": preds}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "05_stateless_features.csv")
print(
    f"\nSubmission written: {submissions_dir / '05_stateless_features.csv'}"
    f"  shape={submission.shape}"
)
