# %% [markdown]
# # 06 — Pharmacokinetic decay features + HistGradientBoostingRegressor
#
# Extends experiment 05 with two explicit drug-decay features that model
# levodopa concentration at the time of the clinical assessment:
#
#   ledd_decay_on  = ledd * exp(-0.5 * time_since_intake_on)
#   ledd_decay_off = ledd * exp(-0.5 * time_since_intake_off)
#
# NaNs in ledd or the intake-time columns propagate naturally through NumPy
# arithmetic; HGBR handles them natively so no imputation is needed.
# rater_id is retained in X and encoded by TableVectorizer (OHE) to capture
# inter-rater scoring bias.

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
    """Stateless engineered features; row order is never changed."""
    out = df.copy()

    # Disease progression
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]

    # Missingness flags
    out["is_off_missing"] = out["off"].isna().astype(int)
    out["is_ledd_missing"] = out["ledd"].isna().astype(int)

    # Log-transformed intake delays
    out["log_time_since_intake_on"] = np.log1p(out["time_since_intake_on"].fillna(0))
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"].fillna(0))

    # Pharmacokinetic decay: levodopa concentration proxy at assessment time.
    # NaNs in ledd or time_since_intake_* propagate naturally; HGBR handles them.
    out["ledd_decay_on"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_on"])
    out["ledd_decay_off"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_off"])

    return out


# ── DataOps graph ─────────────────────────────────────────────────────────────
data = skrub.var("visits", visits)

groups = data["patient_id"]

enriched_op = data.skb.apply_func(add_stateless_features)

# Drop patient_id and target; rater_id is kept and encoded by TableVectorizer.
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
project.put("06_pharmacokinetics", report)

# ── Kaggle submission ─────────────────────────────────────────────────────────
fitted_learner = predictions.skb.make_learner()
fitted_learner.fit({"visits": visits})
preds = fitted_learner.predict({"visits": X_test})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": preds}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "06_pharmacokinetics.csv")
print(
    f"\nSubmission written: {submissions_dir / '06_pharmacokinetics.csv'}"
    f"  shape={submission.shape}"
)
