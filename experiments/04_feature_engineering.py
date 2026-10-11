# %% [markdown]
# # 04 — Feature engineering + HistGradientBoostingRegressor
#
# Adds seven hand-crafted feature groups on top of the raw columns:
#   1. disease_duration (stateless, before mark_as_X)
#   2. visit_number + time_since_first_visit (stateless, before mark_as_X)
#   3. patient_on_mean + patient_off_mean (train-fitted, after mark_as_X)
#   4. is_off_missing + is_ledd_missing (stateless, before mark_as_X)
#   5. log_time_since_intake_on/off (stateless, before mark_as_X)
# GroupKFold(n_splits=5) on patient_id is baked into the DataOps graph.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


# ── Feature functions ─────────────────────────────────────────────────────────

def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless features derived from demographics — no information from target.

    Safe to compute before mark_as_X. Row order is deliberately preserved
    (no sort_values) so that the DataOps CV split indices stay aligned with
    the enriched DataFrame's positional order.
    """
    out = df.copy()

    # 1. Disease progression duration
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]

    # 2. Missingness flags — explicit binary indicators
    out["is_off_missing"] = out["off"].isna().astype(int)
    out["is_ledd_missing"] = out["ledd"].isna().astype(int)

    # 3. Log-transformed drug intake delays (log1p handles NaN→0 via fillna)
    out["log_time_since_intake_on"] = np.log1p(out["time_since_intake_on"].fillna(0))
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"].fillna(0))

    # 4. Visit ordering — rank within patient by age (no sort_values to avoid
    #    reordering rows and misaligning CV split indices)
    out["visit_number"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(int)
    )
    out["time_since_first_visit"] = (
        out["age"] - out.groupby("patient_id")["age"].transform("min")
    )

    return out


class PatientBaselineTransformer(BaseEstimator, TransformerMixin):
    """Per-patient mean on/off scores, fitted on training rows only.

    Uses patient_id (which must still be present in X at this step) to
    compute group means. Falls back to the training global mean for
    patients unseen in training (holdout patients in the test split).
    Drops patient_id from the output so TableVectorizer never sees it.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.on_mean_ = X.groupby("patient_id")["on"].mean()
        self.off_mean_ = X.groupby("patient_id")["off"].mean()
        self.global_on_mean_ = float(X["on"].mean())
        self.global_off_mean_ = float(X["off"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["patient_on_mean"] = (
            X["patient_id"].map(self.on_mean_).fillna(self.global_on_mean_)
        )
        out["patient_off_mean"] = (
            X["patient_id"].map(self.off_mean_).fillna(self.global_off_mean_)
        )
        # Drop patient_id here so it never reaches TableVectorizer.
        return out.drop(columns=["patient_id"])


# ── DataOps graph ─────────────────────────────────────────────────────────────
data = skrub.var("visits", visits)

# groups DataOp — evaluated at the same time as X and y by skore.
groups = data["patient_id"]

# Stateless enrichment before the CV split boundary.
enriched_op = data.skb.apply_func(add_stateless_features)

# Keep patient_id in X so PatientBaselineTransformer can use it.
# Drop only target; patient_id is consumed and dropped inside the transformer.
X_op = (
    enriched_op
    .drop(columns=["target"], errors="ignore")
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)

y_op = data["target"].skb.mark_as_y()

# PatientBaselineTransformer is fitted on each training fold (no_wrap=True
# because it takes a full DataFrame, not individual columns).
with_baselines = X_op.skb.apply(PatientBaselineTransformer(), no_wrap=True)

# TableVectorizer handles categoricals + remaining NaN in one pass.
# HGBR handles any leftover NaN natively.
encoded = with_baselines.skb.apply(skrub.TableVectorizer())
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
project.put("04_feature_engineering", report)

# ── Kaggle submission ─────────────────────────────────────────────────────────
fitted_learner = predictions.skb.make_learner()
fitted_learner.fit({"visits": visits})
preds = fitted_learner.predict({"visits": X_test})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": preds}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "04_feature_engineering.csv")
print(
    f"\nSubmission written: {submissions_dir / '04_feature_engineering.csv'}"
    f"  shape={submission.shape}"
)
