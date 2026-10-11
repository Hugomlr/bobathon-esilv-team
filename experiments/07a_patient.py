"""07a — stateless features, category dtypes, tuned HGBR, no TableVectorizer."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import skrub, skore
from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
visits = X_train.join(y_train, how="left")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

CAT_COLS = ["rater_id", "cohort", "gene"]
all_data = pd.concat([X_train, X_test], axis=0)
CAT_CATEGORIES = {c: sorted(all_data[c].dropna().unique().tolist()) for c in CAT_COLS}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]
    out["is_off_missing"] = out["off"].isna().astype(int)
    out["is_ledd_missing"] = out["ledd"].isna().astype(int)
    out["log_time_since_intake_on"] = np.log1p(out["time_since_intake_on"])
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"])
    out["ledd_decay_on"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_on"])
    out["ledd_decay_off"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_off"])
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c], categories=CAT_CATEGORIES[c])
    return out


HGBR = HistGradientBoostingRegressor(
    learning_rate=0.05, max_iter=600, min_samples_leaf=40,
    l2_regularization=1.0, early_stopping=False, random_state=0,
    categorical_features="from_dtype",
)

data = skrub.var("visits", visits)
groups = data["patient_id"]
enriched_op = data.skb.apply_func(add_features)
X_op = (
    enriched_op
    .drop(columns=["patient_id", "target"], errors="ignore")
    .skb.mark_as_X(cv=GroupKFold(n_splits=5), split_kwargs={"groups": groups})
)
y_op = data["target"].skb.mark_as_y()
predictions = X_op.skb.apply(HGBR, y=y_op)

learner = predictions.skb.make_learner()
report = skore.evaluate(learner, data={"visits": visits})
print("07a RMSE:", report.metrics.rmse())

cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("07a", report)

fitted = predictions.skb.make_learner()
fitted.fit({"visits": visits})
preds = fitted.predict({"visits": X_test})
sub = pd.DataFrame({"target": preds}, index=X_test.index)
sub.index.name = "Index"
(PROJECT_ROOT / "submissions").mkdir(exist_ok=True)
sub.to_csv(PROJECT_ROOT / "submissions" / "07a.csv")
print("07a submission written")
