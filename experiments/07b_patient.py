"""07b — 07a + per-patient features computed via groupby (no row reordering)."""
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

    # ── stateless ──────────────────────────────────────────────────────────────
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]
    out["is_off_missing"] = out["off"].isna().astype(int)
    out["is_ledd_missing"] = out["ledd"].isna().astype(int)
    out["log_time_since_intake_on"] = np.log1p(out["time_since_intake_on"])
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"])
    out["ledd_decay_on"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_on"])
    out["ledd_decay_off"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_off"])
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c], categories=CAT_CATEGORIES[c])

    # ── patient features (groupby on feature columns only, no target) ──────────
    g = out.groupby("patient_id", sort=False)

    out["n_visits"] = g["age"].transform("count")
    out["age_first"] = g["age"].transform("min")
    out["time_since_first_visit"] = out["age"] - out["age_first"]

    for col in ["on", "off", "ledd"]:
        out[f"patient_{col}_mean"] = g[col].transform("mean")
        out[f"patient_{col}_std"] = g[col].transform("std")
        out[f"patient_{col}_min"] = g[col].transform("min")
        out[f"patient_{col}_max"] = g[col].transform("max")

    # leave-one-out patient mean of on and off
    for col in ["on", "off"]:
        s = g[col].transform("sum")
        n = g[col].transform("count")
        # LOO: (sum - x) / (count - 1); NaN when count==1 or x is NaN
        loo = (s - out[col]) / (n - 1)
        loo[(n <= 1) | out[col].isna()] = np.nan
        out[f"patient_{col}_loo_mean"] = loo

    # interpolated on/off at current age from other visits in the same patient
    for col in ["on", "off"]:
        vals = []
        for pid, grp in out.groupby("patient_id", sort=False):
            valid = grp[[col, "age"]].dropna()
            if len(valid) < 2:
                vals.append(pd.Series(np.nan, index=grp.index))
                continue
            interp_col = []
            ages_all = grp["age"].values
            for i, (idx, row) in enumerate(grp.iterrows()):
                # exclude current row for interpolation
                mask = valid.index != idx
                other = valid[mask]
                if len(other) < 2:
                    interp_col.append(np.nan)
                else:
                    age_arr = other["age"].values
                    val_arr = other[col].values
                    sort_idx = np.argsort(age_arr)
                    interp_col.append(
                        np.interp(row["age"], age_arr[sort_idx], val_arr[sort_idx])
                    )
            vals.append(pd.Series(interp_col, index=grp.index))
        out[f"patient_{col}_interp"] = pd.concat(vals).reindex(out.index)

    # patient slope of off vs age (np.polyfit, NaN if < 2 valid points)
    slopes = []
    for pid, grp in out.groupby("patient_id", sort=False):
        valid = grp[["off", "age"]].dropna()
        if len(valid) < 2:
            slopes.append(pd.Series(np.nan, index=grp.index))
        else:
            slope = np.polyfit(valid["age"].values, valid["off"].values, 1)[0]
            slopes.append(pd.Series(slope, index=grp.index))
    out["patient_off_slope"] = pd.concat(slopes).reindex(out.index)

    # on - off where both exist
    out["on_minus_off"] = out["on"] - out["off"]

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
print("07b RMSE:", report.metrics.rmse())

cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("07b", report)

fitted = predictions.skb.make_learner()
fitted.fit({"visits": visits})
preds = fitted.predict({"visits": X_test})
sub = pd.DataFrame({"target": preds}, index=X_test.index)
sub.index.name = "Index"
(PROJECT_ROOT / "submissions").mkdir(exist_ok=True)
sub.to_csv(PROJECT_ROOT / "submissions" / "07b.csv")
print("07b submission written")
