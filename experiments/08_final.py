"""08a — 07b + extended patient features; 08b adds a VotingRegressor ensemble."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import skrub, skore
from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
visits = X_train.join(y_train, how="left")
X_test  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv",  index_col="Index")

CAT_COLS = ["rater_id", "cohort", "gene"]
all_data = pd.concat([X_train, X_test], axis=0)
CAT_CATEGORIES = {c: sorted(all_data[c].dropna().unique().tolist()) for c in CAT_COLS}


def _prev_next(series: pd.Series, ages: pd.Series, cur_age: float, cur_idx):
    """Return (prev_val, prev_age_gap, next_val, next_age_gap) from other visits."""
    others = series.drop(index=cur_idx, errors="ignore")
    other_ages = ages.drop(index=cur_idx, errors="ignore")
    valid = others.dropna()
    if valid.empty:
        return np.nan, np.nan, np.nan, np.nan
    valid_ages = other_ages.loc[valid.index]
    before = valid_ages[valid_ages < cur_age]
    after  = valid_ages[valid_ages > cur_age]
    prev_val = valid.loc[before.idxmax()] if not before.empty else np.nan
    prev_gap = cur_age - before.max()      if not before.empty else np.nan
    next_val = valid.loc[after.idxmin()]   if not after.empty  else np.nan
    next_gap = after.min() - cur_age       if not after.empty  else np.nan
    return prev_val, prev_gap, next_val, next_gap


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ── stateless ──────────────────────────────────────────────────────────────
    out["disease_duration"]        = out["age"] - out["age_at_diagnosis"]
    out["is_off_missing"]          = out["off"].isna().astype(int)
    out["is_ledd_missing"]         = out["ledd"].isna().astype(int)
    out["log_time_since_intake_on"]  = np.log1p(out["time_since_intake_on"])
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"])
    out["ledd_decay_on"]  = out["ledd"] * np.exp(-0.5 * out["time_since_intake_on"])
    out["ledd_decay_off"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_off"])
    out["on_minus_off"]   = out["on"] - out["off"]
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c], categories=CAT_CATEGORIES[c])

    # ── patient groupby (feature cols only) ────────────────────────────────────
    g = out.groupby("patient_id", sort=False)
    out["n_visits"]              = g["age"].transform("count")
    out["age_first"]             = g["age"].transform("min")
    out["time_since_first_visit"] = out["age"] - out["age_first"]
    out["n_off_available"]       = g["off"].transform("count")
    out["n_on_available"]        = g["on"].transform("count")

    for col in ["on", "off", "ledd"]:
        out[f"patient_{col}_mean"] = g[col].transform("mean")
        out[f"patient_{col}_std"]  = g[col].transform("std")
        out[f"patient_{col}_min"]  = g[col].transform("min")
        out[f"patient_{col}_max"]  = g[col].transform("max")

    # LOO mean for on / off
    for col in ["on", "off"]:
        s = g[col].transform("sum")
        n = g[col].transform("count")
        loo = (s - out[col]) / (n - 1)
        loo[(n <= 1) | out[col].isna()] = np.nan
        out[f"patient_{col}_loo_mean"] = loo

    # patient mean of (on - off)
    valid_diff = out["on"].where(out["off"].notna()) - out["off"].where(out["on"].notna())
    out["patient_on_minus_off_mean"] = (
        valid_diff.groupby(out["patient_id"]).transform("mean")
    )

    # per-row prev/next off and on (age-ordered, exclude current row)
    prev_off, prev_off_gap = [], []
    next_off, next_off_gap = [], []
    prev_on,  prev_on_gap  = [], []
    next_on,  next_on_gap  = [], []
    off_interp, on_interp  = [], []
    off_slope, on_slope    = [], []
    off_linfit, on_linfit  = [], []
    off_quadfit            = []

    for pid, grp in out.groupby("patient_id", sort=False):
        ages = grp["age"]
        for col, buf_p, buf_pg, buf_n, buf_ng, buf_interp, buf_slope, buf_lin, buf_quad in [
            ("off", prev_off, prev_off_gap, next_off, next_off_gap,
             off_interp, off_slope, off_linfit, off_quadfit),
            ("on",  prev_on,  prev_on_gap,  next_on,  next_on_gap,
             on_interp, on_slope, on_linfit, None),
        ]:
            valid = grp[[col, "age"]].dropna()
            for idx, row in grp.iterrows():
                pv, pg, nv, ng = _prev_next(grp[col], ages, row["age"], idx)
                buf_p.append(pv); buf_pg.append(pg)
                buf_n.append(nv); buf_ng.append(ng)

                # interp from other visits
                other = valid[valid.index != idx]
                if len(other) >= 2:
                    srt = other.sort_values("age")
                    buf_interp.append(
                        np.interp(row["age"], srt["age"].values, srt[col].values)
                    )
                else:
                    buf_interp.append(np.nan)

                # linear slope
                if len(valid) >= 2:
                    coefs1 = np.polyfit(valid["age"].values, valid[col].values, 1)
                    buf_slope.append(coefs1[0])
                    buf_lin.append(np.polyval(coefs1, row["age"]))
                else:
                    buf_slope.append(np.nan)
                    buf_lin.append(np.nan)

                # quadratic fitted value (off only)
                if buf_quad is not None:
                    if len(valid) >= 4:
                        coefs2 = np.polyfit(valid["age"].values, valid[col].values, 2)
                        buf_quad.append(np.polyval(coefs2, row["age"]))
                    else:
                        buf_quad.append(np.nan)

    idx_order = out.index
    out["prev_off"]     = pd.Series(prev_off,     index=idx_order)
    out["prev_off_gap"] = pd.Series(prev_off_gap, index=idx_order)
    out["next_off"]     = pd.Series(next_off,     index=idx_order)
    out["next_off_gap"] = pd.Series(next_off_gap, index=idx_order)
    out["prev_on"]      = pd.Series(prev_on,      index=idx_order)
    out["prev_on_gap"]  = pd.Series(prev_on_gap,  index=idx_order)
    out["next_on"]      = pd.Series(next_on,      index=idx_order)
    out["next_on_gap"]  = pd.Series(next_on_gap,  index=idx_order)
    out["off_interp"]   = pd.Series(off_interp,   index=idx_order)
    out["on_interp"]    = pd.Series(on_interp,    index=idx_order)
    out["off_slope"]    = pd.Series(off_slope,    index=idx_order)
    out["on_slope"]     = pd.Series(on_slope,     index=idx_order)
    out["off_linfit"]   = pd.Series(off_linfit,   index=idx_order)
    out["on_linfit"]    = pd.Series(on_linfit,    index=idx_order)
    out["off_quadfit"]  = pd.Series(off_quadfit,  index=idx_order)
    out["off_minus_off_interp"] = out["off"] - out["off_interp"]
    out["on_minus_on_interp"]   = out["on"]  - out["on_interp"]
    out["off_interp_minus_on_interp"] = out["off_interp"] - out["on_interp"]

    return out


def _build_graph(visits_df, estimator):
    data = skrub.var("visits", visits_df)
    groups = data["patient_id"]
    enriched_op = data.skb.apply_func(add_features)
    X_op = (
        enriched_op
        .drop(columns=["patient_id", "target"], errors="ignore")
        .skb.mark_as_X(cv=GroupKFold(n_splits=5), split_kwargs={"groups": groups})
    )
    y_op = data["target"].skb.mark_as_y()
    return X_op.skb.apply(estimator, y=y_op)


def run(name, estimator, key, csv_name):
    predictions = _build_graph(visits, estimator)
    learner = predictions.skb.make_learner()
    report = skore.evaluate(learner, data={"visits": visits})
    print(f"{name} RMSE:", report.metrics.rmse())

    cfg = load_skore_credentials()
    skore.login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    project.put(key, report)

    fitted = predictions.skb.make_learner()
    fitted.fit({"visits": visits})
    raw = fitted.predict({"visits": X_test})
    preds = np.clip(raw, 0, 132)
    sub = pd.DataFrame({"target": preds}, index=X_test.index)
    sub.index.name = "Index"
    assert sub.shape[0] == len(X_test), "row count mismatch"
    assert sub["target"].notna().all(), "NaNs in submission"
    (PROJECT_ROOT / "submissions").mkdir(exist_ok=True)
    sub.to_csv(PROJECT_ROOT / "submissions" / csv_name)
    print(f"{name} submission written")


# ── 08a ────────────────────────────────────────────────────────────────────────
HGBR = HistGradientBoostingRegressor(
    learning_rate=0.05, max_iter=600, min_samples_leaf=40,
    l2_regularization=1.0, early_stopping=False, random_state=0,
    categorical_features="from_dtype",
)
run("08a", HGBR, "08a", "08a.csv")

# ── 08b ────────────────────────────────────────────────────────────────────────
# LightGBM: category columns need integer codes (LightGBM categorical_feature)
# We pass them as int codes; LightGBM handles them natively via categorical_feature.
# VotingRegressor feeds the same DataFrame to each sub-estimator; each wraps
# its own preprocessing.

def _lgb(seed):
    return lgb.LGBMRegressor(
        learning_rate=0.03, n_estimators=1200, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=2, random_state=seed,
        verbose=-1,
    )

voter = VotingRegressor(estimators=[
    ("hgbr0", HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=600, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0,
        categorical_features="from_dtype")),
    ("hgbr1", HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=600, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=1,
        categorical_features="from_dtype")),
    ("lgb0", _lgb(0)),
    ("lgb1", _lgb(1)),
])
run("08b", voter, "08b", "08b.csv")
