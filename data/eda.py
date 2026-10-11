# %% [markdown]
# # EDA: Parkinson's Disease UPDRS Prediction
#
# Exploratory data analysis of the ESILV bobathon dataset, run before designing a model.
#
# - **Raw data** is read-only — `RAW` points at `data/X_train.csv` + `data/y_train.csv`.
#   This file never cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): `eda_train.html`,
#   `eda_test.html`, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Merge X_train and y_train on the Index column to form the full training table.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
RAW = X_train.join(y_train, how="left")
RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview — training set
#
# Per-column summary: dtype, fraction missing, number of unique values.

# %%
report = skrub.TableReport(RAW, title="train", verbose=0)
report.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Table overview — test set
#
# Same overview for X_test so we can compare missingness patterns.

# %%
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")
report_test = skrub.TableReport(X_test, title="test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows_test": n_rows_test, "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Missingness deep-dive: on, off, ledd
#
# The `on` and `off` columns capture UPDRS scores under medication (on)
# and without (off). `ledd` is levodopa equivalent daily dose.
# All three may be missing when a visit type was not performed.

# %%
focus_cols = ["on", "off", "ledd"]
missing_counts = RAW[focus_cols].isna().sum()
missing_pct = (RAW[focus_cols].isna().mean() * 100).round(2)
both_missing = RAW[["on", "off"]].isna().all(axis=1).sum()
on_only = (RAW["on"].notna() & RAW["off"].isna()).sum()
off_only = (RAW["on"].isna() & RAW["off"].notna()).sum()
both_present = (RAW["on"].notna() & RAW["off"].notna()).sum()
{
    "missing_count": missing_counts.to_dict(),
    "missing_pct": missing_pct.to_dict(),
    "on_off_pattern": {
        "both_present": int(both_present),
        "on_only": int(on_only),
        "off_only": int(off_only),
        "both_missing": int(both_missing),
    },
    "ledd_missing_when_both_on_off_missing": int(
        RAW.loc[RAW[["on", "off"]].isna().all(axis=1), "ledd"].isna().sum()
    ),
}  # noqa: B018

# %% [markdown]
# ## Target analysis
#
# The target is `target` (a UPDRS-like composite score — regression task).

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Visits per patient_id
#
# High-cardinality `patient_id` suggests repeated measures. Understanding
# the visit distribution per patient is key for choosing a GroupKFold
# or within-patient cross-validator.

# %%
visits_per_patient = RAW.groupby("patient_id").size()
{
    "n_unique_patients": int(visits_per_patient.shape[0]),
    "visits_min": int(visits_per_patient.min()),
    "visits_max": int(visits_per_patient.max()),
    "visits_mean": round(float(visits_per_patient.mean()), 2),
    "visits_median": float(visits_per_patient.median()),
    "distribution": visits_per_patient.value_counts().sort_index().to_dict(),
}  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns and high-cardinality id/group columns.

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links
# are candidate predictors; an implausibly perfect one is a possible
# leakage flag.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
