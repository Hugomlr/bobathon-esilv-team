# %% [markdown]
# # 07 — Hyperparameter tuning with RandomizedSearchCV
#
# Builds on experiment 06 (stateless + PK decay features).
# A RandomizedSearchCV wraps HistGradientBoostingRegressor inside the
# DataOps graph. The inner CV uses KFold(n_splits=5) for HPO selection;
# the outer GroupKFold(n_splits=5) on patient_id is baked into the graph
# via mark_as_X to keep the evaluation patient-grouped and leak-free.
#
# Search space:
#   learning_rate:    [0.01, 0.05, 0.1]
#   max_iter:         [100, 200, 300]
#   l2_regularization:[0.0, 0.5, 1.0, 5.0]
#   min_samples_leaf: [20, 50, 100]

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV

import skrub

import skore
from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data ──────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
visits = X_train.join(y_train, how="left")

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")


# ── Feature function (identical to experiment 06) ─────────────────────────────

def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless + PK-decay features; row order is never changed."""
    out = df.copy()
    out["disease_duration"] = out["age"] - out["age_at_diagnosis"]
    out["is_off_missing"] = out["off"].isna().astype(int)
    out["is_ledd_missing"] = out["ledd"].isna().astype(int)
    out["log_time_since_intake_on"] = np.log1p(out["time_since_intake_on"].fillna(0))
    out["log_time_since_intake_off"] = np.log1p(out["time_since_intake_off"].fillna(0))
    out["ledd_decay_on"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_on"])
    out["ledd_decay_off"] = out["ledd"] * np.exp(-0.5 * out["time_since_intake_off"])
    return out


# ── DataOps graph ─────────────────────────────────────────────────────────────
data = skrub.var("visits", visits)

groups = data["patient_id"]

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

# ── Hyperparameter search ─────────────────────────────────────────────────────
# Inner CV: KFold (no groups) — purely for HPO selection within each training
# fold.  The outer GroupKFold handles the patient-level leakage guard.
# n_iter=36 covers all 3×3 (learning_rate × max_iter) combinations plus
# broad sampling of the regularization and leaf-size axes.
param_distributions = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 200, 300],
    "l2_regularization": [0.0, 0.5, 1.0, 5.0],
    "min_samples_leaf": [20, 50, 100],
}

search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=0),
    param_distributions=param_distributions,
    n_iter=36,
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True,
    random_state=0,
    verbose=1,
)

predictions = encoded.skb.apply(search, y=y_op)

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
project.put("07_hyperparameter_tuning", report)

# ── Extract best parameters ───────────────────────────────────────────────────
# Refit on the full training data to get the best params from the search.
fitted_learner = predictions.skb.make_learner()
fitted_learner.fit({"visits": visits})


def _find_search_in_graph(data_op):
    """Walk the DataOps graph (DFS) to find the first fitted SearchCV node."""
    visited = set()
    stack = [data_op]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        impl = getattr(node, "_skrub_impl", None)
        if impl is not None:
            est = getattr(impl, "estimator_", None)
            if est is not None and hasattr(est, "best_params_"):
                return est
        for attr in ("_inputs", "inputs", "_data_ops"):
            children = getattr(node, attr, None)
            if children:
                vals = children.values() if isinstance(children, dict) else children
                stack.extend(vals)
    return None


fitted_search = _find_search_in_graph(fitted_learner.data_op)
if fitted_search is not None:
    print("\nBest parameters found by RandomizedSearchCV:")
    for k, v in sorted(fitted_search.best_params_.items()):
        print(f"  {k}: {v}")
    print(f"Best inner-CV RMSE: {-fitted_search.best_score_:.4f}")
else:
    print("\nCould not extract best_params_ from the fitted graph.")

# ── Kaggle submission ─────────────────────────────────────────────────────────
preds = fitted_learner.predict({"visits": X_test})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({"target": preds}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "07_hyperparameter_tuning.csv")
print(
    f"\nSubmission written: {submissions_dir / '07_hyperparameter_tuning.csv'}"
    f"  shape={submission.shape}"
)
