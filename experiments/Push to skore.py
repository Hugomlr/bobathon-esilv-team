"""
Push les expériences vers Skore Hub (workspace lu dans .skore).

À lancer depuis la racine du repo, venv activé :
    python experiments/push_to_skore.py            # pousse tout ce qui manque
    python experiments/push_to_skore.py 02_ridge   # pousse seulement certaines clés

Pourquoi ce script marche là où le notebook échouait :
  * skore.evaluate(...) SANS splitter= -> EstimatorReport (holdout 80/20), léger.
    Avec splitter=<liste de 5 splits> on obtient un CrossValidationReport :
    put() recalcule alors tous les displays/inspections pour chaque fold sur
    44 590 lignes, l'upload dépasse le timeout client (30 s, 10 retries) et la
    cellule finit en KeyboardInterrupt dans serialize_model.
    Le concours Kaggle exige d'ailleurs une URL d'EstimatorReport.
  * La clé API de .skore est exportée dans l'environnement AVANT skore.login().
  * Chaque push est isolé (try/except) et chronométré ; un échec n'arrête pas la suite.
  * 06_dataops : data={"visits": visits} est obligatoire.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import skore
import skrub
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skrub import tabular_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROJECT_NAME = "bobathon-esilv"
RANDOM_STATE = 42

FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
DROP_COLS = ["patient_id", "Index"]


# ── Hub ───────────────────────────────────────────────────────────────────────
def connect_hub():
    cfg = json.loads((REPO_ROOT / ".skore").read_text())
    # Indispensable : skore.login(mode="hub") lit ces variables.
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    os.environ.setdefault("SKORE_HUB_URI", cfg["hub_url"])
    skore.login(mode="hub")
    project = skore.Project(name=PROJECT_NAME, mode="hub", workspace=cfg["workspace"])
    print(f"Hub OK : workspace={cfg['workspace']!r}, project={PROJECT_NAME!r}")
    return project, cfg["workspace"]


def push(project, workspace, key, report) -> str | None:
    t0 = time.perf_counter()
    try:
        project.put(key, report)
    except Exception as exc:  # noqa: BLE001
        print(f"  [Hub] {key} ECHEC après {time.perf_counter() - t0:.0f}s : "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    url = f"https://skore.probabl.ai/{workspace}/projects/{PROJECT_NAME}/reports/{key}"
    print(f"  [Hub] {key} OK en {time.perf_counter() - t0:.0f}s -> {url}")
    return url


# ── Données ───────────────────────────────────────────────────────────────────
def load_data():
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    assert (X_train_raw["Index"] == y_train_raw["Index"]).all()
    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]

    X_a = visits[FEATURE_COLS]                                              # set A
    X_b = visits.drop(columns=["Index", "patient_id", "target"])            # set B

    d = X_train_raw.copy()                                                  # set C
    d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
    d["on_off_diff"] = d["off"] - d["on"]
    d["on_off_mean"] = (d["on"] + d["off"]) / 2.0
    X_c = d.drop(columns=DROP_COLS)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_c[CATEGORICAL_COLS] = enc.fit_transform(X_c[CATEGORICAL_COLS].astype(str))
    return visits, y, X_a, X_b, X_c


def build_dataops(visits):
    data_op = skrub.var("visits", visits)
    X_op = data_op.drop("target", axis=1, errors="ignore").skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data_op["patient_id"]},
    )
    y_op = data_op["target"].skb.mark_as_y()
    return (
        X_op.skb.apply(skrub.TableVectorizer())
        .skb.apply(HistGradientBoostingRegressor(random_state=RANDOM_STATE), y=y_op)
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main(only: set[str]) -> None:
    project, workspace = connect_hub()
    visits, y, X_a, X_b, X_c = load_data()

    best_params = dict(max_iter=500, learning_rate=0.05, max_depth=5,
                       min_samples_leaf=20, l2_regularization=0.0,
                       random_state=RANDOM_STATE)

    # key -> fonction qui construit le report (pas de splitter= => EstimatorReport)
    jobs = {
        "01_dummy": lambda: skore.evaluate(DummyRegressor(strategy="mean"), X_a, y),
        "02_ridge": lambda: skore.evaluate(
            make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)), X_a, y),
        "03_hist_gradient_boosting": lambda: skore.evaluate(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE), X_a, y),
        "04_hist_gradient_boosting_tuned": lambda: skore.evaluate(
            HistGradientBoostingRegressor(**best_params), X_c, y),
        "05_tabular_pipeline": lambda: skore.evaluate(
            tabular_pipeline("regressor"), X_b, y),
        "06_dataops": lambda: skore.evaluate(
            build_dataops(visits).skb.make_learner(), data={"visits": visits}),
    }

    urls: dict[str, str | None] = {}
    for key, make_report in jobs.items():
        if only and key not in only:
            continue
        print(f"\n{key}")
        t0 = time.perf_counter()
        report = make_report()
        print(f"  évaluation : {time.perf_counter() - t0:.0f}s ({type(report).__name__})")
        urls[key] = push(project, workspace, key, report)

    print("\nRésumé")
    for key, url in urls.items():
        print(f"  {key:<34} {url or 'ÉCHEC'}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))