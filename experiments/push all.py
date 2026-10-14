"""
Pousse TOUTES les expériences sur Skore Hub + écrit les fichiers Kaggle.

Depuis la racine du repo, venv activé :
    python experiments/push_all.py                  # tout ce qui n'est pas déjà sur le Hub
    python experiments/push_all.py 02_ridge 05_tabular_pipeline   # seulement ces expériences
    python experiments/push_all.py --force          # re-pousse même si la clé existe
    python experiments/push_all.py --dry-run        # évalue + écrit les CSV, sans toucher au Hub
    python experiments/push_all.py --trace          # affiche chaque requête réseau (hôte, taille, durée)

Pour chaque expérience :
  1. skore.evaluate(...) SANS splitter=  -> EstimatorReport (holdout 80/20), léger
  2. project.put(<clé>, report)          -> skore affiche l'URL du rapport : c'est elle
                                            qu'il faut coller dans la Submission Description Kaggle
  3. refit sur tout le train -> submissions/<clé>.csv (colonnes Index,target)

Les clés 01..09 du workspace sont déjà prises par Maxence et Linda : les tiennes
sont préfixées (KEY_PREFIX) pour éviter les doublons et garder nom de fichier == clé.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import skore
import skrub
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skrub import tabular_pipeline

try:  # présents quand le support Hub de skore est installé (c'est le cas avec skore-cli)
    import httpx
    import skore._plugins.hub.artifact.upload as _skore_upload
except ImportError:  # --dry-run reste utilisable sans
    httpx = _skore_upload = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
PROJECT_NAME = "bobathon-esilv"
KEY_PREFIX = "hugo_"          # clés Hub finales : hugo_01_dummy, hugo_02_ridge, ...
RANDOM_STATE = 42

FEATURE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
DROP_COLS = ["patient_id", "Index"]
BEST_PARAMS = dict(max_iter=500, learning_rate=0.05, max_depth=5,
                   min_samples_leaf=20, l2_regularization=0.0,
                   random_state=RANDOM_STATE)


# ── Données ───────────────────────────────────────────────────────────────────
class Data:
    """Les 3 jeux de features (A numérique, B toutes colonnes, C ingénierie) train + test."""

    def __init__(self) -> None:
        X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
        y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
        self.X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
        self.sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
        assert (X_train_raw["Index"] == y_train_raw["Index"]).all(), "Index X/y décalés"

        self.visits = X_train_raw.merge(y_train_raw, on="Index")
        self.y = self.visits["target"]
        self.test_index = self.X_test_raw["Index"]

        # A : colonnes numériques
        self.A = (self.visits[FEATURE_COLS], self.X_test_raw[FEATURE_COLS])
        # B : tout sauf identifiants / cible
        self.B = (self.visits.drop(columns=["Index", "patient_id", "target"]),
                  self.X_test_raw.drop(columns=["Index", "patient_id"]))
        # C : features d'ingénierie + catégorielles encodées (encodeur fit sur le train seul)
        c_train, c_test = self._engineer(X_train_raw), self._engineer(self.X_test_raw)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        c_train[CATEGORICAL_COLS] = enc.fit_transform(c_train[CATEGORICAL_COLS].astype(str))
        c_test[CATEGORICAL_COLS] = enc.transform(c_test[CATEGORICAL_COLS].astype(str))
        self.C = (c_train, c_test)

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["disease_duration"] = d["age"] - d["age_at_diagnosis"]
        d["on_off_diff"] = d["off"] - d["on"]
        d["on_off_mean"] = (d["on"] + d["off"]) / 2.0
        return d.drop(columns=DROP_COLS)


def build_dataops(visits: pd.DataFrame):
    """Graphe skrub : on retire Index, patient_id et target des features."""
    data_op = skrub.var("visits", visits)
    X_op = data_op.drop(columns=["Index", "patient_id", "target"], errors="ignore").skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data_op["patient_id"]},
    )
    y_op = data_op["target"].skb.mark_as_y()
    return (
        X_op.skb.apply(skrub.TableVectorizer())
        .skb.apply(HistGradientBoostingRegressor(random_state=RANDOM_STATE), y=y_op)
    )


# ── Définition des expériences ────────────────────────────────────────────────
# name -> (feature set, fabrique de modèle).  "dataops" est traité à part.
EXPERIMENTS: dict[str, tuple[str, callable]] = {
    "01_dummy": ("A", lambda: DummyRegressor(strategy="mean")),
    "02_ridge": ("A", lambda: make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))),
    "03_hist_gradient_boosting": ("A", lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    "04_hist_gradient_boosting_tuned": ("C", lambda: HistGradientBoostingRegressor(**BEST_PARAMS)),
    "05_tabular_pipeline": ("B", lambda: tabular_pipeline("regressor")),
    "06_dataops": ("dataops", None),
}


def evaluate_experiment(name: str, d: Data):
    """Retourne le report skore (sans splitter= -> EstimatorReport)."""
    feat, factory = EXPERIMENTS[name]
    if feat == "dataops":
        learner = build_dataops(d.visits).skb.make_learner()
        # splitter=0.2 : sans lui, le CV GroupKFold du graphe donne un CrossValidationReport
        # (5 folds, bien plus lourd à pousser, et Kaggle veut un EstimatorReport).
        return skore.evaluate(learner, data={"visits": d.visits}, splitter=0.2)
    X, _ = getattr(d, feat)
    return skore.evaluate(factory(), X, d.y)


def write_submission(name: str, key: str, d: Data) -> Path:
    """Refit sur tout le train, prédit le test, écrit submissions/<key>.csv."""
    feat, factory = EXPERIMENTS[name]
    if feat == "dataops":
        learner = build_dataops(d.visits).skb.make_learner()
        learner.fit({"visits": d.visits})
        preds = learner.predict({"visits": d.X_test_raw})
    else:
        X, X_test = getattr(d, feat)
        preds = clone(factory()).fit(X, d.y).predict(X_test)

    sub = pd.DataFrame({"Index": d.test_index, "target": preds})
    assert list(sub.columns) == ["Index", "target"]
    assert len(sub) == len(d.sample), "Nombre de lignes != sample_submission"
    assert all(math.isfinite(v) for v in sub["target"]), "Prédictions NaN/inf"
    assert sub["Index"].equals(d.sample["Index"]), "Ordre des Index != sample_submission"

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / f"{key}.csv"
    sub.to_csv(path, index=False)
    return path


# ── Réseau : upload tolérant aux connexions lentes ───────────────────────────
# Chaque rapport est envoyé en 1 PUT de ~5 à 10 Mo (voir `--trace`). skore impose 30 s
# par PUT : sur une liaison montante lente ou instable, il expire et recommence depuis
# zéro, d'où les "Request failed to reach server" à répétition (alors que les petits
# rapports de test passent). On laisse ici 10 minutes par PUT.
def _upload_chunk_patient(filepath, client, url, offset, length, content_type):
    with filepath.open("rb") as file:
        file.seek(offset)
        response = client.put(
            url=url,
            content=file.read(length),
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        return response.headers["etag"]


if _skore_upload is not None:
    _skore_upload.upload_chunk = _upload_chunk_patient


def enable_trace() -> None:
    """Affiche chaque requête HTTP de skore : hôte + chemin (sans signature), taille, durée."""
    if httpx is None:
        sys.exit("--trace nécessite httpx (installé avec le support Hub de skore).")
    original_send = httpx.Client.send

    def traced_send(self, request, **kwargs):
        u = request.url
        label = f"{request.method} {u.host}{u.path}"
        size = request.headers.get("content-length", "?")
        t0 = time.perf_counter()
        print(f"    -> {label}  [{size} octets]", flush=True)
        try:
            response = original_send(self, request, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"    !! {label}  {type(exc).__name__} après {time.perf_counter() - t0:.1f}s",
                  flush=True)
            raise
        print(f"    <- {response.status_code} {label}  en {time.perf_counter() - t0:.1f}s",
              flush=True)
        return response

    httpx.Client.send = traced_send


# ── Hub ───────────────────────────────────────────────────────────────────────
def connect_hub():
    cfg = json.loads((REPO_ROOT / ".skore").read_text())
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]      # AVANT login
    os.environ.setdefault("SKORE_HUB_URI", cfg["hub_url"])
    skore.login(mode="hub")
    project = skore.Project(name=PROJECT_NAME, mode="hub", workspace=cfg["workspace"])
    print(f"Hub OK : workspace={cfg['workspace']!r}, project={PROJECT_NAME!r}")
    return project, cfg


def existing_keys(cfg: dict) -> set[str]:
    """Clés déjà présentes sur le Hub (lecture seule). Vide si la requête échoue."""
    url = f"{cfg['hub_url'].rstrip('/')}/projects/{cfg['workspace']}/{PROJECT_NAME}/estimator-reports/"
    req = urllib.request.Request(url, headers={"X-API-Key": cfg["api_key"]})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return {item["key"] for item in json.loads(r.read().decode())}
    except Exception as exc:  # noqa: BLE001
        print(f"(impossible de lister les clés du Hub : {type(exc).__name__} — on continue)")
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="expériences à lancer (défaut : toutes)")
    ap.add_argument("--force", action="store_true", help="re-pousse même si la clé existe déjà")
    ap.add_argument("--dry-run", action="store_true", help="pas de Hub : évalue + écrit les CSV")
    ap.add_argument("--trace", action="store_true", help="affiche chaque requête réseau")
    args = ap.parse_args()

    unknown = [n for n in args.names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"Expériences inconnues : {unknown}. Disponibles : {list(EXPERIMENTS)}")
    names = args.names or list(EXPERIMENTS)
    if args.trace:
        enable_trace()

    d = Data()
    project = cfg = None
    on_hub: set[str] = set()
    if not args.dry_run:
        project, cfg = connect_hub()
        on_hub = existing_keys(cfg)

    rows = []
    for name in names:
        key = f"{KEY_PREFIX}{name}"
        print(f"\n=== {key} ===")
        status = "ok"
        t0 = time.perf_counter()

        if key in on_hub and not args.force:
            print("  déjà sur le Hub -> push ignoré (--force pour le refaire)")
            status = "déjà sur le Hub"
        else:
            report = evaluate_experiment(name, d)
            print(f"  évaluation : {time.perf_counter() - t0:.0f}s ({type(report).__name__})")
            if args.dry_run:
                status = "dry-run (pas de push)"
            else:
                t1 = time.perf_counter()
                try:
                    project.put(key, report)    # skore affiche l'URL du rapport
                    print(f"  push OK en {time.perf_counter() - t1:.0f}s")
                except Exception as exc:  # noqa: BLE001
                    status = f"ÉCHEC push : {type(exc).__name__}"
                    print(f"  push ÉCHEC après {time.perf_counter() - t1:.0f}s : "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)

        csv = SUBMISSIONS_DIR / f"{key}.csv"
        if not csv.exists():
            csv = write_submission(name, key, d)
            print(f"  CSV écrit : submissions/{csv.name}")
        rows.append((key, status, f"submissions/{csv.name}"))

    print("\n" + "=" * 78)
    print(f"{'Clé Hub':<42}{'Statut':<26}Fichier Kaggle")
    print("-" * 78)
    for key, status, csv in rows:
        print(f"{key:<42}{status:<26}{csv}")
    print("=" * 78)
    print("Pour Kaggle : colle l'URL du rapport affichée par skore (ligne 'push OK')\n"
          "dans la Submission Description, avec le CSV du même nom que la clé.")
    return 0 if all(not s.startswith("ÉCHEC") for _, s, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())