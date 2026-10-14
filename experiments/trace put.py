"""
Trace réseau d'un push Skore : montre QUEL hôte répond ou bloque.

À lancer depuis la racine du repo, venv activé :
    python experiments/_trace_put.py

- Pousse un mini rapport dans un projet jetable "trace-test" (pas dans bobathon-esilv).
- Affiche chaque requête HTTP (hôte + chemin, sans la signature de l'URL) avec sa durée.
- Dès que tu vois une ligne "!!" (timeout), tu peux faire Ctrl+C : le diagnostic est fait.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
import skore
from sklearn.dummy import DummyRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]

_orig_send = httpx.Client.send


def _traced_send(self, request, **kwargs):
    u = request.url
    label = f"{request.method} {u.host}{u.path}"  # la query (URL signée) n'est pas affichée
    size = request.headers.get("content-length", "?")
    t0 = time.perf_counter()
    print(f"-> {label}  [{size} octets]", flush=True)
    try:
        response = _orig_send(self, request, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"!! {label}  {type(exc).__name__} après {time.perf_counter() - t0:.1f}s",
              flush=True)
        raise
    print(f"<- {response.status_code} {label}  en {time.perf_counter() - t0:.1f}s",
          flush=True)
    return response


httpx.Client.send = _traced_send

cfg = json.loads((REPO_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
os.environ.setdefault("SKORE_HUB_URI", cfg["hub_url"])
skore.login(mode="hub")

X = pd.read_csv(REPO_ROOT / "data" / "X_train.csv")[["age", "ledd"]].head(500)
y = pd.read_csv(REPO_ROOT / "data" / "y_train.csv")["target"].head(500)
report = skore.evaluate(DummyRegressor(), X, y)

project = skore.Project(name="trace-test", mode="hub", workspace=cfg["workspace"])
project.put("trace_dummy", report)
print("PUSH OK")