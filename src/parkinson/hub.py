"""Helpers for authenticating against Skore Hub using the `.skore` file."""

from __future__ import annotations

import json
import os
from pathlib import Path

_SKORE_FILE = Path(__file__).resolve().parents[2] / ".skore"


def load_skore_credentials() -> dict:
    """Load credentials from `.skore` at the repo root.

    Sets the ``SKORE_HUB_API_KEY`` environment variable so that
    ``skore.login(mode='hub')`` can pick it up without browser interaction,
    and returns the config dict (keys: hub_url, workspace, workspace_id,
    api_key, harness).
    """
    config = json.loads(_SKORE_FILE.read_text())
    os.environ["SKORE_HUB_API_KEY"] = config["api_key"]
    os.environ.setdefault("SKORE_HUB_URI", config["hub_url"])
    return config
