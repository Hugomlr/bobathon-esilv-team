"""Read-only Skore audit — no uploads, no changes, no secrets printed."""
import json, importlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
issues = []

# 1. .skore file
skore_path = ROOT / ".skore"
if not skore_path.is_file():
    issues.append("MISSING: .skore file not found")
else:
    try:
        cfg = json.loads(skore_path.read_text())
        for key in ("hub_url", "workspace", "workspace_id", "api_key"):
            if not cfg.get(key):
                issues.append(f"MISSING: .skore key '{key}' is empty")
        print(f"  .skore: workspace={cfg.get('workspace')}, hub={cfg.get('hub_url')}")
    except json.JSONDecodeError as e:
        issues.append(f"INVALID: .skore is not valid JSON: {e}")

# 2. skore importable
try:
    import skore
    print(f"  skore: version {skore.__version__} — importable OK")
except ImportError as e:
    issues.append(f"IMPORT ERROR: skore — {e}")

# 3. skore_cli importable
try:
    import skore_cli
    print(f"  skore_cli: importable OK")
except ImportError as e:
    issues.append(f"IMPORT ERROR: skore_cli — {e}")

# 4. skrub importable
try:
    import skrub
    print(f"  skrub: version {skrub.__version__} — importable OK")
    from skrub import tabular_pipeline, TableVectorizer, var
    print(f"  skrub.tabular_pipeline, TableVectorizer, var — OK")
except ImportError as e:
    issues.append(f"IMPORT ERROR: skrub — {e}")

# 5. skore.evaluate importable
try:
    from skore import evaluate, Project, login
    print(f"  skore.evaluate, Project, login — importable OK")
except ImportError as e:
    issues.append(f"IMPORT ERROR: skore.evaluate — {e}")

# 6. notebooks/day2_skore.py — syntax check
day2 = ROOT / "notebooks" / "day2_skore.py"
if day2.is_file():
    import ast
    try:
        ast.parse(day2.read_text(encoding="utf-8"))
        print(f"  day2_skore.py: syntax OK")
    except SyntaxError as e:
        issues.append(f"SYNTAX ERROR in day2_skore.py: {e}")
else:
    issues.append("MISSING: notebooks/day2_skore.py not found")

# 7. Hub connectivity (short timeout, read-only)
try:
    import urllib.request, urllib.error
    cfg = json.loads(skore_path.read_text())
    url = f"{cfg['hub_url']}/projects/{cfg['workspace']}/bobathon-esilv/estimator-reports/"
    req = urllib.request.Request(url, headers={"X-API-Key": cfg["api_key"]})
    with urllib.request.urlopen(req, timeout=6) as r:
        print(f"  Hub connectivity: HTTP {r.status} (reachable)")
except urllib.error.HTTPError as e:
    print(f"  Hub connectivity: HTTP {e.code}")
except Exception as e:
    issues.append(f"HUB UNREACHABLE: {type(e).__name__} (uploads will fail)")

print()
if issues:
    print("ISSUES FOUND:")
    for i in issues:
        print(f"  - {i}")
else:
    print("No Skore issues found.")
