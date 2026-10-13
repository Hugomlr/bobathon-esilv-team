"""Quick check: can we reach Skore Hub within 8 seconds?"""
import json, urllib.request, urllib.error
from pathlib import Path

cfg = json.loads(Path(".skore").read_text())
url = f"{cfg['hub_url']}/projects/{cfg['workspace']}/bobathon-esilv/estimator-reports/"
req = urllib.request.Request(url, headers={"X-API-Key": cfg["api_key"]})
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        print(f"Hub reachable: HTTP {r.status}")
except urllib.error.HTTPError as e:
    print(f"Hub reachable: HTTP {e.code}")
except Exception as e:
    print(f"Hub NOT reachable: {type(e).__name__}: {str(e)[:100]}")
