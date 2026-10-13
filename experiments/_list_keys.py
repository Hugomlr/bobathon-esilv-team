"""List just the keys already on the Hub project."""
import json, urllib.request, urllib.error
from pathlib import Path

cfg = json.loads(Path(".skore").read_text())
url = f"{cfg['hub_url']}/projects/{cfg['workspace']}/bobathon-esilv/estimator-reports/"
req = urllib.request.Request(url, headers={"X-API-Key": cfg["api_key"]})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
        keys = sorted(set(r["key"] for r in data))
        print("Keys on Hub:")
        for k in keys:
            print(" ", k)
        print(f"\nTotal reports: {len(data)}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
