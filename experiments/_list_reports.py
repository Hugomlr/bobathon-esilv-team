"""List existing reports on the Hub project."""
import json, urllib.request, urllib.error
from pathlib import Path

cfg = json.loads(Path(".skore").read_text())
url = f"{cfg['hub_url']}/projects/{cfg['workspace']}/bobathon-esilv/estimator-reports/"
req = urllib.request.Request(url, headers={"X-API-Key": cfg["api_key"]})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
        print(json.dumps(data, indent=2)[:3000])
except urllib.error.HTTPError as e:
    body = e.read().decode()[:500]
    print(f"HTTP {e.code}: {body}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
