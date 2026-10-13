"""
Network and authentication diagnostic for Skore Hub.
- Reads credentials from .skore (never prints the full key).
- No model training, no report creation, no Hub writes.
- Distinguishes DNS, TLS, HTTP-auth and server-availability failures.
Run from repo root:
    python experiments/_diagnose_hub.py
"""
from __future__ import annotations

import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg  = json.loads((ROOT / ".skore").read_text())

hub_url   = cfg["hub_url"].rstrip("/")
workspace = cfg["workspace"]
api_key   = cfg["api_key"]
project   = "bobathon-esilv"
host      = urllib.parse.urlparse(hub_url).hostname

print("=== Skore credentials (redacted) ===")
print(f"  hub_url   : {hub_url}")
print(f"  workspace : {workspace}")
print(f"  api_key   : {api_key[:8]}...{api_key[-4:]}  ({len(api_key)} chars)")
print()

# ── 1. DNS resolution ──────────────────────────────────────────────────────────
print(f"=== 1. DNS resolution: {host} ===")
t0 = time.perf_counter()
try:
    addrs = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    ms = (time.perf_counter() - t0) * 1000
    for a in addrs[:3]:
        print(f"  {a[4][0]}  ({ms:.0f} ms)")
    print("  DNS: OK")
except socket.gaierror as e:
    print(f"  DNS FAILED: {e}")
    print("  Conclusion: no internet connection or DNS unreachable.")
    sys.exit(1)
print()

# ── 2. TCP + TLS handshake ─────────────────────────────────────────────────────
print(f"=== 2. TCP + TLS to {host}:443 ===")
t0 = time.perf_counter()
try:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout=10) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as s:
            ms = (time.perf_counter() - t0) * 1000
            print(f"  TLS: OK  ({ms:.0f} ms, {s.version()}, cipher={s.cipher()[0]})")
except ssl.SSLError as e:
    print(f"  TLS FAILED (cert/cipher): {e}")
except OSError as e:
    print(f"  TCP FAILED (timeout/refused): {e}")
print()

# ── 3. GET estimator-reports — identical to setup test ───────────────────────
reports_url = (
    f"{hub_url}/projects/"
    f"{urllib.parse.quote(workspace, safe='')}"
    f"/{project}/estimator-reports/"
)
print(f"=== 3. GET {reports_url} ===")
req = urllib.request.Request(reports_url, headers={"X-API-Key": api_key})
t0  = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        ms   = (time.perf_counter() - t0) * 1000
        body = r.read(200)
        print(f"  HTTP {r.status}  ({ms:.0f} ms)")
        print(f"  Body prefix: {body[:80]}")
        print("  Authentication: ACCEPTED (key is valid)")
except urllib.error.HTTPError as e:
    ms   = (time.perf_counter() - t0) * 1000
    body = e.read(200)
    print(f"  HTTP {e.code}  ({ms:.0f} ms)")
    print(f"  Body: {body[:200]}")
    if e.code in (401, 403):
        print("  Conclusion: API key REJECTED by server.")
    elif e.code == 404:
        print("  Conclusion: project not yet created (first PUT creates it) — key accepted.")
    else:
        print(f"  Conclusion: unexpected HTTP {e.code} — server-side issue.")
except urllib.error.URLError as e:
    print(f"  URLError: {e.reason}")
    print("  Conclusion: network/DNS/proxy error — server unreachable for GET.")
except TimeoutError:
    print("  Timed out (15s) on GET — server unresponsive.")
print()

# ── 4. PUT endpoint — send tiny OPTIONS probe (no body, no report) ────────────
print(f"=== 4. OPTIONS {reports_url} ===")
req2 = urllib.request.Request(
    reports_url,
    headers={"X-API-Key": api_key},
    method="OPTIONS",
)
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req2, timeout=10) as r:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  OPTIONS HTTP {r.status}  ({ms:.0f} ms)")
        allow_hdr = r.headers.get("Allow", "(not present)")
        print(f"  Allow header: {allow_hdr}")
except urllib.error.HTTPError as e:
    ms = (time.perf_counter() - t0) * 1000
    print(f"  OPTIONS HTTP {e.code}  ({ms:.0f} ms)")
except TimeoutError:
    print("  Timed out on OPTIONS — PUT endpoint unreachable.")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
print()

# ── 5. skore version and internal client URL ──────────────────────────────────
print("=== 5. skore package diagnostics ===")
try:
    import skore
    print(f"  skore version : {skore.__version__}")

    # Inspect what URL the skore client actually uses
    try:
        from skore._plugins.hub.client import client as hub_client_mod
        src = Path(hub_client_mod.__file__).read_text(encoding="utf-8")
        # Look for hard-coded base URL patterns
        import re
        urls = re.findall(r'https?://[^\s\'"]+', src)
        unique_urls = sorted(set(urls))
        if unique_urls:
            print(f"  URLs in hub client source:")
            for u in unique_urls[:10]:
                print(f"    {u}")
    except Exception as e:
        print(f"  Could not inspect hub client source: {e}")

    # Check if there is an env-var or config override
    import os
    for var in ("SKORE_HUB_URL", "SKORE_API_URL", "SKORE_BASE_URL", "HUB_URL"):
        val = os.environ.get(var)
        if val:
            print(f"  ENV {var}={val}")
        else:
            print(f"  ENV {var}: (not set)")

except ImportError as e:
    print(f"  skore not importable: {e}")
print()

# ── 6. Latency to PUT with a minimal payload (no actual report) ───────────────
print("=== 6. Latency probe: small POST to reports endpoint ===")
import json as _json
# Send an intentionally malformed but minimal body to see if the server
# responds quickly (connectivity) or hangs (routing/proxy issue).
# We expect HTTP 4xx (bad request) — we are NOT creating a report.
tiny_body = _json.dumps({"probe": True}).encode()
req3 = urllib.request.Request(
    reports_url,
    data=tiny_body,
    headers={
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    },
    method="POST",
)
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req3, timeout=10) as r:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  POST HTTP {r.status}  ({ms:.0f} ms) — unexpected 2xx, check response")
except urllib.error.HTTPError as e:
    ms   = (time.perf_counter() - t0) * 1000
    body = e.read(300)
    print(f"  POST HTTP {e.code}  ({ms:.0f} ms) — server IS responding")
    print(f"  Response body: {body[:200]}")
    if ms > 8000:
        print("  WARNING: >8s to respond — server is very slow but reachable.")
    else:
        print("  Latency looks normal.")
except TimeoutError:
    print("  POST timed out (10s) — server hangs on write path.")
    print("  Conclusion: server accepts connections but does not process POST requests.")
    print("  This matches the 'Request failed, retrying' loop seen during report push.")
except urllib.error.URLError as e:
    print(f"  URLError: {e.reason}")
print()

print("=== Diagnosis complete ===")
