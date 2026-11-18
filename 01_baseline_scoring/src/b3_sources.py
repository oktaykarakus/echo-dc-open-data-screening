"""B3 helper: query size/date/checksum availability of the dated Geofabrik snapshots (HEAD requests only)."""
import requests, json
from common import *
rows = []
for cc, slug in EXTRACTS.items():
    url = GEOFABRIK.format(slug=slug)
    h = requests.head(url, timeout=30, allow_redirects=True)
    rows.append({"country": cc, "url": url, "status": h.status_code, "bytes": int(h.headers.get("content-length", 0)),
                 "last_modified": h.headers.get("last-modified")})
    print(f'{cc} {h.status_code} {rows[-1]["bytes"]/1e9:6.2f} GB  {rows[-1]["last_modified"]}')
print("TOTAL GB", round(sum(r["bytes"] for r in rows) / 1e9, 1))
json.dump(rows, open(DATA / "geofabrik_250101_sources.json", "w"), indent=1)
