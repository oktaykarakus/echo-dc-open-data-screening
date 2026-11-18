"""Shared paths/constants for ECHO-DC V2 stage 01. Run all scripts with the project venv (see requirements.txt)."""
from pathlib import Path
import os

PB = Path(__file__).resolve().parents[1]                 # .../01_baseline_scoring
ROOT = PB.parent                                         # repository root
DATA, RESULTS = PB / "data", PB / "results"

# Input (read-only; never written to)
MASTER87 = ROOT / "data" / "inputs" / "inventory_indicator_master_87col.csv"

# Large transient cache (raw PBFs). Override with ECHO_CACHE; default is outside the project tree.
CACHE = Path(os.environ.get("ECHO_CACHE", ROOT / "cache"))   # raw PBF files (not versioned; see .gitignore)

SNAPSHOT = "250101"                                      # Geofabrik 1 Jan 2025 snapshot
GEOFABRIK = "https://download.geofabrik.de/europe/{slug}-" + SNAPSHOT + ".osm.pbf"
# DC-table country -> (Geofabrik extract code used by the original notebook, Geofabrik URL slug)
EXTRACTS = {
    "NL": "netherlands", "DE": "germany", "FR": "france", "GB": "united-kingdom",
    "IE": "ireland-and-northern-ireland", "BE": "belgium", "ES": "spain", "IT": "italy",
    "PL": "poland", "SE": "sweden", "FI": "finland", "NO": "norway", "DK": "denmark",
    "AT": "austria", "CH": "switzerland", "PT": "portugal", "CZ": "czech-republic",
    "HU": "hungary", "RO": "romania", "GR": "greece",
}
ROAD_CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"]
BUFFER_KM = [1, 2.5, 5, 10]
