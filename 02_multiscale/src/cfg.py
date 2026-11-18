"""stage 02 paths/constants. stage 01 code is imported READ-ONLY (never modified); nothing in Phase A/B is written to."""
import os, sys
from pathlib import Path

PC = Path(__file__).resolve().parents[1]              # .../02_multiscale
V2 = PC.parent
PB = V2 / "01_baseline_scoring"
RESULTS = Path(os.environ.get("ECHO_C_RESULTS", PC / "results"))   # override only for determinism tests
FIGURES = Path(os.environ.get("ECHO_C_FIGURES", PC / "figures"))
sys.path.insert(0, str(PB / "src"))                   # makes stage 01 `scoring`, `common`, `roads` importable

TABLE105 = PB / "data" / "analytical_table_105col.csv"
B0QA = PB / "results" / "scores_B0QA.csv"
B0 = PB / "results" / "scores_B0.csv"

SCALES = ["1km", "2p5km", "5km", "10km"]
SCALE_KM = {"1km": 1.0, "2p5km": 2.5, "5km": 5.0, "10km": 10.0}
SCALE_LABEL = {"1km": "1 km", "2p5km": "2.5 km", "5km": "5 km", "10km": "10 km"}
REF = "2p5km"
SCENARIOS = ["equal", "literature_informed", "entropy"]
