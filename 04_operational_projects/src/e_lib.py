"""ECHO-DC V2 stage 04: shared library.

stage 01 `scoring.py`, stage 02 `multiscale.py` and the stage 01/C/D result tables are READ ONLY inputs. Nothing from
Phase A-D is modified or copied. All temp/cache paths are forced inside the stage 04 folder.
"""
import os, sys, json, hashlib, math, tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PE = HERE.parent                                    # .../04_operational_projects
V2 = PE.parent                                      # .../ECHO_DC_V2
_TMP = PE / ".tmp"
for _k, _sub in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl"}.items():
    (_TMP / _sub).mkdir(parents=True, exist_ok=True)
    os.environ[_k] = str(_TMP / _sub)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
tempfile.tempdir = str(_TMP / "tmp")

PA, PB, PC, PD = (V2 / n for n in ("Phase_A_Feasibility", "01_baseline_scoring", "02_multiscale", "03_robustness"))
RESULTS = Path(os.environ.get("ECHO_E_RESULTS", PE / "results"))
FIGURES = Path(os.environ.get("ECHO_E_FIGURES", PE / "figures"))
RESULTS.mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd

N = 1216
SCALES = ["1km", "2p5km", "5km", "10km"]
RETRIEVED = "2026-11-07"                      # date all web sources were accessed
TOP10, TOP20, TOP50 = math.ceil(N * .10), math.ceil(N * .20), math.ceil(N * .50)   # 122, 244, 608 (stage 02/D convention for 10%)


def load_scores():
    """B0-QA record table (authoritative baseline)."""
    return pd.read_csv(PB / "results" / "scores_B0QA.csv")


def load_multiscale():
    """Long stage 02 table (equal weights columns used): 4 x 1,216 rows."""
    return pd.read_csv(PC / "results" / "multiscale_scores_long.csv")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def phase_files(phases=(PA, PB, PC, PD)):
    out = []
    for ph in phases:
        for p in sorted(ph.rglob("*")):
            if p.is_file() and ".tmp" not in p.parts and "__pycache__" not in p.parts:
                out.append(p)
    return out


def manifest(phases=(PA, PB, PC, PD)):
    return {str(p.relative_to(V2)): sha256(p) for p in phase_files(phases)}


def rank_desc(score):
    """stage 01 rule: descending raw score, ties by input order."""
    s = pd.Series(np.asarray(score, float))
    return s.rank(method="first", ascending=False).astype(int).values


def pct_top(rank, n=N):
    """'Top x %' position = rank / n x 100 (rank 1 of 1,216 -> 0.08 %)."""
    return 100.0 * np.asarray(rank, float) / n
