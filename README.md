# Open-data geospatial screening for data-centre waste-heat reuse (ECHO-DC): analysis code and derived data

This repository contains the analysis code, configuration, and shareable derived data behind the manuscript

> *Evaluating Open-Data Geospatial Screening for Data-Centre Waste-Heat Reuse: Incremental Information and Spatial-Scale Sensitivity* (I. Turvey, O. Karakuş, C. Eyüpoğlu).

The study develops **ECHO-DC**, an open-data screening index for data-centre waste-heat reuse in Europe (1,216 OpenStreetMap data-centre records in 20 countries), and evaluates (i) how much information the composite adds to simple indicators, (ii) how its ranking and shortlist respond to the screening radius (1, 2.5, 5, 10 km), weighting scheme and neighbouring-record grouping, benchmarked against a Gaussian-copula rank-dependence model, and (iii) how it discriminates mapped district-heating-network proximity in France.

## Attribution

ECHO-DC, the original research, the original data-extraction notebook and the original scoring code are the work of **Isobelle Turvey (I.T.)**. The later analyses, the verification of the workflow and this reproducibility package (stages `01_` to `07_`, tests and documentation) were prepared by the authors with generative-AI assistance, as disclosed in the manuscript, and were reviewed by the authors.

Manuscript identifier / archive link: *to be added on publication.*

## Repository layout

| Folder | Content | Manuscript items it supports |
|---|---|---|
| `00_original_data_extraction/` | The original Colab notebook (outputs removed) in which the inventory and the non-road indicators were built; not re-run | Origin of `data/inputs/` |
| `data/inputs/` | `inventory_indicator_master_87col.csv` — the 1,216-record inventory with all non-road indicators (see `DATA_SOURCES.md`) | Inputs |
| `01_baseline_scoring/` | Index definition and scoring (`scoring.py`), strategic-road extraction (`roads.py`, needs raw OSM extracts), baseline scores (`results/scores_B0QA.csv`) | Table 1, Equations 1–4 |
| `02_multiscale/` | Indicators and scores at 1/2.5/5/10 km, rank stability, weighting scenarios | Table 4; Figure 4 |
| `03_robustness/` | Composite vs simple indicators, population double counting, ablation, normalisation window, weighting perturbations | Tables 2–3; Figures 2–3 |
| `04_operational_projects/` | Documented heat-reuse projects (evidence table with source URLs/dates), campus grouping of neighbouring records | Table 4 (campus rows); Table 7; Figure 6 |
| `05_france_network_proximity/` | Heat-network proximity benchmark for France (labels, AUC, paired block bootstrap) | Table 6; Figure 5 |
| `06_gaussian_benchmark/` | Gaussian-copula benchmark of shortlist turnover | Table 5, Table S1; Figure 4 |
| `07_manuscript_assets/` | Scripts and outputs that assemble the manuscript tables (`make_tables.py`) and figures (`make_figures.py`) | Tables 1–7, S1; Figures 1–6 |

Each stage has `src/` (code), `results/` (stored outputs) and, where applicable, `tests/`. Stored outputs were produced by the code in the same stage. Stages 02–07 read only the outputs of earlier stages.

## What can and cannot be reproduced from this repository

* **Reproducible from the contents of this repository (no downloads):** stages 01 (scoring, `b4_score.py`, `b6_qa_checks.py`), 02, 03, 04, 06 and 07. Re-running these regenerates the stored result tables, which were checked to be identical to the stored files (see "Verification").
* **Needs a download:** stage 05 requires the *France Chaleur Urbaine* heat-network file (see `05_france_network_proximity/data/README.md`; SHA-256 given). Strategic-road extraction in stage 01 (`b3_run_all.py`) requires the dated Geofabrik extracts of 1 January 2025 (≈ 25 GB in total; list and hashes in `01_baseline_scoring/data/`). The road indicators it produces are included (`road_indicators_reconstructed_250101.csv`). Likewise `02_multiscale/src/c3_road_recompute.py` (recomputing road indicators at the other radii) needs the extracted road cache; its stored result is included and used by the tests.
* **Not re-run:** the raster/vector extraction of the non-road indicators (GHS-POP, GHS-BUILT-S, Dynamic World, Hotmaps heat demand, ERA5-Land heating degree days, OSM anchor loads) was performed in the original Colab notebook in `00_original_data_extraction/` (it needs Google Drive, Earth Engine and downloaded rasters); its outputs are provided as the input table `data/inputs/inventory_indicator_master_87col.csv`. The OpenStreetMap anchor-load extracts were taken from Geofabrik "latest" files in 2026; the exact download date was not recorded.
* The stored literature-weighted reference scores used for the validation reported in Section 2.2 are in `01_baseline_scoring/data/recovered_literature_scores.csv`; `b5_validate_original_scores.py` recomputes the validation statistics from them.

## Quick start

Python 3.9, packages in `requirements.txt` (versions used for the stored results). From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# stage 01: scoring and checks (road extraction is optional; its output is included)
(cd 01_baseline_scoring/src && python b4_score.py && python b6_qa_checks.py && python b5_validate_original_scores.py)

# stage 02
(cd 02_multiscale/src && python c1_availability.py && python c4_scores.py && python c5_stability.py && python c6_weighting_protocolB.py)
(cd 02_multiscale && python -m pytest -q tests)

# stages 03 and 04 (each ends with its test script)
(cd 03_robustness/src && bash run_all.sh)
(cd 04_operational_projects/src && bash run_all.sh)

# stage 05 (after placing the downloaded and extracted heat-network file as described in its data README)
(cd 05_france_network_proximity && python src/f2b_01_labels.py && python src/f2b_02_evaluate.py && python src/f2b_03_figures.py && python src/test_f2b.py)

# stage 06 and manuscript assets
(cd 06_gaussian_benchmark/src && python gaussian_rank_benchmark.py)
(cd 07_manuscript_assets/src && python make_tables.py && python make_figures.py)
```

All random components use fixed seeds (the Gaussian-copula benchmark: seed 20261107, 2,000 replicates; the France bootstrap: 2,000 resamples of 50 km blocks). Scripts write only inside their own stage folder (temporary files in a stage-local `.tmp/`, ignored by Git).

## Verification

The pipeline of stages 02–07 was re-run in a clean copy of this repository on macOS (arm64) with Python 3.9.6, pandas 2.3.1, NumPy 2.0.2, SciPy 1.13.1, GeoPandas 1.0.1 and Matplotlib 3.9.4. All stored result tables (CSV) and the six manuscript figures were reproduced byte-for-byte; the only difference was the operating-system version string recorded in `06_gaussian_benchmark/results/benchmark_meta.json`. Test scripts: 13 (stage 02, pytest), plus the plain-assert tests of stages 03, 04 and 05, all passed.

## Data, attribution and licences

* **Code:** MIT licence (`LICENSE`).
* **Data:** the data files are derived from OpenStreetMap and are therefore made available under the **Open Database Licence (ODbL) 1.0** (`LICENSE-ODbL-1.0.txt`), with the attribution below. They also contain values derived from datasets released under CC BY 4.0 or the Licence Ouverte 2.0, which must be credited as listed in `DATA_SOURCES.md` and `LICENSE-DATA.md`.

> Contains information from OpenStreetMap, © OpenStreetMap contributors, available under the Open Database Licence (https://www.openstreetmap.org/copyright).
* Operational-project evidence (`04_operational_projects/results/e1_external_evidence_table.csv`) cites public web pages with retrieval dates; the pages remain the property of their publishers.

## Citation

See `CITATION.cff`. A persistent identifier for the repository has not been requested.

## Contact

Corresponding author: see the manuscript.
