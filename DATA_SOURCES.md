# Data sources and how to obtain them

| Indicator / use | Source | Access |
|---|---|---|
| Data-centre inventory (1,216 records; tags `telecom=data_center` or `building=data_center`; confirmed to exist on 31 Dec 2024) | OpenStreetMap contributors (ODbL) | https://www.openstreetmap.org ; included as part of `data/inputs/inventory_indicator_master_87col.csv` |
| Population density, built-up share | GHS-POP and GHS-BUILT-S, R2023A, 2025 epochs, 100 m (Schiavina et al. 2023; Pesaresi et al. 2023, doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE and doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA) | https://ghsl.jrc.ec.europa.eu |
| Built-up share (second measure) | Dynamic World 10 m, 2024 modal land-cover class (Brown et al. 2022, doi:10.1038/s41597-022-01307-4) | Google Earth Engine |
| Heat-demand density | Hotmaps `heat_tot_curr_density` (hectare level, reference year 2015; CC BY 4.0; Müller et al. 2019, doi:10.3390/en12244789) | https://gitlab.com/hotmaps/buildings/heat/heat_tot_curr_density |
| Heating degree days 2024 | ERA5-Land 2 m air temperature (Muñoz-Sabater et al. 2021, doi:10.5194/essd-13-4349-2021), Eurostat 18/15 °C convention | https://cds.climate.copernicus.eu |
| Anchor loads (242,631 hospitals, education, care facilities, sports centres) | OpenStreetMap, Geofabrik country extracts downloaded in 2026 (exact date not recorded); tags listed in manuscript Section 2.2 | https://download.geofabrik.de |
| Strategic roads | OpenStreetMap, Geofabrik dated extracts of 1 Jan 2025 (`*-250101.osm.pbf`, OSM data timestamp 2025-01-01T21:21Z). URLs, sizes, SHA-256: `01_baseline_scoring/data/geofabrik_250101_sources.json`, `osm_snapshot_manifest.json` | https://download.geofabrik.de/europe/ |
| Heat-network traces (France) | France Chaleur Urbaine, *Tracés des réseaux de chaleur et de froid*, file of 16 Oct 2026 (Licence Ouverte 2.0), data.gouv.fr | see `05_france_network_proximity/data/README.md` |
| Country outlines for Figure 1 | Natural Earth 1:50m admin-0 | https://www.naturalearthdata.com |
| Documented heat-reuse projects | Public operator, utility, municipal and press pages; URLs and dates in `04_operational_projects/results/e1_external_evidence_table.csv` | as listed |

The master table was assembled outside this repository (see README, "Not included"). Its column names are documented in the headers of the file and in `01_baseline_scoring/src/scoring.py`.

Licences and required credits: `LICENSE-DATA.md`.
