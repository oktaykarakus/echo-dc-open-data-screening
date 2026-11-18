# Heat-network data (France) — download instructions

Stage 05 uses the layer of existing heat-network line traces published by France Chaleur Urbaine on data.gouv.fr:
*Tracés des réseaux de chaleur et de froid* (Licence Ouverte 2.0), file `opendata-fcu_20261016.zip` (file of 16 October 2026; accessed 7 November 2026 — see `ACCESS_DATE.txt` and `dataset_metadata_20261107.json`).

1. Download the dataset from https://www.data.gouv.fr/datasets/traces-des-reseaux-de-chaleur-et-de-froid and save the archive here as `opendata-fcu_20261016.zip`.
2. Verify the checksum against `opendata-fcu_20261016.sha256` (`shasum -a 256 -c opendata-fcu_20261016.sha256`). The dataset may be updated by the publisher; if the checksum differs, results may differ slightly from those stored.
3. Extract the archive into `data/extracted/` so that `data/extracted/reseaux_de_chaleur.shp` and `data/extracted/reseaux_de_chaleur_sans_traces.shp` exist.

The archive and the extracted files are not redistributed here (see `.gitignore`); the derived outcome labels are in `../results/f2b_labels.csv`.
