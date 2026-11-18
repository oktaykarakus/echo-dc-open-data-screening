# Original data-extraction notebook

`Europe_DC_Map.ipynb` is the original Google Colab notebook by Isobelle Turvey in which the data-centre inventory was built and the non-road indicators were extracted (GHS-POP and GHS-BUILT-S population and built-up area, Dynamic World land cover, Hotmaps heat demand, ERA5-Land heating degree days, OpenStreetMap anchor loads) and from which the original strategic-road extraction was written. It is the origin of `../data/inputs/inventory_indicator_master_87col.csv`.

* It is published as a record of the original method. **It was not re-run** for this repository: it needs Google Colab, a Google Drive with the intermediate files, a Google Earth Engine project (the personal project identifier was replaced by `YOUR-EARTH-ENGINE-PROJECT`) and downloaded raster layers.
* Saved cell outputs and Colab metadata were removed; the code cells are otherwise unchanged.
* The scoring code of the original analysis was ported to `../01_baseline_scoring/src/scoring.py`; the later stages (`01_` to `07_`) are the verified reproducibility package.
