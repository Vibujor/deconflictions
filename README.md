# A Catalogue of Deconfliction Actions Extracted from Historical ADS-B Data

## Overview

This code corresponds to the work published in the 2023 OpenSky Symposium.

## Dependencies

```
python >= 3.10
traffic >= 2.8.2
```

## Folder Structure

- The **data/** folder is made to store source files that were used to extract deviations:

  - `adsb_track_A2207.parquet`: ADS-B data corresponding to the AIRAC cycle 2207 (July 14 to August 10, 2022) in the Air Control Center of Bordeaux, France (LFBBDX).
  - `metadata.parquet`: Contains information to assign a unique identifier and a flight plan to each flight in adsb_track_A2207.parquet.

- The **results/** folder contains deviations extracted from flight data and metadata.

- The **src/** folder contains:

  - `format_data.py`: Preprocesses ADS-B flight data.
  - `extract_deviations.py`: Extracts deviations from data produced in format_data.py.
  - `functions_heuristic.py`: Provides a function that predicts a trajectory based on its associated flight plan.
  - `draw_figures.py` Provides functions to draw and save figures using the files from data/ and results/.

- The **fig/** folder contains figures created with functions from draw_figures.py.
