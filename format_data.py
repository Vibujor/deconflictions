from __future__ import annotations
from pathlib import Path
from traffic.data import (  # noqa: F401
    aixm_airspaces,
    aixm_navaids,
    aixm_airways,
)
from traffic.core import Traffic
import pandas as pd


def download_data(
    trajectory_folder: Path,
) -> None:
    ...


def preprocess_data(
    trajectory_data: Path,
    metadata_file: Path,
    output_file: Path,
    extent: str,
    altitude_min: int = 20000,
) -> None:
    # read_parquet accepts files or folders
    # metadata = pd.read_parquet("A2207_old.parquet")
    metadata = pd.read_parquet(metadata_file)
    metadata_simple = (
        metadata.groupby("flight_id", as_index=False)
        .last()
        .eval("icao24 = icao24.str.lower()")
    )
    # directory contains the trajectories as parquet files
    concatenated_df = pd.read_parquet(trajectory_data)

    # first filter on the data
    # we drop unground value (useless)
    t = (
        Traffic(concatenated_df)
        .drop(columns=["onground"])
        .clip(aixm_airspaces[extent])
        .query(f"altitude>{altitude_min}")
        .eval(max_workers=4)
    )

    # in t2, we assign flight ids using metadata_simple
    t2 = (
        t.iterate_lazy(iterate_kw=dict(by=metadata_simple))
        .resample("1s")
        .eval(desc="", max_workers=4)
    )

    # filter and resample
    assert t2 is not None
    t2 = t2.filter().resample("1s").eval(max_workers=4)
    assert t2 is not None

    t2.to_parquet(output_file)


if __name__ == "__main__":
    preprocess_data()
