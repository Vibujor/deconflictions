import argparse
from pathlib import Path
from traffic.data import aixm_airspaces, opensky
from traffic.core import Traffic
import pandas as pd


def download_data(
    trajectory_folder: Path,
) -> None:
    opensky.history("", "", bounds=aixm_airspaces["LFBBBDX"])
    ...


def preprocess_data(
    trajectory_data: Path,
    metadata_file: Path,
    output_file: Path,
    extent: str,
    altitude_min: int = 20000,
) -> None:
    """
    Formats trajectory data based on metadata.

    :param trajectory_data: Path to trajectory data file
    :param metadata_file: Path to metadata file
    :param output_file: Path to output file
    :param extent: Extent
    :param altitude_min: Minimum altitude
    """
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
    # we drop onground value (useless)
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
    parser = argparse.ArgumentParser(description="Preprocess trajectory data.")

    parser.add_argument(
        "trajectory_data", type=Path, help="Path to the trajectory data file"
    )
    parser.add_argument("metadata_file", type=Path, help="Path to the metadata file")
    parser.add_argument("output_file", type=Path, help="Path to the output file")
    parser.add_argument("extent", type=str, help="Extent parameter, for example LFBBDX")
    parser.add_argument(
        "--altitude_min",
        type=int,
        default=20000,
        help="Minimum altitude (default: 20000)",
    )

    args = parser.parse_args()

    # Call the preprocess_data function directly with the parsed arguments
    preprocess_data(
        trajectory_data=args.trajectory_data,
        metadata_file=args.metadata_file,
        output_file=args.output_file,
        extent=args.extent,
        altitude_min=args.altitude_min,
    )
