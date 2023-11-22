from __future__ import annotations
from traffic.data import (  # noqa: F401
    aixm_airspaces,
    aixm_navaids,
    aixm_airways,
)
from traffic.core import Traffic, Flight, FlightPlan
import pandas as pd

# from intervals import IntervalCollection
from pathlib import Path
import datetime

from typing import Any, Dict, cast  # noqa: F401
from functions_heuristic import predict_fp

# import os

from traffic.core.mixins import DataFrameMixin

import multiprocessing as mp
from typing import Tuple, List, Callable

extent = "LFBBBDX"
prefix_sector = "LFBB"
margin_fl = 50  # margin for flight level
altitude_min = 20000
angle_precision = 2
forward_time = 20
min_distance = 200
nbworkers = 60


def dist_lat_min(f1: Flight, f2: Flight) -> Any:
    try:
        if f1 & f2 is None:  # no overlap
            print(f"no overlap with {f2.flight_id}")
            return None
        return cast(pd.DataFrame, f1.distance(f2))["lateral"].min()
    except TypeError as e:
        print(
            f"exception in dist_lat_min for flights {f1.flight_id} and {f2.flight_id}"
        )
        return None


def extract_flight_deviations(
    flight: Flight,
    flightplan: FlightPlan,
    context_traffic: Traffic,
    margin_fl: int = 50,
    angle_precision: int = 2,
    min_distance: int = 200,
    forward_time: int = 20,
) -> None | pd.DataFrame:
    """
    Examines all deviations in flight and returns selected ones in a dataframe.

    :param flight: Flight of interest
    :param flightplan: Flight plan of flight
    :param context_traffic: Surrounding flights
    :param margin_fl: Margin in ft to check altitude stability
    :param angle_precision: Desired precision in alignment computation
    :param min_distance: Distance from which we consider a navpoint for alignment
    :param forward_time: Duration of trajectory prediction

    :return: None or DataFrame containing selected deviations
    """
    list_dicts = []
    for hole in flight - flight.aligned_on_navpoint(
        # metadata_simple[cast(str, flight.flight_id)],
        flightplan,
        angle_precision=angle_precision,
        min_distance=min_distance,
    ):
        temp_dict = hole.summary(["flight_id", "start", "stop", "duration"])
        temp_dict = {
            **temp_dict,
            **dict(
                min_f_dist=None,
                min_fp_dist=None,
                min_fp_id=None,
                min_fp_time=None,
                neighbour_id=None,
            ),
        }
        if (
            hole is not None
            and hole.duration > pd.Timedelta("120s")
            and hole.altitude_max - hole.altitude_min < margin_fl
            and hole.start > flight.start
            and hole.stop < flight.stop
        ):
            flight = flight.resample("1s")
            hole = hole.resample("1s")

            flmin = hole.altitude_min - margin_fl
            flmax = hole.altitude_max + margin_fl

            stop_neighbours = min(
                hole.start + pd.Timedelta(minutes=forward_time),
                flight.stop,
            )
            flight_interest = flight.between(hole.start, stop_neighbours)
            assert flight_interest is not None

            offlimits = flight_interest.query(f"altitude>{flmax} or altitude<{flmin}")
            # if there is at least one off-limits portion, we cut
            if offlimits is not None:
                istop = offlimits.data.index[0]
                flight_interest.data = flight_interest.data.loc[:istop]
                stop_neighbours = flight_interest.stop

            neighbours = (
                cast(Traffic, context_traffic - flight)
                .between(
                    start=hole.start,
                    stop=stop_neighbours,
                    strict=False,
                )
                .iterate_lazy()
                .query(f"{flmin} <= altitude <= {flmax}")
                .feature_gt("duration", datetime.timedelta(seconds=2))
                .eval()
            )

            pred_possible = flight.before(hole.start) is not None

            if neighbours is None and not pred_possible:
                continue

            if pred_possible:
                # compute prediction
                pred_fp = predict_fp(
                    flight,
                    flightplan,
                    hole.start,
                    hole.stop,
                    minutes=forward_time,
                    min_distance=min_distance,
                )

            if neighbours is not None:
                # distance to closest neighbor + flight_id + timestamp
                (min_f, idmin_f) = min(
                    (dist_lat_min(flight_interest, f), f.flight_id) for f in neighbours
                )
                temp_dict["neighbour_id"] = idmin_f
                temp_dict["min_f_dist"] = min_f
                # temp_dict["min_f_id"] = idmin_f
                df_dist = flight_interest.distance(neighbours[idmin_f])
                temp_dict["min_f_time"] = df_dist.loc[
                    df_dist.lateral == df_dist.lateral.min()
                ].timestamp.iloc[0]

                if pred_possible:
                    df_dist_fp = pred_fp.distance(neighbours[idmin_f])
                    temp_dict["min_fp_id"] = idmin_f
                    temp_dict["min_fp_dist"] = df_dist_fp.lateral.min()
                    temp_dict["min_fp_time"] = df_dist_fp.loc[
                        df_dist_fp.lateral == df_dist_fp.lateral.min()
                    ].timestamp.iloc[0]

            list_dicts.append(temp_dict)
    if len(list_dicts) == 0:
        return None
    deviations = pd.DataFrame(list_dicts)
    # we compute the difference between actual and predicted separation
    deviations["difference"] = deviations["min_f_dist"] - deviations["min_fp_dist"]
    # we clear the cases for which trajectories exist more than once
    deviations = deviations[deviations.min_f_dist != 0.0]
    return deviations


def do_parallel(
    # f: function,
    f: Callable[[Tuple[Traffic | Any, str]], None],
    datas: List[Tuple[Traffic | Any, str]],
    nworkers: int = mp.cpu_count() // 2,
) -> None:
    with mp.Pool(nworkers) as pool:
        # l = pool.map(f, datas, chunksize=1)
        pool.map(f, datas, chunksize=1)


# do_parallel(traitement, couples_datas, nbworkers)
