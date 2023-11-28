from datetime import timedelta
from traffic.core.flight import Position
from traffic.core.flightplan import _Point
from traffic.core.geodesy import distance  # distance in meters
from traffic.core import Flight
from traffic.core import FlightPlan
import pandas as pd
from traffic.data import aixm_navaids
from typing import List, Union, Dict, cast


def predict_fp(
    flight: Flight,
    fp: FlightPlan,
    start: Union[str, pd.Timestamp],
    stop: Union[str, pd.Timestamp],
    minutes: int = 15,
    angle_precision: int = 2,
    min_distance: int = 150,
) -> Flight:
    """
    Predicts trajectory based on the corresponding flight plan.

    :param flight: Flight of interest
    :param fp: Flight plan corresponding to flight
    :param start: Start of deviation
    :param stop: Stop of deviation
    :param minutes: Number of minutes to predict
    :param angle_precision: Angle precision for alignment to navaids
    :param min_distance: Distance from which we consider a navpoint for alignment
    """
    data_points: Dict[str, List[Union[float, str, pd.Timestamp]]] = {
        "latitude": [],
        "longitude": [],
        "timestamp": [],
    }

    assert flight is not None
    hole = flight.between(start, stop, strict=False)
    assert hole is not None
    section = cast(Flight, flight.before(hole.start, strict=False)).last(minutes=20)
    assert section is not None
    gs = section.groundspeed_mean * 0.514444  # conversion to m/s

    data_points["latitude"].append(
        cast(Position, cast(Flight, flight.before(hole.start)).at_ratio(1)).latitude
    )
    data_points["longitude"].append(
        cast(Position, cast(Flight, flight.before(hole.start)).at_ratio(1)).longitude
    )
    data_points["timestamp"].append(
        cast(Position, cast(Flight, flight.before(hole.start)).at_ratio(1)).timestamp
    )

    navaids = fp.all_points

    # initialize first navaid
    g = section.aligned_on_navpoint(
        fp, angle_precision=angle_precision, min_distance=min_distance
    ).final()

    start_nav_name = cast(Flight, g).data.navaid.iloc[0]
    start_nav = next((point for point in navaids if point.name == start_nav_name), None)
    start_index = navaids.index(cast(_Point, start_nav))
    rest_navaids = navaids[start_index:]
    start_point = _Point(
        lat=cast(
            Position, cast(Flight, flight.before(hole.start)).at_ratio(1)
        ).latitude,
        lon=cast(
            Position, cast(Flight, flight.before(hole.start)).at_ratio(1)
        ).longitude,
        name=start_nav_name,
    )
    new_timestamp = hole.start
    # iterate on navaids
    for navaid in rest_navaids:
        dmin = distance(
            start_point.latitude,
            start_point.longitude,
            navaid.latitude,
            navaid.longitude,
        )
        t = int(dmin / gs)
        new_timestamp = new_timestamp + timedelta(seconds=t)
        start_point = navaid
        data_points["latitude"].append(navaid.latitude)
        data_points["longitude"].append(navaid.longitude)
        data_points["timestamp"].append(new_timestamp)
        # compute difference between hole.start and new_timestamp
        time_difference_seconds = (new_timestamp - hole.start).total_seconds()
        time_difference_minutes = time_difference_seconds / 60
        if time_difference_minutes > minutes and len(data_points["timestamp"]) > 1:
            break
    # create prediction as flight
    new_columns = {
        **data_points,
        "icao24": flight.icao24,
        "callsign": flight.callsign,
        "altitude": cast(Position, flight.at(hole.start)).altitude,
        "flight_id": flight.flight_id,
    }
    return Flight(pd.DataFrame(new_columns)).resample("1s").first(minutes * 60 + 1)
