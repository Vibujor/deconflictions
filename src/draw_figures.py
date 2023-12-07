import argparse
from datetime import timedelta
from typing import Any, cast

import altair as alt
import matplotlib.pyplot as plt
from cartes.crs import Lambert93

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights
from sklearn.utils import check_array
from functions_heuristic import predict_fp
from traffic.core import Flight, FlightPlan, Traffic
from traffic.core.flight import Position
from traffic.core.mixins import DataFrameMixin

import numpy as np
import pandas as pd


class MedianKNNRegressor(KNeighborsRegressor):  # https://stackoverflow.com/a/33716704
    def __init__(self, quantile: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.quantile = quantile

    def predict(self, X: Any) -> Any:
        X = check_array(X, accept_sparse="csr")

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.quantile(_y[neigh_ind], q=self.quantile, axis=1)
        else:
            raise NotImplementedError("weighted median")

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


# FIGURE 4
def plot_difference_scatter(
    stats: pd.DataFrame,
    figname: str = "fig/plot_difference",
    threshold: int = 50,
    quantile: float = 0.5,
    n_neighbors: int = 100,
) -> None:
    df = stats.query(f"min_f_dist < {threshold} and min_fp_dist < {threshold}")
    x = df.min_fp_dist.values
    y = df.difference.values

    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    plt.scatter(x, y, alpha=0.1)

    xinp = np.expand_dims(x, 1)

    def draw(q: float) -> KNeighborsRegressor:
        """
        Fit and return a MedianKNNRegressor.

        Parameters:
        - q: Quantile for KNN regression.

        Returns:
        - KNeighborsRegressor: Fitted KNN regression model.
        """
        model = MedianKNNRegressor(
            n_neighbors=n_neighbors, weights="uniform", quantile=q
        )
        model.fit(xinp, y)
        return model

    z = sorted(zip(x, draw(quantile).predict(xinp)), key=lambda x: x[0])
    sorted_x, sorted_draw = zip(*z)
    plt.plot(sorted_x, sorted_draw, c="black", alpha=1, linewidth=3.5)

    plt.xlabel("Min_pred [NM]")
    plt.ylabel("Diff [NM]")
    plt.title("")

    plt.vlines(x=8, ymin=y.min(), ymax=y.max(), color="red", linestyles="dashed")
    plt.savefig(figname)


# FIGURE 3
def plot_layered_chart(
    stats: pd.DataFrame, figname: str = "fig/plot_layered_chart"
) -> None:
    source = stats[["min_fp_dist", "min_f_dist"]]
    source = source.dropna()
    chart2 = (
        alt.Chart(source.query("min_fp_dist<40 and min_f_dist<40"))
        .transform_fold(
            ["min_fp_dist", "min_f_dist"], as_=["Experiment", "Measurement"]
        )
        .transform_calculate(
            Experiment_Label="datum.Experiment == 'min_fp_dist' ? 'Predicted' : 'Actual'"
        )
        .mark_area(opacity=0.3, interpolate="step", binSpacing=0)
        .encode(
            alt.X("Measurement:Q", title="Separation (NM)").bin(maxbins=100),
            alt.Y("count()", title="Count").stack(None),
            alt.Color("Experiment_Label:N", title="Values"),
        )
        .properties(height=150)
    )

    chart2.save(f"{figname}.pdf")  # requires vl-convert-python


# FIGURE 1
def plot_compare_fp_traj(
    f1: Flight,
    f2: Flight,
    fp: FlightPlan,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    figname: str = "fig/plot_compare_fp_traj",
) -> None:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, subplot_kw=dict(projection=Lambert93())
    )
    pred_fp = predict_fp(
        f2,
        fp,
        f2.start + timedelta(seconds=35),
        f2.stop,
        minutes=50,
        min_distance=150,
    )

    # AX 1 : flight plan
    for i in fp.all_points[4:12]:
        i.plot(
            ax1,
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    ax1.spines["geo"].set_visible(False)

    # AX 2 : flight plan + prediction
    for i in fp.all_points[4:12]:
        i.plot(
            ax2,
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    pred_fp.plot(ax2, color="#4c79a8")
    ax2.spines["geo"].set_visible(False)

    # AX 3 : flight plan + straight trajectory
    for i in fp.all_points[4:12]:
        i.plot(
            ax3,
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    f1.plot(ax3, color="#4c79a8")
    ax3.spines["geo"].set_visible(False)

    # AX 4 : flight plan + deviated trajectory
    for i in fp.all_points[4:12]:
        i.plot(
            ax4,
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    f2.plot(ax4, color="#4c79a8")
    cast(Flight, f2.between(t1, t2)).plot(ax4, color="#f58518")
    ax4.spines["geo"].set_visible(False)
    fig.savefig(figname, transparent=True)


# FIGURE 5
def plot_conflict(
    f1: Flight,
    f2: Flight,
    fp: FlightPlan,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    ratio: float,
    figname: str = "fig/conflict.png",
) -> None:
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    (a1,) = cast(
        Flight, f1.between(t1 - pd.Timedelta("5T"), t2 + pd.Timedelta("5T"))
    ).plot(ax, zorder=1)
    (a2,) = cast(
        Flight, f2.between(t1 - pd.Timedelta("5T"), t2 + pd.Timedelta("5T"))
    ).plot(ax, color="#54a24b")

    cast(Flight, f1.between(t1, t2)).plot(ax, color="#f58518", zorder=2)

    cast(Flight, f1.at(t1)).plot(
        ax,
        color=a1.get_color(),
        text_kw=dict(
            s=f"{f1.callsign}\nFL{f1.altitude_max//100:.0f}",
        ),
        zorder=3,
    )
    cast(Flight, f2.at(t1)).plot(
        ax,
        color=a2.get_color(),
        text_kw=dict(
            s=f"{f2.callsign}\nFL{f2.altitude_max//100:.0f}",
        ),
    )

    cast(Position, f1.at(t1 + ratio * (t2 - t1))).plot(
        ax, color=a1.get_color(), text_kw=dict(s=None), zorder=2
    )
    cast(Position, f2.at(t1 + ratio * (t2 - t1))).plot(
        ax, color=a2.get_color(), text_kw=dict(s=None)
    )
    f1_fp = metadata_simple[cast(str, f1.flight_id)]

    pred_fp = predict_fp(
        f1,
        cast(FlightPlan, f1_fp),
        t1,
        t2,
        minutes=(ratio * (t2 - t1)).seconds / 60,
    )
    pred_fp.data["track"] = (
        cast(Flight, f1.before(t1)).forward(ratio * (t2 - t1)).data.track.iloc[0]
    )
    cast(Position, pred_fp.at()).plot(ax, color="#bab0ac", text_kw=dict(s=None))
    pred_fp.plot(ax, color="#bab0ac", ls="dashed")

    ax.spines["geo"].set_visible(False)

    # fig.set_tight_layout(True)
    fig.savefig(figname, transparent=True)


# FIGURE 2
def plot_compare_preds(
    f1: Flight,
    f2: Flight,
    fp: FlightPlan,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    ratio: float,
    figname: str = "fig/compare_preds.png",
) -> None:
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

    pred_fp = predict_fp(
        f1,
        fp,
        t1,
        t2,
        minutes=20,
        min_distance=150,
    )

    pred_fp.plot(ax, color="#88d27a", ls="dashed", zorder=1)

    for i in fp.all_points[5:10]:
        i.plot(
            ax,
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                color="#79706e",
                ha="left",
                size=12,
            ),
        )

    (a1,) = cast(
        FlightPlan,
        f1.between(t1 - pd.Timedelta("5T"), t2 + pd.Timedelta("20T")),
    ).plot(ax, zorder=3)
    (a2,) = cast(
        FlightPlan,
        f2.between(t1 - pd.Timedelta("5T"), t2 + pd.Timedelta("20T")),
    ).plot(ax)

    cast(Position, f1.at(t1)).plot(
        ax,
        color=a1.get_color(),
        text_kw=dict(
            s=f"{f1.callsign}\nFL{f1.altitude_max//100:.0f}",
            # fontproperties=fp,
        ),
        zorder=4,
    )
    cast(Position, f2.at(t1)).plot(
        ax,
        color=a2.get_color(),
        text_kw=dict(
            s=f"{f2.callsign}\nFL{f2.altitude_max//100:.0f}",
            # fontproperties=fp,
        ),
    )

    cast(Position, f1.at(t1 + ratio * (t2 - t1))).plot(
        ax, color=a1.get_color(), text_kw=dict(s=None), zorder=3
    )
    cast(Position, f2.at(t1 + ratio * (t2 - t1))).plot(
        ax, color=a2.get_color(), text_kw=dict(s=None)
    )

    cast(Flight, f1.before(t1)).forward(minutes=20).plot(
        ax, color="#bab0ac", ls="dashed"
    )  # straight-line pred

    cast(Position, cast(Flight, f1.before(t1)).forward(ratio * (t2 - t1)).at()).plot(
        ax, color="#bab0ac", text_kw=dict(s=None)
    )

    ax.spines["geo"].set_visible(False)

    # fig.set_tight_layout(True)
    fig.savefig(figname, transparent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating figures.")

    parser.add_argument("--t_path", type=str, help="Path to trajectory data")
    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Path to metadata file containing flight ids and flight plans",
    )
    parser.add_argument("--stats_path", type=str, help="Path to deviations file")
    args = parser.parse_args()

    t = Traffic.from_file(args.t_path)
    metadata = pd.read_parquet(args.metadata_path)
    stats = pd.read_parquet(args.stats_path)

    # PLOT LAYERED CHART
    stats["difference"] = stats["min_f_dist"] - stats["min_fp_dist"]
    stats = stats[stats.min_f_dist != 0.0]
    plot_layered_chart(stats)

    # PLOT SCATTER
    plot_difference_scatter(stats)

    # PLOT COMPARE
    f = t["AA39047319"]
    dev = t["AA38880885"]
    assert dev is not None
    assert f is not None

    class Metadata(DataFrameMixin):
        def __getitem__(self, key: str) -> None | FlightPlan:
            df = self.data.query(f'flight_id == "{key}"')
            if df.shape[0] == 0:
                return None
            return FlightPlan(df.iloc[0]["route"])

    metadata_simple = Metadata(
        metadata.groupby("flight_id", as_index=False)
        .last()
        .eval("icao24 = icao24.str.lower()")
    )

    fp = metadata_simple[cast(str, f.flight_id)]
    plot_compare_fp_traj(
        f,
        dev,
        cast(FlightPlan, fp),
        pd.Timestamp("2022-07-14 09:17:40+00:00"),
        pd.Timestamp("2022-07-14 09:26:05+00:00"),
    )

    # PLOT CONFLICT
    plot_conflict(
        cast(Flight, t["AA38871389"]),
        cast(Flight, t["AA38894800"]),
        cast(
            FlightPlan,
            metadata_simple[cast(str, cast(Flight, t["AA38871389"]).flight_id)],
        ),
        pd.Timestamp("2022-07-14 13:43:52+00:00"),
        pd.Timestamp("2022-07-14 13:53:43+00:00"),
        0.8,
        "fig/conflict1.png",
    )
    plot_conflict(
        cast(Flight, t["AA38865857"]),
        cast(Flight, t["AA38889279"]),
        cast(
            FlightPlan,
            metadata_simple[cast(str, cast(Flight, t["AA38871389"]).flight_id)],
        ),
        pd.Timestamp("2022-07-14 12:27:40+00:00"),
        pd.Timestamp("2022-07-14 12:37:54+00:00"),
        0.9,
        "fig/conflict2.png",
    )

    # PLOT COMPARE PREDS
    first = t["AA38880885"]
    second = t["AA38882693"]
    assert first is not None
    assert second is not None

    t1 = pd.Timestamp("2022-07-14 09:17:40+00:00")
    t2 = pd.Timestamp("2022-07-14 09:26:05+00:00")
    plot_compare_preds(
        first,
        second,
        cast(FlightPlan, metadata_simple[cast(str, first.flight_id)]),
        t1,
        t2,
        ratio=0.8,
    )
