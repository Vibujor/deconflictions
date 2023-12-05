from pathlib import Path
from typing import Any, cast
import altair as alt
import pandas as pd
from sklearn.utils import check_array
from sklearn.neighbors._base import _get_weights

# from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
from traffic.core import Flight, FlightPlan
import matplotlib.font_manager as fm
from cartes.crs import Lambert93
from functions_heuristic import predict_fp
from datetime import datetime, timedelta

from traffic.core.mixins import DataFrameMixin


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


def plot_difference_scatter(
    stats: pd.DataFrame,
    figname: str = "plot_difference",
    threshold: int = 50,
    quantile: float = 0.5,
    n_neighbors: int = 100,
) -> None:
    """
    Plots difference between actual and predicted minimum separation as a function of the predicted closest separation.

    :param stats: DataFrame compiling all extracted deviations
    :param figname: Name of the output file
    :param threshold: Threshold for maximum separation to increase legibility
    :param quantile: Quantile for regressor (0.5 for median)
    :param n_neighbors: Number of neighbors in the regression

    :return: None
    """
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

    # Example usage:
    # plot_difference_scatter(stats)


def plot_layered_chart(
    stats: pd.DataFrame, figname: str = "plot_layered_chart"
) -> None:
    """
    Plots distribution of actual and predicted lateral separation.

    :param stats: DataFrame compiling all extracted deviations
    :param figname: Name of the output file

    :return: None
    """
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

    chart2.save(f"{figname}.pdf")  # needs vl-convert-python


def plot_compare_fp_traj(
    f1: Flight,
    f2: Flight,
    fp: FlightPlan,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    figname: str = "plot_compare_fp_traj",
) -> None:
    """
    Plots flight plan, predicted trajectory according to flight plan, usual trajectory and deviated trajectory.

    :param f1: Aligned trajectory
    :param f2: Deviated trajectory
    :param fp: Associated flight plan
    :param t1: Start time of deviation in f2
    :param t2: Stop time of deviation in f2
    :param figname: Name of the output file

    :return: None
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, subplot_kw=dict(projection=Lambert93())
    )
    pred_fp = predict_fp(
        f1,
        fp,
        f1.start + timedelta(seconds=35),
        f1.stop,
        minutes=50,
        min_distance=150,
    )
    # AX 3 : flight plan + straight trajectory
    for i in fp.all_points[4:12]:
        i.plot(
            ax3,
            # color="#4c78a8",
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                # color="#9ecae9",
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    f1.plot(ax3, color="#4c79a8")
    ax3.spines["geo"].set_visible(False)

    # AX 2 : flight plan + prediction
    for i in fp.all_points[4:12]:
        i.plot(
            ax2,
            # color="#4c78a8",
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                # color="#9ecae9",
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    pred_fp.plot(ax2, color="#4c79a8")
    ax2.spines["geo"].set_visible(False)

    # AX 1 : flight plan
    for i in fp.all_points[4:12]:
        i.plot(
            ax1,
            # color="#4c78a8",
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                # color="#9ecae9",
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    ax1.spines["geo"].set_visible(False)

    # AX 4 : flight plan + deviated trajectory
    for i in fp.all_points[4:12]:
        i.plot(
            ax4,
            # color="#4c78a8",
            color="#79706e",
            marker="x",
            s=20,
            text_kw=dict(
                # color="#9ecae9",
                color="#79706e",
                ha="left",
                size=12,
            ),
        )
    f2.plot(ax4, color="#4c79a8")
    cast(Flight, f2.between(t1, t2)).plot(ax4, color="#f58518")
    ax4.spines["geo"].set_visible(False)
    fig.savefig(figname, transparent=True)


# TEST LAYERED
# stats = pd.read_parquet("stats_devs_pack/para")
# stats["difference"] = stats["min_f_dist"] - stats["min_fp_dist"]
# stats = stats[stats.min_f_dist != 0.0]
# # plot_difference_scatter(stats)
# plot_layered_chart(stats)


# TEST PLOT_COMPARE
# from traffic.core import Traffic

# t = Traffic.from_file("./t2_0722_noonground2.parquet")
# assert t is not None
# f = t["AA39013366"]
# f = t["AA39047319"]
# dev = t["AA38880885"]
# assert dev is not None
# assert f is not None


# class Metadata(DataFrameMixin):
#     def __getitem__(self, key: str) -> None | FlightPlan:
#         df = self.data.query(f'flight_id == "{key}"')
#         if df.shape[0] == 0:
#             return None
#         return FlightPlan(df.iloc[0]["route"])


# metadata = pd.read_parquet("A2207_old.parquet")

# metadata_simple = Metadata(
#     metadata.groupby("flight_id", as_index=False)
#     .last()
#     .eval("icao24 = icao24.str.lower()")
# )
# fp = metadata_simple[cast(str, f.flight_id)]
# plot_compare_fp_traj(
#     f,
#     dev,
#     cast(FlightPlan, fp),
#     pd.Timestamp("2022-07-14 09:17:40+00:00"),
#     pd.Timestamp("2022-07-14 09:26:05+00:00"),
# )