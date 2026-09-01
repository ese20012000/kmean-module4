"""
Figures for the report. Four of them, each answering one question:

  1. How many clusters?          -> elbow and silhouette curves
  2. Where are the clusters?     -> stations plotted on latitude and longitude
  3. What defines each cluster?  -> standardized profile heatmap
  4. Do the clusters separate?   -> temperature against seasonality

Matplotlib only. No basemap or cartopy dependency, so the project installs and
runs anywhere without a geospatial toolchain. Plotting longitude against
latitude on an equal-aspect axis is enough for the continents to be recognisable
from a few hundred stations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display; PyCharm-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

PALETTE = plt.get_cmap("tab10")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_k_selection(diagnostics: pd.DataFrame, chosen_k: int, output_dir: Path) -> Path:
    """Elbow curve and silhouette curve side by side, with the chosen k marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(diagnostics.index, diagnostics["inertia"], marker="o", color="steelblue")
    ax1.axvline(chosen_k, color="crimson", linestyle="--", label=f"k = {chosen_k}")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow method")
    ax1.legend()

    ax2.plot(diagnostics.index, diagnostics["silhouette"], marker="o", color="darkorange")
    ax2.axvline(chosen_k, color="crimson", linestyle="--", label=f"k = {chosen_k}")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Mean silhouette score")
    ax2.set_title("Silhouette analysis (higher is better)")
    ax2.legend()

    return _save(fig, output_dir / "01_k_selection.png")


def plot_world_map(
    table: pd.DataFrame, labels: np.ndarray, cluster_names: dict, output_dir: Path
) -> Path:
    """
    Stations on a longitude/latitude grid, coloured by cluster.

    This is the validation figure. Coordinates were never given to the model, so
    any latitudinal banding visible here was derived from weather alone.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            table.loc[mask, "longitude"],
            table.loc[mask, "latitude"],
            s=55,
            alpha=0.85,
            color=PALETTE(cluster % 10),
            edgecolor="white",
            linewidth=0.5,
            label=f"{cluster}: {cluster_names.get(cluster, '')} (n={mask.sum()})",
        )

    for latitude in (-66.5, -23.4, 0, 23.4, 66.5):
        ax.axhline(latitude, color="grey", linewidth=0.4, linestyle=":")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "K-Means climate clusters, positioned geographically\n"
        "Coordinates were withheld from the model; dotted lines mark the "
        "tropics, equator and polar circles"
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    return _save(fig, output_dir / "02_world_map.png")


def plot_cluster_profiles(
    profile: pd.DataFrame, cluster_names: dict, output_dir: Path
) -> Path:
    """
    Heatmap of cluster feature means, standardized across clusters.

    Standardizing per feature is what makes the map readable: it shows which
    clusters are high or low on each feature relative to the others, rather than
    letting precipitation's larger numbers wash everything else out.
    """
    features = profile[config.FEATURE_COLUMNS]
    z_scores = (features - features.mean()) / features.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(11, 0.85 * len(profile) + 3))
    image = ax.imshow(z_scores.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")

    ax.set_xticks(range(len(config.FEATURE_COLUMNS)))
    ax.set_xticklabels(
        [config.FEATURE_NOTES[c] for c in config.FEATURE_COLUMNS],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels(
        [
            f"{cluster}: {cluster_names.get(cluster, '')} (n={int(row.n_stations)})"
            for cluster, row in profile.iterrows()
        ],
        fontsize=9,
    )

    # Annotate with the real values, since the colour only conveys rank.
    for row_index, cluster in enumerate(profile.index):
        for col_index, column in enumerate(config.FEATURE_COLUMNS):
            value = features.loc[cluster, column]
            text = f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=7)

    ax.set_title("Cluster climate profiles\n(colour = standardized across clusters, text = real units)")
    fig.colorbar(image, ax=ax, shrink=0.7, label="standard deviations from the mean cluster")

    return _save(fig, output_dir / "03_cluster_profiles.png")


def plot_feature_space(
    table: pd.DataFrame, labels: np.ndarray, cluster_names: dict, output_dir: Path
) -> Path:
    """
    The two most interpretable features against each other.

    Mean temperature and seasonality together capture most of what separates
    world climates, so this is where the cluster structure is easiest to see
    without any dimensionality reduction.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            table.loc[mask, "temp_mean_c"],
            table.loc[mask, "temp_seasonality_c"],
            s=60,
            alpha=0.85,
            color=PALETTE(cluster % 10),
            edgecolor="white",
            linewidth=0.5,
            label=f"{cluster}: {cluster_names.get(cluster, '')}",
        )

    ax.set_xlabel("Annual mean temperature (deg C)")
    ax.set_ylabel("Seasonality: warmest month minus coldest month (deg C)")
    ax.set_title("Climate clusters in the two most interpretable dimensions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    return _save(fig, output_dir / "04_feature_space.png")
