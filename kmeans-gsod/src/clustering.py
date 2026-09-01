"""
K-Means clustering of station climate fingerprints, plus the diagnostics needed
to defend the choice of k and to validate the result.

The modelling decisions worth knowing about:

  * Features are standardized first. K-Means minimises Euclidean distance, and
    precipitation runs to hundreds of millimetres while wet-day fraction sits
    between 0 and 1. Unscaled, rainfall totals would decide every cluster.
  * k is chosen from the elbow of the inertia curve, detected geometrically, and
    three other validation metrics are reported next to it. On this dataset
    silhouette is almost flat, roughly 0.21 to 0.23 across every k, because
    climate is a continuum rather than a set of disjoint blobs -- there is no
    clean gap between "warm temperate" and "subtropical". Taking the arithmetic
    maximum of a flat curve picks a k on a difference of under 0.001, which is
    noise. See find_elbow.
  * n_init is raised well above the default. K-Means converges to a local
    optimum determined by its starting centroids; multiple restarts make the
    result stable across runs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)


def scale_features(
    table: pd.DataFrame, columns: list[str] | None = None
) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize the clustering features to zero mean and unit variance.

    Returns the scaled matrix and the fitted scaler, so that cluster centroids
    can later be translated back into real units for reporting.
    """
    columns = columns or config.FEATURE_COLUMNS
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise KeyError(f"Feature table is missing columns: {missing}")

    values = table[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains NaN or infinity; clean it first")

    scaler = StandardScaler().fit(values)
    return scaler.transform(values), scaler


def evaluate_k_range(
    matrix: np.ndarray, k_range=config.K_RANGE, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Fit K-Means across a range of k and score each fit three ways.

    silhouette        higher is better, in [-1, 1]
    calinski_harabasz higher is better
    davies_bouldin    lower is better

    Three metrics rather than one because they disagree sometimes, and a k that
    wins on all three is a much safer choice to defend in a write-up.
    """
    if len(matrix) <= max(k_range):
        raise ValueError(
            f"Cannot fit k up to {max(k_range)} with only {len(matrix)} samples"
        )

    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=config.N_INIT, random_state=seed)
        labels = model.fit_predict(matrix)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            }
        )
        logger.debug("k=%d silhouette=%.4f", k, rows[-1]["silhouette"])

    return pd.DataFrame(rows).set_index("k")


def rank_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each candidate k on every metric, 1 being best, and average the ranks.

    Ranking rather than averaging the raw scores, because the three metrics are
    on entirely different scales: silhouette sits in [-1, 1], Calinski-Harabasz
    runs into the tens or hundreds, and Davies-Bouldin is better when lower.
    Ranks make them comparable without inventing a weighting.
    """
    if diagnostics.empty:
        raise ValueError("No diagnostics to rank")

    ranks = pd.DataFrame(index=diagnostics.index)
    ranks["silhouette_rank"] = diagnostics["silhouette"].rank(ascending=False)
    ranks["calinski_rank"] = diagnostics["calinski_harabasz"].rank(ascending=False)
    ranks["davies_bouldin_rank"] = diagnostics["davies_bouldin"].rank(ascending=True)
    ranks["mean_rank"] = ranks.mean(axis=1)
    return ranks


def choose_k_by_silhouette(diagnostics: pd.DataFrame) -> int:
    """
    Select k purely by the highest silhouette score.

    Kept for comparison and reported alongside the consensus choice. When the
    silhouette curve is flat this is unstable, which is exactly why it is not the
    method used by default.
    """
    if diagnostics.empty:
        raise ValueError("No diagnostics to choose from")
    return int(diagnostics["silhouette"].idxmax())


def find_elbow(diagnostics: pd.DataFrame) -> int:
    """
    Locate the elbow of the inertia curve geometrically.

    Both axes are rescaled to [0, 1], a straight chord is drawn from the first
    point to the last, and the k lying furthest below that chord is the elbow.
    This is the standard "maximum distance to the chord" construction, and it
    replaces the usual practice of squinting at the plot with something
    reproducible that a reviewer can check.

    Why the elbow and not the best silhouette: adding clusters can only reduce
    inertia, so the curve always falls, and the elbow marks where the returns
    stop being worth the extra cluster. Silhouette and Davies-Bouldin tend to
    keep improving as k grows, which on continuous data like climate pushes the
    answer to whichever end of the search range you happened to stop at.

    Returns the smallest k if the curve is straight enough to have no elbow.
    """
    if diagnostics.empty:
        raise ValueError("No diagnostics to search for an elbow")
    if len(diagnostics) < 3:
        return int(diagnostics.index.min())

    k_values = diagnostics.index.to_numpy(dtype=float)
    inertia = diagnostics["inertia"].to_numpy(dtype=float)

    k_span = k_values.max() - k_values.min()
    inertia_span = inertia.max() - inertia.min()
    if k_span == 0 or inertia_span == 0:
        return int(diagnostics.index.min())

    x = (k_values - k_values.min()) / k_span
    y = (inertia - inertia.min()) / inertia_span

    # Chord runs from (0, 1) to (1, 0), so the line is x + y = 1. Points below it
    # give a positive residual; the largest is the elbow.
    below_chord = 1.0 - x - y
    return int(k_values[int(np.argmax(below_chord))])


def choose_k(diagnostics: pd.DataFrame) -> int:
    """
    Select the number of clusters. The elbow of the inertia curve is the primary
    criterion; see find_elbow for why, and choose_k_by_silhouette for the
    alternative that is reported alongside it for comparison.
    """
    return find_elbow(diagnostics)


def fit_kmeans(
    matrix: np.ndarray, k: int, seed: int = config.RANDOM_SEED
) -> tuple[KMeans, np.ndarray]:
    """Fit the final model and return it alongside its cluster assignments."""
    model = KMeans(n_clusters=k, n_init=config.N_INIT, random_state=seed)
    labels = model.fit_predict(matrix)

    # Silhouette needs at least two clusters to compare against, so guard k=1
    # rather than letting sklearn raise from inside a log statement.
    if k > 1:
        logger.info(
            "Fitted k=%d, inertia=%.2f, silhouette=%.4f",
            k,
            model.inertia_,
            silhouette_score(matrix, labels),
        )
    else:
        logger.info(
            "Fitted k=1, inertia=%.2f (silhouette undefined for a single cluster)",
            model.inertia_,
        )

    return model, labels


def profile_clusters(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Summarise each cluster: size, mean of every feature, and mean geography.

    The geography columns were never shown to the model, so they are the honest
    test of whether the clusters mean anything.
    """
    if len(labels) != len(table):
        raise ValueError(
            f"Got {len(labels)} labels for {len(table)} stations; lengths must match"
        )

    annotated = table.copy()
    annotated["cluster"] = labels

    columns = config.FEATURE_COLUMNS + config.REPORTING_COLUMNS
    profile = annotated.groupby("cluster")[columns].mean()
    profile.insert(0, "n_stations", annotated.groupby("cluster").size())

    return profile.sort_values("temp_mean_c")


def variance_explained(
    table: pd.DataFrame, labels: np.ndarray, column: str
) -> float:
    """
    Share of variance in `column` explained by cluster membership (eta squared).

    Applied to absolute latitude, this is the validation headline: the model
    never saw a coordinate, so any latitude structure in the clusters was
    inferred from weather alone. Returns a value between 0 and 1.
    """
    values = table[column].to_numpy(dtype=float)
    mask = np.isfinite(values)
    values, groups = values[mask], np.asarray(labels)[mask]

    if len(values) < 2:
        return float("nan")

    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return float("nan")

    within = 0.0
    for group in np.unique(groups):
        members = values[groups == group]
        within += float(((members - members.mean()) ** 2).sum())

    return 1.0 - within / total


def representative_stations(
    table: pd.DataFrame,
    matrix: np.ndarray,
    model: KMeans,
    labels: np.ndarray,
    count: int = 4,
) -> dict[int, list[str]]:
    """
    The stations nearest each centroid, which are the ones that actually typify
    the cluster.

    Listing the first few members instead would be misleading: they arrive in
    download order, so an outlier sitting at the edge of a cluster can end up
    presented as its exemplar. An early version of this report described the
    tropical cluster using a station in northern Norway for exactly that reason.
    """
    representatives: dict[int, list[str]] = {}

    for cluster in range(model.n_clusters):
        members = np.flatnonzero(np.asarray(labels) == cluster)
        if members.size == 0:
            representatives[cluster] = []
            continue
        distances = np.linalg.norm(
            matrix[members] - model.cluster_centers_[cluster], axis=1
        )
        nearest = members[np.argsort(distances)][:count]
        representatives[cluster] = table["name"].iloc[nearest].astype(str).tolist()

    return representatives


def boundary_fragility(
    matrix: np.ndarray, model: KMeans, labels: np.ndarray, ratio: float = 0.9
) -> float:
    """
    Share of stations that sit almost as close to another centroid as their own.

    K-Means assigns every point to exactly one cluster however marginal the
    decision is, and this quantifies how many of those decisions were marginal.
    It matters here: the sample contains two separate stations at Montreal
    airport reporting within 0.2 C and 0.1 C of each other, and they were placed
    in different clusters. That is not a bug, it is what a hard partition does to
    continuous data, and it is worth reporting rather than hiding.

    A station counts as fragile when its distance to its own centroid exceeds
    `ratio` times its distance to the next nearest.
    """
    distances = model.transform(matrix)
    if distances.shape[1] < 2:
        return 0.0

    ordered = np.sort(distances, axis=1)
    nearest, runner_up = ordered[:, 0], ordered[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        closeness = np.where(runner_up > 0, nearest / runner_up, 0.0)

    return float((closeness > ratio).mean())


def centroids_in_real_units(
    model: KMeans, scaler: StandardScaler, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Convert scaled centroids back into physical units.

    Standardized centroid coordinates are unreadable in a report; degrees
    Celsius and millimetres are not.
    """
    columns = columns or config.FEATURE_COLUMNS
    return pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=columns,
        index=pd.Index(range(model.n_clusters), name="cluster"),
    )
