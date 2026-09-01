"""
Feature engineering: turn a year of daily weather observations into one
"climate fingerprint" row per station.

This is the step that makes the clustering meaningful. K-Means on raw daily rows
would just cluster days, which tells us nothing interesting. Aggregating each
station's year into ten descriptive numbers means each point in the feature
space is a place, and the clusters become climate types.

Geography (latitude, longitude, elevation) is carried through but explicitly NOT
part of the feature set. It is held back as an independent check: if clusters
built from weather alone still align with latitude, the model found something
real rather than something we told it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class InsufficientDataError(ValueError):
    """Raised when a station has too few valid observations to characterise."""


def _seasonality(observations: pd.DataFrame) -> float:
    """
    Temperature range between the warmest and coldest month.

    This is the continentality signal. Coastal Lisbon and continental Astana can
    share an annual mean temperature while behaving completely differently
    across the year, and this single feature separates them.

    Months with too few readings are dropped so that a month represented by two
    unusually cold days cannot masquerade as the coldest month.
    """
    monthly = observations.groupby("month")["temp_c"].agg(["mean", "count"])
    monthly = monthly[monthly["count"] >= config.MIN_DAYS_PER_MONTH]

    if len(monthly) < config.MIN_MONTHS_FOR_SEASONALITY:
        raise InsufficientDataError(
            f"only {len(monthly)} usable months, need {config.MIN_MONTHS_FOR_SEASONALITY}"
        )

    return float(monthly["mean"].max() - monthly["mean"].min())


def build_station_features(observations: pd.DataFrame) -> dict:
    """
    Reduce one station's cleaned daily observations to a single feature row.

    Raises InsufficientDataError if the station does not meet the quality
    thresholds in config. Callers are expected to catch this and skip the
    station -- a sparse record produces a plausible-looking but meaningless
    fingerprint, which is worse than no record at all.
    """
    if observations.empty:
        raise InsufficientDataError("no observations")

    temp_days = int(observations["temp_c"].notna().sum())
    if temp_days < config.MIN_DAYS_WITH_TEMP:
        raise InsufficientDataError(
            f"only {temp_days} days with temperature, need {config.MIN_DAYS_WITH_TEMP}"
        )

    precip_days = int(observations["precip_mm"].notna().sum())
    if precip_days < config.MIN_DAYS_WITH_PRECIP:
        raise InsufficientDataError(
            f"only {precip_days} days with precipitation, need {config.MIN_DAYS_WITH_PRECIP}"
        )

    precip = observations["precip_mm"].dropna()
    annual_precip = float(precip.sum() * 365.0 / len(precip))
    latitude = float(observations["LATITUDE"].dropna().iloc[0])

    features = {
        # Identity and geography -- reporting and validation only.
        "station_id": str(observations["STATION"].iloc[0]),
        "name": str(observations["NAME"].iloc[0]) if "NAME" in observations else "",
        "latitude": latitude,
        "longitude": float(observations["LONGITUDE"].dropna().iloc[0]),
        "elevation_m": float(observations["ELEVATION"].dropna().iloc[0])
        if observations["ELEVATION"].notna().any()
        else np.nan,
        "abs_latitude": abs(latitude),
        "observation_days": int(len(observations)),
        # The ten clustering features.
        "temp_mean_c": float(observations["temp_c"].mean()),
        "temp_seasonality_c": _seasonality(observations),
        "diurnal_range_c": float(
            (observations["temp_max_c"] - observations["temp_min_c"]).mean()
        ),
        "dewpoint_depression_c": float(
            (observations["temp_c"] - observations["dewpoint_c"]).mean()
        ),
        # Scaled to a full year so that stations with a few missing days are not
        # penalised as though it simply did not rain on those days.
        "precip_total_mm": annual_precip,
        # The model sees the log, not the raw total.
        #
        # Rainfall is strongly right-skewed: this sample spans roughly 20 mm to
        # 3,500 mm, so a monsoon station sits several standard deviations out
        # even after standardization and drags centroids toward itself.
        # Standardizing equalises variance but does nothing about skew. On the
        # untransformed feature the clustering was driven almost entirely by
        # rainfall, which put a Norwegian station at 3.9 C in the same cluster as
        # the genuine tropics purely because both are wet. log1p pulls the tail
        # in and restores temperature to comparable influence. It is also the
        # conventional transform for precipitation, and log1p rather than log
        # because a genuinely dry year of 0 mm is legitimate.
        "precip_log_mm": float(np.log1p(annual_precip)),
        "wet_day_fraction": float((precip >= config.WET_DAY_THRESHOLD_MM).mean()),
        "snow_day_fraction": float(observations["flag_snow"].mean()),
        "wind_mean_ms": float(observations["wind_ms"].mean()),
        "fog_day_fraction": float(observations["flag_fog"].mean()),
        "thunder_day_fraction": float(observations["flag_thunder"].mean()),
    }

    return features


def build_feature_table(station_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the station-by-feature table from a mapping of station id to cleaned
    observations.

    Stations that fail the quality checks are logged and dropped. The returned
    table is indexed by station_id and is guaranteed to contain no NaN in the
    clustering columns, because K-Means cannot handle missing values.
    """
    rows = []
    rejected = 0

    for station_id, observations in station_frames.items():
        try:
            rows.append(build_station_features(observations))
        except (InsufficientDataError, KeyError, IndexError) as exc:
            rejected += 1
            logger.debug("Rejected station %s: %s", station_id, exc)

    if not rows:
        raise InsufficientDataError(
            "No station met the quality thresholds; loosen the limits in config.py"
        )

    table = pd.DataFrame(rows).set_index("station_id")

    before = len(table)
    table = table.dropna(subset=config.FEATURE_COLUMNS)
    dropped_for_nan = before - len(table)

    logger.info(
        "Feature table: %d stations kept, %d rejected on quality, %d dropped for NaN",
        len(table),
        rejected,
        dropped_for_nan,
    )
    return table


def describe_cluster(profile: pd.Series) -> str:
    """
    Attach a plain-English climate label to a cluster from its feature means.

    This is a transparent rule of thumb for the write-up, loosely following the
    logic behind the Koppen classification, not an output of the model. The
    clusters stand on their own; the labels exist so the executive summary can
    say "arid subtropical" instead of "cluster 3".
    """
    temp = profile["temp_mean_c"]
    seasonality = profile["temp_seasonality_c"]
    precip = profile["precip_total_mm"]
    dryness = profile["dewpoint_depression_c"]
    snow = profile["snow_day_fraction"]

    # Order matters, and two orderings were wrong before this one.
    #
    # Aridity is tested before continentality, because a desert can be strongly
    # seasonal: the Iranian plateau stations swing 27 C across the year on 211 mm
    # of rain. Testing seasonality first labelled them "temperate continental".
    #
    # Sub-freezing clusters are then split on seasonality, because two very
    # different climates both average below zero. Antarctic and Alaskan coastal
    # stations swing about 20 C; Fairbanks and inland Canada swing nearly 40. A
    # single "polar" label collapsed them together.
    if precip < 400 and dryness > 10:
        return "Arid / semi-arid"
    if temp < 0:
        return "Subarctic continental" if seasonality > 30 else "Polar / tundra"
    if seasonality > 30 or (snow > 0.10 and seasonality > 24):
        return "Cold continental"
    if precip < 500 and temp > 14 and seasonality < 18:
        # Dry but humid air: coastal deserts and Mediterranean regimes.
        return "Mediterranean / coastal dry"
    if temp > 22 and precip > 900:
        return "Tropical humid"
    if seasonality > 22:
        return "Temperate continental"
    if temp > 18:
        return "Warm subtropical"
    return "Maritime temperate"
