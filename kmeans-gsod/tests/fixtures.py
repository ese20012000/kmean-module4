"""
Synthetic GSOD data for the unit tests.

The tests must not hit the network: a test suite that fails because S3 is slow or
a station file moved is not testing our code. These helpers build DataFrames with
exactly the shape and quirks of a real GSOD file, including the missing-value
sentinels, so the cleaning logic is genuinely exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GSOD_COLUMNS = [
    "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
    "TEMP", "TEMP_ATTRIBUTES", "DEWP", "DEWP_ATTRIBUTES",
    "SLP", "SLP_ATTRIBUTES", "STP", "STP_ATTRIBUTES",
    "VISIB", "VISIB_ATTRIBUTES", "WDSP", "WDSP_ATTRIBUTES",
    "MXSPD", "GUST", "MAX", "MAX_ATTRIBUTES", "MIN", "MIN_ATTRIBUTES",
    "PRCP", "PRCP_ATTRIBUTES", "SNDP", "FRSHTT",
]


def make_raw_gsod(
    station_id: str = "72503014732",
    name: str = "LAGUARDIA AIRPORT, NY US",
    year: int = 2023,
    days: int = 365,
    temp_f=None,
    mean_temp_f: float = 55.0,
    amplitude_f: float = 25.0,
    latitude: float = 40.78,
    longitude: float = -73.88,
    elevation: float = 3.0,
    dewpoint_offset_f: float = 10.0,
    diurnal_f: float = 18.0,
    precip_in: float = 0.05,
    wind_knots: float = 8.0,
    frshtt: str = "000000",
    prcp_flag: str = "G",
    temp_sentinel_days: int = 0,
    precip_sentinel_days: int = 0,
    prcp_not_reported_days: int = 0,
) -> pd.DataFrame:
    """
    Build a raw GSOD frame in Fahrenheit and inches, exactly as S3 serves it.

    temp_f                  explicit daily temperatures, overriding the sine wave
    *_sentinel_days         days poisoned with GSOD's missing-value codes
    prcp_not_reported_days  days written as 0.00 with flag "I", GSOD's way of
                            saying "no precipitation data", which is the trap
                            that makes wet places look like deserts
    """
    dates = pd.date_range(f"{year}-01-01", periods=days, freq="D")

    if temp_f is None:
        # Annual cycle: coldest in January, warmest in July.
        day_index = np.arange(days)
        temp_f = mean_temp_f - amplitude_f * np.cos(2 * np.pi * day_index / 365.0)
    temp_f = np.asarray(temp_f, dtype=float)

    frame = pd.DataFrame(
        {
            "STATION": station_id,
            "DATE": dates.strftime("%Y-%m-%d"),
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "ELEVATION": elevation,
            "NAME": name,
            "TEMP": temp_f,
            "TEMP_ATTRIBUTES": 24,
            "DEWP": temp_f - dewpoint_offset_f,
            "DEWP_ATTRIBUTES": 24,
            "SLP": 1013.2,
            "SLP_ATTRIBUTES": 22,
            "STP": 9.6,
            "STP_ATTRIBUTES": 24,
            "VISIB": 9.5,
            "VISIB_ATTRIBUTES": 24,
            "WDSP": wind_knots,
            "WDSP_ATTRIBUTES": 24,
            "MXSPD": wind_knots + 6.0,
            "GUST": 999.9,          # routinely missing in real files
            "MAX": temp_f + diurnal_f / 2.0,
            "MAX_ATTRIBUTES": " ",
            "MIN": temp_f - diurnal_f / 2.0,
            "MIN_ATTRIBUTES": " ",
            "PRCP": precip_in,
            "PRCP_ATTRIBUTES": prcp_flag,
            "SNDP": 999.9,          # routinely missing in real files
            "FRSHTT": frshtt,
        },
        columns=GSOD_COLUMNS,
    )

    # Poison selected days with the real sentinel values.
    if temp_sentinel_days:
        frame.loc[: temp_sentinel_days - 1, ["TEMP", "DEWP", "MAX", "MIN"]] = 9999.9
    if precip_sentinel_days:
        frame.loc[: precip_sentinel_days - 1, "PRCP"] = 99.99

    # Flag I days: value 0.00, flag "I". Applied from the end of the year so they
    # do not collide with the sentinel days written from the start.
    if prcp_not_reported_days:
        tail = frame.index[-prcp_not_reported_days:]
        frame.loc[tail, "PRCP"] = 0.00
        frame.loc[tail, "PRCP_ATTRIBUTES"] = "I"

    return frame


def make_two_season_gsod(cold_c: float = 0.0, warm_c: float = 20.0, **kwargs):
    """
    A station that is exactly `cold_c` for the first half of the year and exactly
    `warm_c` for the second half.

    Used where a test needs a precisely known seasonality rather than an
    approximate one: the answer is warm_c - cold_c, to the decimal.
    """
    days = kwargs.pop("days", 365)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    cold_f = cold_c * 9.0 / 5.0 + 32.0
    warm_f = warm_c * 9.0 / 5.0 + 32.0
    temp_f = np.where(dates.month <= 6, cold_f, warm_f)
    return make_raw_gsod(temp_f=temp_f, days=days, **kwargs)


def make_station_catalogue() -> pd.DataFrame:
    """
    A fake S3 listing spanning four WMO blocks: 03 (UK), 47 (Japan), 72 (USA)
    and 94 (Australia), five stations each with distinct sizes.
    """
    rows = []
    for block in ["03", "47", "72", "94"]:
        for n in range(5):
            rows.append(
                {
                    "station_id": f"{block}{n:09d}",
                    "key": f"2023/{block}{n:09d}.csv",
                    "size_bytes": 10_000 + n * 1_000,
                }
            )
    return pd.DataFrame(rows)


def make_feature_table(n_per_group: int = 12, seed: int = 7) -> pd.DataFrame:
    """
    A feature table with three well-separated synthetic climate groups.

    Latitude is generated to correlate with the groups but is never a feature, so
    tests can verify that variance_explained() picks the relationship up.
    """
    from src import config

    rng = np.random.default_rng(seed)

    # Keyed by column name rather than by position, so that reordering or
    # renaming a feature in config breaks loudly instead of silently shuffling
    # the fixture's meaning.
    centres = {
        "tropical": {
            "temp_mean_c": 27.0, "temp_seasonality_c": 4.0, "diurnal_range_c": 9.0,
            "dewpoint_depression_c": 4.0, "precip_log_mm": float(np.log1p(1800.0)),
            "wet_day_fraction": 0.45, "snow_day_fraction": 0.002,
            "wind_mean_ms": 3.0, "fog_day_fraction": 0.20,
            "thunder_day_fraction": 0.30,
        },
        "temperate": {
            "temp_mean_c": 11.0, "temp_seasonality_c": 18.0, "diurnal_range_c": 9.0,
            "dewpoint_depression_c": 6.0, "precip_log_mm": float(np.log1p(750.0)),
            "wet_day_fraction": 0.30, "snow_day_fraction": 0.03,
            "wind_mean_ms": 4.5, "fog_day_fraction": 0.15,
            "thunder_day_fraction": 0.05,
        },
        "polar": {
            "temp_mean_c": -8.0, "temp_seasonality_c": 32.0, "diurnal_range_c": 8.0,
            "dewpoint_depression_c": 3.0, "precip_log_mm": float(np.log1p(300.0)),
            "wet_day_fraction": 0.20, "snow_day_fraction": 0.35,
            "wind_mean_ms": 5.0, "fog_day_fraction": 0.10,
            "thunder_day_fraction": 0.005,
        },
    }
    for group, centre in centres.items():
        if set(centre) != set(config.FEATURE_COLUMNS):
            raise AssertionError(
                f"fixture group '{group}' does not match config.FEATURE_COLUMNS; "
                f"missing {set(config.FEATURE_COLUMNS) - set(centre)}, "
                f"unexpected {set(centre) - set(config.FEATURE_COLUMNS)}"
            )

    latitudes = {"tropical": 8.0, "temperate": 45.0, "polar": 70.0}

    rows = []
    for index, (group, centre) in enumerate(centres.items()):
        for member in range(n_per_group):
            row = {
                column: value * (1.0 + rng.normal(0, 0.04))
                for column, value in centre.items()
            }
            row["precip_total_mm"] = float(np.expm1(row["precip_log_mm"]))
            row.update(
                station_id=f"{index}{member:010d}",
                name=f"{group.upper()} STATION {member}",
                latitude=latitudes[group] + rng.normal(0, 2.0),
                longitude=rng.uniform(-180, 180),
                elevation_m=abs(rng.normal(120, 60)),
                observation_days=365,
                true_group=group,
            )
            row["abs_latitude"] = abs(row["latitude"])
            rows.append(row)

    return pd.DataFrame(rows).set_index("station_id")
