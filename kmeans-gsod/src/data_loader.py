"""
Acquisition and cleaning of NOAA GSOD station data from the AWS Registry of
Open Data.

Responsibilities, in order:
  1. list the station files available for a given year in the public S3 bucket,
  2. choose a globally distributed sample of stations,
  3. download them once and cache them on disk,
  4. parse each CSV and convert it into clean SI-unit observations.

Nothing in this module knows about clustering. Keeping acquisition separate from
modelling is what makes the unit tests in tests/test_data_loader.py possible
without touching the network.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src import config

logger = logging.getLogger(__name__)


class DataAcquisitionError(RuntimeError):
    """Raised when station data cannot be retrieved from S3."""


# --------------------------------------------------------------------------- #
# Unit conversions. GSOD reports in imperial units; everything downstream works
# in SI so that the cluster profiles are readable to an international audience.
# --------------------------------------------------------------------------- #
def fahrenheit_to_celsius(values):
    """Convert Fahrenheit to Celsius. NaN propagates unchanged."""
    return (values - 32.0) * 5.0 / 9.0


def inches_to_mm(values):
    """Convert inches to millimetres."""
    return values * 25.4


def knots_to_ms(values):
    """Convert knots to metres per second."""
    return values * 0.514444


# --------------------------------------------------------------------------- #
# S3 access
# --------------------------------------------------------------------------- #
def build_s3_client():
    """
    Create an anonymous S3 client.

    The GSOD bucket is public, so requests are unsigned. This is what lets the
    application run with no AWS account, no credentials file and no charges.
    """
    return boto3.client(
        "s3",
        region_name=config.S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def list_station_objects(year: int, client=None) -> pd.DataFrame:
    """
    List every station file for `year`, returning station_id, key and size.

    The bucket holds roughly 12,000 files per year and S3 pages results 1,000 at
    a time, so this walks the paginator to completion.
    """
    client = client or build_s3_client()
    prefix = f"{year}/"
    records = []

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".csv"):
                    continue
                records.append(
                    {
                        "station_id": Path(key).stem,
                        "key": key,
                        "size_bytes": item["Size"],
                    }
                )
    except (BotoCoreError, ClientError) as exc:
        raise DataAcquisitionError(
            f"Could not list s3://{config.S3_BUCKET}/{prefix} -- {exc}"
        ) from exc

    if not records:
        raise DataAcquisitionError(f"No station files found for year {year}")

    logger.info("Found %d station files for %d", len(records), year)
    return pd.DataFrame(records)


def select_global_sample(
    catalogue: pd.DataFrame,
    stations_per_block: int = config.STATIONS_PER_BLOCK,
    max_stations: int = config.MAX_STATIONS,
) -> pd.DataFrame:
    """
    Pick a globally distributed subset of stations from a listing.

    The first two digits of a GSOD station ID are the WMO block number, which is
    allocated by region. Sampling the largest few files from each block gives
    worldwide coverage while favouring stations with complete records.

    Deterministic: the same catalogue always yields the same sample.
    """
    if catalogue.empty:
        raise ValueError("Station catalogue is empty; nothing to sample")

    working = catalogue.copy()
    working["wmo_block"] = working["station_id"].str[:2]

    sample = (
        working.sort_values(["wmo_block", "size_bytes"], ascending=[True, False])
        .groupby("wmo_block", group_keys=False)
        .head(stations_per_block)
        .sort_values("size_bytes", ascending=False)
        .head(max_stations)
        .reset_index(drop=True)
    )

    logger.info(
        "Selected %d stations across %d WMO blocks",
        len(sample),
        sample["wmo_block"].nunique(),
    )
    return sample


def download_station(key: str, year: int, client=None, use_cache: bool = True) -> Path:
    """
    Download one station CSV to the local cache and return its path.

    Cached files are reused, so re-running the application costs nothing and
    works offline once the first run has completed.
    """
    client = client or build_s3_client()
    destination = config.DATA_DIR / "gsod" / str(year) / Path(key).name

    if use_cache and destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(config.S3_BUCKET, key, str(destination))
    except (BotoCoreError, ClientError) as exc:
        # Leave no truncated file behind to poison the cache on the next run.
        destination.unlink(missing_ok=True)
        raise DataAcquisitionError(f"Failed to download {key} -- {exc}") from exc

    return destination


def download_stations(
    keys: list[str], year: int, use_cache: bool = True
) -> list[Path]:
    """
    Download many station files concurrently.

    The files are small but numerous, so latency dominates; a small thread pool
    turns several minutes of sequential requests into a few seconds. Individual
    failures are logged and skipped rather than aborting the whole run -- one
    unavailable station should not cost us the other 250.
    """
    client = build_s3_client()
    paths: list[Path] = []

    def fetch(key: str) -> Path | None:
        try:
            return download_station(key, year, client=client, use_cache=use_cache)
        except DataAcquisitionError as exc:
            logger.warning("Skipping station: %s", exc)
            return None

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as pool:
        for result in pool.map(fetch, keys):
            if result is not None:
                paths.append(result)

    if not paths:
        raise DataAcquisitionError("Every station download failed")

    logger.info("Retrieved %d of %d station files", len(paths), len(keys))
    return paths


# --------------------------------------------------------------------------- #
# Parsing and cleaning
# --------------------------------------------------------------------------- #
def read_station_csv(path: Path | str) -> pd.DataFrame:
    """
    Read one GSOD station CSV into a DataFrame.

    FRSHTT is forced to string: it is a six-digit flag field where "011000"
    means rain and snow, and reading it as a number would destroy the leading
    zero and the positional meaning.
    """
    frame = pd.read_csv(
        path,
        dtype={"STATION": str, "FRSHTT": str},
        keep_default_na=False,
        na_values=[""],
    )

    required = {"STATION", "DATE", "LATITUDE", "LONGITUDE", "TEMP"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is not a GSOD file; missing columns: {sorted(missing)}")

    return frame


def clean_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a raw GSOD frame into clean daily observations in SI units.

    Four things happen here, and all four matter:

      1. Sentinel values become NaN. GSOD encodes "missing" as 9999.9, 999.9 or
         99.99 depending on the column. Skip this and a handful of gaps will
         drag a station's annual mean temperature into the hundreds.
      2. Unreported precipitation becomes NaN. GSOD writes 0.00 with flag "I"
         rather than a sentinel, so a station that never measures rain looks
         like a desert. See config.PRCP_NO_REPORT_FLAG.
      3. Units are converted to Celsius, millimetres and metres per second.
      4. The FRSHTT flag string is expanded into six boolean columns.
    """
    clean = frame.copy()

    # Anything numeric that will not parse becomes NaN rather than raising.
    for column in config.MISSING_SENTINELS:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    # 1. Sentinels -> NaN. Compared with >= rather than == because some files
    #    carry values marginally above the documented sentinel.
    for column, sentinel in config.MISSING_SENTINELS.items():
        if column in clean.columns:
            clean.loc[clean[column] >= sentinel, column] = np.nan

    # GSOD dates are always ISO. Stating the format is faster than letting
    # pandas infer it and, more importantly, means a malformed date becomes NaT
    # instead of being silently reinterpreted under some other convention.
    # 2. Flag "I" means the station reported nothing, but the value written is
    #    0.00 rather than the missing sentinel. Left alone, these fabricated
    #    zeros turn wet places into deserts.
    if {"PRCP", "PRCP_ATTRIBUTES"} <= set(clean.columns):
        not_reported = (
            clean["PRCP_ATTRIBUTES"].astype(str).str.strip()
            == config.PRCP_NO_REPORT_FLAG
        )
        clean.loc[not_reported, "PRCP"] = np.nan

    clean["date"] = pd.to_datetime(clean["DATE"], format="%Y-%m-%d", errors="coerce")
    clean = clean.dropna(subset=["date"])
    clean["month"] = clean["date"].dt.month

    # 3. Units.
    clean["temp_c"] = fahrenheit_to_celsius(clean["TEMP"])
    clean["dewpoint_c"] = fahrenheit_to_celsius(clean["DEWP"])
    clean["temp_max_c"] = fahrenheit_to_celsius(clean["MAX"])
    clean["temp_min_c"] = fahrenheit_to_celsius(clean["MIN"])
    clean["precip_mm"] = inches_to_mm(clean["PRCP"])
    clean["wind_ms"] = knots_to_ms(clean["WDSP"])

    # 4. FRSHTT -> booleans. Missing or malformed flags count as "not observed",
    #    which is the correct reading: GSOD only sets a flag when the phenomenon
    #    was actually reported.
    if "FRSHTT" in clean.columns:
        flags = clean["FRSHTT"].fillna("").astype(str).str.zfill(6)
    else:
        flags = pd.Series("000000", index=clean.index)
    for position, name in enumerate(config.WEATHER_FLAGS):
        clean[f"flag_{name}"] = flags.str[position].eq("1")

    return clean


def load_station(path: Path | str) -> pd.DataFrame:
    """Convenience wrapper: read a station file and clean it in one step."""
    return clean_observations(read_station_csv(path))
