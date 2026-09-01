"""
Central configuration for the GSOD climate clustering application.

Every tunable value lives here rather than being scattered through the code, so
a reviewer can see the whole experimental setup in one place and change it
without hunting through modules.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- #
# Data source: NOAA Global Surface Summary of the Day, Registry of Open Data
# on AWS. The bucket is public, so no AWS credentials are required -- requests
# are made anonymously (unsigned).
#
#   s3://noaa-gsod-pds/<year>/<station_id>.csv
#
# https://registry.opendata.aws/noaa-gsod/
# --------------------------------------------------------------------------- #
S3_BUCKET = "noaa-gsod-pds"
S3_REGION = "us-east-1"
YEAR = 2023

# --------------------------------------------------------------------------- #
# Station sampling
#
# The bucket holds roughly 12,000 station files per year and has no station
# index, so we cannot filter by country or latitude before downloading. What we
# can use is the station ID itself: the first two digits are the WMO block
# number, which is allocated geographically (03 = UK, 47 = Japan, 72 = USA,
# 94 = Australia, and so on). Taking a few stations from every block therefore
# gives global coverage cheaply.
#
# Within each block we take the LARGEST files, because file size is a good
# proxy for record completeness. This is deterministic, so runs are
# reproducible, but it does bias the sample toward well-instrumented sites
# (typically international airports). That limitation is stated in the summary.
# --------------------------------------------------------------------------- #
STATIONS_PER_BLOCK = 6
MAX_STATIONS = 600
DOWNLOAD_WORKERS = 8  # modest parallelism; the files are small

# --------------------------------------------------------------------------- #
# Quality thresholds. A station must clear all of these to enter the model,
# otherwise its climate fingerprint would be built from too little evidence.
# --------------------------------------------------------------------------- #
MIN_DAYS_WITH_TEMP = 300      # out of ~365
MIN_MONTHS_FOR_SEASONALITY = 10
MIN_DAYS_PER_MONTH = 15
MIN_DAYS_WITH_PRECIP = 200

# --------------------------------------------------------------------------- #
# GSOD missing-value sentinels.
#
# This is the single most important detail in the whole dataset. GSOD does not
# leave gaps blank; it fills them with magic numbers. Read the file naively and
# a station with a few missing readings gets an annual mean temperature of
# several hundred degrees. Every one of these must become NaN before any
# arithmetic happens.
#
# Source: NOAA GSOD documentation (readme.txt).
# --------------------------------------------------------------------------- #
MISSING_SENTINELS = {
    "TEMP": 9999.9,
    "DEWP": 9999.9,
    "MAX": 9999.9,
    "MIN": 9999.9,
    "SLP": 9999.9,
    "VISIB": 999.9,
    "WDSP": 999.9,
    "MXSPD": 999.9,
    "GUST": 999.9,
    "SNDP": 999.9,
    "PRCP": 99.99,
}

# STP (station pressure) is deliberately excluded. Older records store it in a
# truncated form (1009.6 written as "009.6"), and mixing the two conventions
# would silently corrupt the feature. Sea-level pressure carries the same
# information without the ambiguity.

# FRSHTT is a six-character flag string: Fog, Rain, Snow, Hail, Thunder,
# Tornado. It must be read as text, not as a number, or "011000" becomes 11000.
WEATHER_FLAGS = ["fog", "rain", "snow", "hail", "thunder", "tornado"]

# A day counts as wet at or above 1 mm, the standard meteorological threshold.
WET_DAY_THRESHOLD_MM = 1.0

# The subtlest trap in GSOD, and the one that cost the most debugging time here.
#
# When a station reports no precipitation data at all for a day, GSOD does NOT
# write the 99.99 missing sentinel. It writes 0.00 and sets PRCP_ATTRIBUTES to
# "I". The value is therefore indistinguishable from a genuinely dry day unless
# the flag is read.
#
# It is not a rare edge case: flag I covers roughly a fifth of all station-days
# in this sample, and about one station in sixteen carries it on every single
# day of the year. Untreated, it reported 0 mm of annual rainfall for a Scottish
# Highland pass that receives well over 1,000 mm, and for two stations in
# Brazil. Those fabricated zeros then pulled cluster centroids toward a
# non-existent "temperate desert".
#
# Flag H is kept: the station reported zero but observed some precipitation,
# which is a real trace amount rather than an absence of data.
PRCP_NO_REPORT_FLAG = "I"

# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
K_RANGE = range(2, 11)
N_INIT = 25  # restarts per k; K-Means is sensitive to initial centroids

# Features fed to K-Means. Geography is deliberately absent -- see FEATURE_NOTES.
FEATURE_COLUMNS = [
    "temp_mean_c",
    "temp_seasonality_c",
    "diurnal_range_c",
    "dewpoint_depression_c",
    "precip_log_mm",
    "wet_day_fraction",
    "snow_day_fraction",
    "wind_mean_ms",
    "fog_day_fraction",
    "thunder_day_fraction",
]

# Held back from the model and used only to validate the result afterwards.
# If clusters built purely from weather still line up with latitude and
# elevation, the model has recovered real geography on its own.
GEOGRAPHY_COLUMNS = ["latitude", "longitude", "elevation_m", "abs_latitude"]

# Carried into the cluster profiles for readability but not clustered on.
# precip_total_mm is here because millimetres mean something to a reader and
# log-millimetres do not; precip_log_mm is what the model actually sees.
REPORTING_COLUMNS = ["precip_total_mm"] + GEOGRAPHY_COLUMNS

FEATURE_NOTES = {
    "temp_mean_c": "Annual mean temperature",
    "temp_seasonality_c": "Warmest month minus coldest month (continentality)",
    "diurnal_range_c": "Mean daily max minus min (cloud cover and aridity proxy)",
    "dewpoint_depression_c": "Mean temperature minus dew point (dryness of the air)",
    "precip_log_mm": "Annual precipitation, log scale",
    "precip_total_mm": "Annual precipitation total",
    "wet_day_fraction": "Share of days with at least 1 mm of rain",
    "snow_day_fraction": "Share of days reporting snow or ice pellets",
    "wind_mean_ms": "Mean daily wind speed",
    "fog_day_fraction": "Share of days reporting fog",
    "thunder_day_fraction": "Share of days reporting thunder (convective activity)",
}
