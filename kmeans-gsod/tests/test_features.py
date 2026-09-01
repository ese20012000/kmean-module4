"""
Unit tests for feature engineering.

Two concerns:
  * the features compute the quantity they claim to (seasonality really is the
    warmest-minus-coldest month range, not something adjacent), and
  * stations with thin records are rejected rather than quietly contributing a
    fabricated fingerprint.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import config, data_loader, features
from tests import fixtures


def cleaned(**kwargs) -> pd.DataFrame:
    return data_loader.clean_observations(fixtures.make_raw_gsod(**kwargs))


class TestBuildStationFeatures(unittest.TestCase):
    def test_produces_every_configured_feature(self):
        row = features.build_station_features(cleaned(days=365))
        for column in config.FEATURE_COLUMNS:
            self.assertIn(column, row)
            self.assertTrue(np.isfinite(row[column]), f"{column} is not finite")

    def test_carries_identity_and_geography(self):
        row = features.build_station_features(
            cleaned(days=365, station_id="03772099999", latitude=51.48, longitude=-0.45)
        )
        self.assertEqual(row["station_id"], "03772099999")
        self.assertAlmostEqual(row["latitude"], 51.48, places=2)
        self.assertAlmostEqual(row["abs_latitude"], 51.48, places=2)

    def test_absolute_latitude_folds_the_southern_hemisphere(self):
        row = features.build_station_features(cleaned(days=365, latitude=-33.9))
        self.assertAlmostEqual(row["abs_latitude"], 33.9, places=2)

    def test_mean_temperature_is_correct(self):
        row = features.build_station_features(
            cleaned(days=365, temp_f=np.full(365, 68.0))
        )
        self.assertAlmostEqual(row["temp_mean_c"], 20.0, places=6)

    def test_seasonality_is_warmest_minus_coldest_month(self):
        """Exactly 0 C for six months then exactly 20 C: seasonality is 20."""
        observations = data_loader.clean_observations(
            fixtures.make_two_season_gsod(cold_c=0.0, warm_c=20.0)
        )
        row = features.build_station_features(observations)
        self.assertAlmostEqual(row["temp_seasonality_c"], 20.0, places=6)

    def test_constant_temperature_means_zero_seasonality(self):
        row = features.build_station_features(
            cleaned(days=365, temp_f=np.full(365, 80.0))
        )
        self.assertAlmostEqual(row["temp_seasonality_c"], 0.0, places=6)

    def test_seasonality_rises_with_the_annual_swing(self):
        mild = features.build_station_features(cleaned(days=365, amplitude_f=5.0))
        harsh = features.build_station_features(cleaned(days=365, amplitude_f=40.0))
        self.assertGreater(harsh["temp_seasonality_c"], mild["temp_seasonality_c"])

    def test_dewpoint_depression_measures_dryness(self):
        humid = features.build_station_features(cleaned(days=365, dewpoint_offset_f=2.0))
        dry = features.build_station_features(cleaned(days=365, dewpoint_offset_f=40.0))
        self.assertLess(humid["dewpoint_depression_c"], dry["dewpoint_depression_c"])

    def test_precipitation_is_annualised(self):
        """0.1 inch every day for a full year is 2.54 mm x 365."""
        row = features.build_station_features(cleaned(days=365, precip_in=0.1))
        self.assertAlmostEqual(row["precip_total_mm"], 2.54 * 365, places=3)

    def test_precipitation_log_matches_the_total(self):
        row = features.build_station_features(cleaned(days=365, precip_in=0.1))
        self.assertAlmostEqual(
            row["precip_log_mm"], float(np.log1p(row["precip_total_mm"])), places=9
        )

    def test_log_transform_compresses_the_rainfall_tail(self):
        """
        A monsoon station receives roughly 12 times the rainfall of a dry one but
        should sit only about 2.5 log units away, not 12 multiples away. This is
        what stops rainfall dominating the distance metric.
        """
        dry = features.build_station_features(cleaned(days=365, precip_in=0.01))
        wet = features.build_station_features(cleaned(days=365, precip_in=0.35))
        raw_ratio = wet["precip_total_mm"] / dry["precip_total_mm"]
        log_gap = wet["precip_log_mm"] - dry["precip_log_mm"]
        self.assertGreater(raw_ratio, 10)
        self.assertLess(log_gap, 5)

    def test_unreported_rain_does_not_fabricate_a_desert(self):
        """
        Same station, same actual rainfall, but half the year carries GSOD's "no
        report" flag. The annual total must be unchanged, not halved.
        """
        complete = features.build_station_features(cleaned(days=365, precip_in=0.1))
        flagged = features.build_station_features(
            cleaned(days=365, precip_in=0.1, prcp_not_reported_days=150)
        )
        self.assertAlmostEqual(
            complete["precip_total_mm"], flagged["precip_total_mm"], places=3
        )

    def test_annualisation_compensates_for_missing_days(self):
        """
        A station missing 30 precipitation days should not be reported as drier
        than an identical station with a complete record.
        """
        complete = features.build_station_features(cleaned(days=365, precip_in=0.1))
        partial = features.build_station_features(
            cleaned(days=365, precip_in=0.1, precip_sentinel_days=30)
        )
        self.assertAlmostEqual(
            complete["precip_total_mm"], partial["precip_total_mm"], places=3
        )

    def test_wet_day_fraction_uses_the_one_millimetre_threshold(self):
        # 0.01 in = 0.254 mm, below the threshold, so no day counts as wet.
        drizzle = features.build_station_features(cleaned(days=365, precip_in=0.01))
        self.assertAlmostEqual(drizzle["wet_day_fraction"], 0.0)

        # 0.1 in = 2.54 mm, above the threshold, so every day counts.
        wet = features.build_station_features(cleaned(days=365, precip_in=0.1))
        self.assertAlmostEqual(wet["wet_day_fraction"], 1.0)

    def test_flag_fractions_come_from_frshtt(self):
        row = features.build_station_features(cleaned(days=365, frshtt="101010"))
        self.assertAlmostEqual(row["fog_day_fraction"], 1.0)
        self.assertAlmostEqual(row["snow_day_fraction"], 1.0)
        self.assertAlmostEqual(row["thunder_day_fraction"], 1.0)


class TestQualityGates(unittest.TestCase):
    """Thin records must be refused, not guessed at."""

    def test_empty_station_is_rejected(self):
        with self.assertRaises(features.InsufficientDataError):
            features.build_station_features(cleaned(days=0))

    def test_too_few_temperature_days_is_rejected(self):
        with self.assertRaises(features.InsufficientDataError):
            features.build_station_features(cleaned(days=120))

    def test_too_few_precipitation_days_is_rejected(self):
        with self.assertRaises(features.InsufficientDataError) as context:
            features.build_station_features(
                cleaned(days=365, precip_sentinel_days=200)
            )
        self.assertIn("precipitation", str(context.exception))

    def test_partial_year_is_rejected(self):
        """A station that stopped reporting in August cannot describe a climate."""
        observations = cleaned(days=365)
        observations = observations[observations["month"] <= 8]
        with self.assertRaises(features.InsufficientDataError):
            features.build_station_features(observations)

    def test_sparse_months_are_ignored_when_ranking(self):
        """
        A month represented by only two freak-cold days must not be allowed to
        become the coldest month and inflate seasonality.
        """
        observations = data_loader.clean_observations(
            fixtures.make_two_season_gsod(cold_c=10.0, warm_c=20.0)
        )
        january = observations["month"] == 1
        observations.loc[january, "temp_c"] = -50.0
        # Leave January with only two days, below MIN_DAYS_PER_MONTH.
        keep = ~january | (january & (observations.groupby("month").cumcount() < 2))
        row = features.build_station_features(observations[keep])
        self.assertAlmostEqual(row["temp_seasonality_c"], 10.0, places=6)


class TestBuildFeatureTable(unittest.TestCase):
    def test_good_stations_are_kept_and_bad_ones_dropped(self):
        frames = {
            "good_a": cleaned(days=365, station_id="72503014732"),
            "good_b": cleaned(days=365, station_id="03772099999", mean_temp_f=45.0),
            "too_short": cleaned(days=60, station_id="94767099999"),
        }
        table = features.build_feature_table(frames)
        self.assertEqual(len(table), 2)
        self.assertIn("72503014732", table.index)

    def test_indexed_by_station_id(self):
        table = features.build_feature_table({"a": cleaned(days=365)})
        self.assertEqual(table.index.name, "station_id")

    def test_no_missing_values_reach_the_model(self):
        table = features.build_feature_table(
            {f"s{i}": cleaned(days=365, station_id=f"7250301473{i}") for i in range(3)}
        )
        self.assertFalse(table[config.FEATURE_COLUMNS].isna().any().any())

    def test_all_stations_rejected_raises(self):
        with self.assertRaises(features.InsufficientDataError):
            features.build_feature_table({"bad": cleaned(days=30)})


class TestDescribeCluster(unittest.TestCase):
    """The heuristic labels are for readability, but they should not be wrong."""

    @staticmethod
    def profile(**overrides) -> pd.Series:
        base = {
            "temp_mean_c": 12.0,
            "temp_seasonality_c": 15.0,
            "diurnal_range_c": 9.0,
            "dewpoint_depression_c": 5.0,
            "precip_total_mm": 800.0,
            "wet_day_fraction": 0.3,
            "snow_day_fraction": 0.02,
            "wind_mean_ms": 4.0,
            "fog_day_fraction": 0.1,
            "thunder_day_fraction": 0.05,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_subfreezing_and_mildly_seasonal_is_tundra(self):
        self.assertEqual(features.describe_cluster(self.profile(temp_mean_c=-9.0)),
                         "Polar / tundra")

    def test_subfreezing_and_wildly_seasonal_is_subarctic_continental(self):
        """
        Regression test. Antarctic coastal stations and Fairbanks both average
        below freezing, but one swings 20 C across the year and the other nearly
        40. A single "polar" label collapsed two unrelated climates into one.
        """
        tundra = features.describe_cluster(
            self.profile(temp_mean_c=-1.0, temp_seasonality_c=20.4, precip_total_mm=621.0)
        )
        subarctic = features.describe_cluster(
            self.profile(temp_mean_c=-0.5, temp_seasonality_c=39.7, precip_total_mm=570.0)
        )
        self.assertEqual(tundra, "Polar / tundra")
        self.assertEqual(subarctic, "Subarctic continental")
        self.assertNotEqual(tundra, subarctic)

    def test_the_five_observed_clusters_all_get_distinct_labels(self):
        """The labels have to actually discriminate on the real cluster profiles."""
        observed = [
            dict(temp_mean_c=-1.0, temp_seasonality_c=20.4, precip_total_mm=621.0,
                 dewpoint_depression_c=2.6, snow_day_fraction=0.34),
            dict(temp_mean_c=-0.5, temp_seasonality_c=39.7, precip_total_mm=570.0,
                 dewpoint_depression_c=5.3, snow_day_fraction=0.26),
            dict(temp_mean_c=12.4, temp_seasonality_c=26.9, precip_total_mm=211.0,
                 dewpoint_depression_c=14.0, snow_day_fraction=0.03),
            dict(temp_mean_c=16.5, temp_seasonality_c=15.9, precip_total_mm=1011.0,
                 dewpoint_depression_c=5.6, snow_day_fraction=0.03),
            dict(temp_mean_c=27.1, temp_seasonality_c=3.6, precip_total_mm=2802.0,
                 dewpoint_depression_c=3.9, snow_day_fraction=0.001),
        ]
        labels = [features.describe_cluster(self.profile(**o)) for o in observed]
        self.assertEqual(len(set(labels)), 5, f"labels collided: {labels}")

    def test_snowy_and_seasonal_is_cold_continental(self):
        label = features.describe_cluster(
            self.profile(temp_mean_c=4.0, snow_day_fraction=0.2, temp_seasonality_c=28.0)
        )
        self.assertEqual(label, "Cold continental")

    def test_dry_is_arid(self):
        label = features.describe_cluster(
            self.profile(precip_total_mm=180.0, dewpoint_depression_c=18.0)
        )
        self.assertEqual(label, "Arid / semi-arid")

    def test_seasonal_desert_is_arid_not_continental(self):
        """
        The regression test for a real mislabelling. The New Mexico and Iranian
        plateau cluster swings 26 C across the year on 112 mm of rain. Testing
        seasonality before dryness called it "temperate continental".
        """
        label = features.describe_cluster(
            self.profile(
                temp_mean_c=16.2,
                temp_seasonality_c=26.0,
                precip_total_mm=112.0,
                dewpoint_depression_c=16.3,
            )
        )
        self.assertEqual(label, "Arid / semi-arid")

    def test_dry_but_humid_air_is_mediterranean(self):
        """Coastal deserts: little rain, but the air is not dry. Santiago, Tacna."""
        label = features.describe_cluster(
            self.profile(
                temp_mean_c=21.8,
                temp_seasonality_c=12.5,
                precip_total_mm=150.0,
                dewpoint_depression_c=6.4,
            )
        )
        self.assertEqual(label, "Mediterranean / coastal dry")

    def test_hot_and_wet_is_tropical(self):
        label = features.describe_cluster(
            self.profile(temp_mean_c=27.0, temp_seasonality_c=4.0, precip_total_mm=2000.0)
        )
        self.assertEqual(label, "Tropical humid")

    def test_mild_default_is_maritime(self):
        self.assertEqual(features.describe_cluster(self.profile()), "Maritime temperate")

    def test_every_label_is_a_non_empty_string(self):
        for temperature in [-20, -1, 5, 15, 25, 35]:
            for seasonality in [2, 15, 28, 45]:
                label = features.describe_cluster(
                    self.profile(temp_mean_c=temperature, temp_seasonality_c=seasonality)
                )
                self.assertIsInstance(label, str)
                self.assertTrue(label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
