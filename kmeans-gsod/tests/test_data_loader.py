"""
Unit tests for data acquisition and cleaning.

The bug these tests exist to prevent: GSOD encodes missing readings as 9999.9,
999.9 or 99.99 rather than leaving them blank. Miss that and every downstream
number is wrong while looking perfectly plausible. TestCleanObservations is the
heart of this file.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src import config, data_loader
from tests import fixtures


class TestUnitConversions(unittest.TestCase):
    """Conversions are trivial but wrong units would invalidate every finding."""

    def test_freezing_point(self):
        self.assertAlmostEqual(data_loader.fahrenheit_to_celsius(32.0), 0.0)

    def test_boiling_point(self):
        self.assertAlmostEqual(data_loader.fahrenheit_to_celsius(212.0), 100.0)

    def test_negative_forty_is_the_same_in_both_scales(self):
        self.assertAlmostEqual(data_loader.fahrenheit_to_celsius(-40.0), -40.0)

    def test_inches_to_mm(self):
        self.assertAlmostEqual(data_loader.inches_to_mm(1.0), 25.4)

    def test_knots_to_ms(self):
        self.assertAlmostEqual(data_loader.knots_to_ms(1.0), 0.514444, places=6)

    def test_conversions_preserve_nan(self):
        result = data_loader.fahrenheit_to_celsius(pd.Series([32.0, np.nan]))
        self.assertAlmostEqual(result.iloc[0], 0.0)
        self.assertTrue(np.isnan(result.iloc[1]))


class TestSelectGlobalSample(unittest.TestCase):
    """Station selection must be geographically spread and reproducible."""

    def setUp(self):
        self.catalogue = fixtures.make_station_catalogue()

    def test_covers_every_wmo_block(self):
        sample = data_loader.select_global_sample(self.catalogue, stations_per_block=2)
        blocks = sample["station_id"].str[:2].unique()
        self.assertCountEqual(blocks, ["03", "47", "72", "94"])

    def test_respects_stations_per_block(self):
        sample = data_loader.select_global_sample(self.catalogue, stations_per_block=2)
        counts = sample["station_id"].str[:2].value_counts()
        self.assertTrue((counts == 2).all(), f"uneven block counts: {counts.to_dict()}")

    def test_prefers_largest_files_as_a_completeness_proxy(self):
        sample = data_loader.select_global_sample(self.catalogue, stations_per_block=2)
        uk = sample[sample["station_id"].str.startswith("03")]
        # Fixture sizes run 10000..14000, so the two largest are 13000 and 14000.
        self.assertCountEqual(uk["size_bytes"].tolist(), [14_000, 13_000])

    def test_honours_max_stations(self):
        sample = data_loader.select_global_sample(
            self.catalogue, stations_per_block=3, max_stations=5
        )
        self.assertEqual(len(sample), 5)

    def test_is_deterministic(self):
        first = data_loader.select_global_sample(self.catalogue, stations_per_block=2)
        second = data_loader.select_global_sample(self.catalogue, stations_per_block=2)
        pd.testing.assert_frame_equal(first, second)

    def test_empty_catalogue_is_rejected(self):
        with self.assertRaises(ValueError):
            data_loader.select_global_sample(pd.DataFrame())


class TestListStationObjects(unittest.TestCase):
    """S3 listing is mocked: we are testing our pagination, not AWS."""

    def _client_returning(self, pages):
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = pages
        return client

    def test_walks_every_page(self):
        pages = [
            {"Contents": [{"Key": "2023/03001099999.csv", "Size": 1000}]},
            {"Contents": [{"Key": "2023/72503014732.csv", "Size": 2000}]},
        ]
        catalogue = data_loader.list_station_objects(2023, client=self._client_returning(pages))
        self.assertEqual(len(catalogue), 2)
        self.assertIn("72503014732", catalogue["station_id"].tolist())

    def test_ignores_non_csv_keys(self):
        pages = [
            {
                "Contents": [
                    {"Key": "2023/03001099999.csv", "Size": 1000},
                    {"Key": "2023/index.html", "Size": 50},
                    {"Key": "2023/README.txt", "Size": 20},
                ]
            }
        ]
        catalogue = data_loader.list_station_objects(2023, client=self._client_returning(pages))
        self.assertEqual(len(catalogue), 1)

    def test_empty_year_raises(self):
        with self.assertRaises(data_loader.DataAcquisitionError):
            data_loader.list_station_objects(1850, client=self._client_returning([{}]))


class TestReadStationCsv(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "72503014732.csv"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reads_a_well_formed_file(self):
        fixtures.make_raw_gsod(days=10).to_csv(self.path, index=False)
        frame = data_loader.read_station_csv(self.path)
        self.assertEqual(len(frame), 10)

    def test_frshtt_keeps_its_leading_zeros(self):
        """Read as a number, "011000" becomes 11000 and every flag shifts."""
        fixtures.make_raw_gsod(days=5, frshtt="011000").to_csv(self.path, index=False)
        frame = data_loader.read_station_csv(self.path)
        self.assertEqual(frame["FRSHTT"].iloc[0], "011000")

    def test_non_gsod_file_is_rejected_clearly(self):
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(self.path, index=False)
        with self.assertRaises(ValueError) as context:
            data_loader.read_station_csv(self.path)
        self.assertIn("missing columns", str(context.exception))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            data_loader.read_station_csv(Path(self.tempdir.name) / "absent.csv")


class TestCleanObservations(unittest.TestCase):
    """The most important tests in the suite."""

    def test_temperature_sentinels_become_nan(self):
        raw = fixtures.make_raw_gsod(days=100, temp_sentinel_days=10)
        clean = data_loader.clean_observations(raw)
        self.assertEqual(clean["temp_c"].isna().sum(), 10)
        self.assertEqual(clean["temp_c"].notna().sum(), 90)

    def test_sentinels_do_not_poison_the_mean(self):
        """
        The regression test for the whole project. A station at a constant 50 F
        (10 C) with ten missing days must average 10 C, not several hundred.
        """
        raw = fixtures.make_raw_gsod(
            days=100, temp_f=np.full(100, 50.0), temp_sentinel_days=10
        )
        clean = data_loader.clean_observations(raw)
        self.assertAlmostEqual(clean["temp_c"].mean(), 10.0, places=6)

    def test_precipitation_sentinel_becomes_nan(self):
        raw = fixtures.make_raw_gsod(days=50, precip_sentinel_days=5)
        clean = data_loader.clean_observations(raw)
        self.assertEqual(clean["precip_mm"].isna().sum(), 5)

    def test_unreported_precipitation_is_missing_not_zero(self):
        """
        The second regression test for the project. GSOD writes 0.00 with flag "I"
        when a station reported no precipitation data at all. Treating those as
        dry days reported 0 mm of annual rainfall for a Scottish Highland pass.
        """
        raw = fixtures.make_raw_gsod(days=100, prcp_not_reported_days=40)
        clean = data_loader.clean_observations(raw)
        self.assertEqual(clean["precip_mm"].isna().sum(), 40)
        self.assertEqual(clean["precip_mm"].notna().sum(), 60)

    def test_a_station_that_never_reports_rain_has_no_precipitation_data(self):
        raw = fixtures.make_raw_gsod(days=365, prcp_not_reported_days=365)
        clean = data_loader.clean_observations(raw)
        self.assertTrue(clean["precip_mm"].isna().all())

    def test_genuine_zero_precipitation_is_kept(self):
        """A real dry day, flag G, is data and must survive."""
        raw = fixtures.make_raw_gsod(days=30, precip_in=0.0, prcp_flag="G")
        clean = data_loader.clean_observations(raw)
        self.assertEqual(clean["precip_mm"].notna().sum(), 30)
        self.assertEqual(clean["precip_mm"].sum(), 0.0)

    def test_trace_amount_flag_h_is_kept(self):
        """Flag H means the station observed precipitation but logged zero. Real."""
        raw = fixtures.make_raw_gsod(days=30, precip_in=0.0, prcp_flag="H")
        clean = data_loader.clean_observations(raw)
        self.assertEqual(clean["precip_mm"].notna().sum(), 30)

    def test_absent_prcp_attributes_column_is_tolerated(self):
        raw = fixtures.make_raw_gsod(days=20).drop(columns=["PRCP_ATTRIBUTES"])
        clean = data_loader.clean_observations(raw)
        self.assertEqual(clean["precip_mm"].notna().sum(), 20)

    def test_routinely_missing_columns_are_all_nan(self):
        """GUST and SNDP are 999.9 in the fixture, as they usually are for real."""
        clean = data_loader.clean_observations(fixtures.make_raw_gsod(days=20))
        self.assertTrue(clean["GUST"].isna().all())
        self.assertTrue(clean["SNDP"].isna().all())

    def test_units_are_converted(self):
        raw = fixtures.make_raw_gsod(days=5, temp_f=np.full(5, 32.0),
                                     precip_in=1.0, wind_knots=1.0)
        clean = data_loader.clean_observations(raw)
        self.assertAlmostEqual(clean["temp_c"].iloc[0], 0.0)
        self.assertAlmostEqual(clean["precip_mm"].iloc[0], 25.4)
        self.assertAlmostEqual(clean["wind_ms"].iloc[0], 0.514444, places=6)

    def test_diurnal_range_survives_conversion(self):
        """An 18 F spread is 10 C; a common slip is to convert the range as if
        it were an absolute temperature."""
        raw = fixtures.make_raw_gsod(days=5, diurnal_f=18.0)
        clean = data_loader.clean_observations(raw)
        spread = (clean["temp_max_c"] - clean["temp_min_c"]).mean()
        self.assertAlmostEqual(spread, 10.0, places=6)

    def test_frshtt_expands_to_the_right_flags(self):
        # "010100" = rain and hail, no fog, snow, thunder or tornado.
        clean = data_loader.clean_observations(
            fixtures.make_raw_gsod(days=5, frshtt="010100")
        )
        self.assertFalse(clean["flag_fog"].any())
        self.assertTrue(clean["flag_rain"].all())
        self.assertFalse(clean["flag_snow"].any())
        self.assertTrue(clean["flag_hail"].all())
        self.assertFalse(clean["flag_thunder"].any())
        self.assertFalse(clean["flag_tornado"].any())

    def test_all_six_flags_are_created(self):
        clean = data_loader.clean_observations(fixtures.make_raw_gsod(days=3))
        for name in config.WEATHER_FLAGS:
            self.assertIn(f"flag_{name}", clean.columns)

    def test_absent_frshtt_column_is_tolerated(self):
        raw = fixtures.make_raw_gsod(days=5).drop(columns=["FRSHTT"])
        clean = data_loader.clean_observations(raw)
        self.assertFalse(clean["flag_snow"].any())

    def test_unparseable_dates_are_dropped(self):
        raw = fixtures.make_raw_gsod(days=10)
        raw.loc[0, "DATE"] = "not-a-date"
        clean = data_loader.clean_observations(raw)
        self.assertEqual(len(clean), 9)

    def test_month_column_is_added(self):
        clean = data_loader.clean_observations(fixtures.make_raw_gsod(days=365))
        self.assertEqual(sorted(clean["month"].unique().tolist()), list(range(1, 13)))

    def test_input_frame_is_not_mutated(self):
        raw = fixtures.make_raw_gsod(days=10)
        before = raw.copy()
        data_loader.clean_observations(raw)
        pd.testing.assert_frame_equal(raw, before)


@unittest.skipUnless(
    os.environ.get("RUN_NETWORK_TESTS") == "1",
    "set RUN_NETWORK_TESTS=1 to test live S3 access",
)
class TestLiveS3Access(unittest.TestCase):
    """
    Optional smoke test against the real bucket.

    Excluded by default so the suite stays fast and offline, but useful for
    confirming the bucket is still public and the layout has not changed.
    """

    def test_can_list_and_download_a_station(self):
        catalogue = data_loader.list_station_objects(config.YEAR)
        self.assertGreater(len(catalogue), 1000)

        sample = data_loader.select_global_sample(catalogue, stations_per_block=1,
                                                  max_stations=1)
        path = data_loader.download_station(sample["key"].iloc[0], config.YEAR)
        self.assertTrue(path.exists())
        self.assertGreater(len(data_loader.load_station(path)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
