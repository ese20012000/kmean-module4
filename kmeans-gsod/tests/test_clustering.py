"""
Unit tests for scaling, k selection, K-Means fitting and validation.

The fixture is a feature table with three deliberately well-separated synthetic
climate groups, so the tests can assert that the pipeline recovers a structure
whose correct answer is known in advance.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src import clustering, config
from tests import fixtures


class TestScaleFeatures(unittest.TestCase):
    def setUp(self):
        self.table = fixtures.make_feature_table()

    def test_shape_matches_the_feature_configuration(self):
        matrix, _ = clustering.scale_features(self.table)
        self.assertEqual(matrix.shape, (len(self.table), len(config.FEATURE_COLUMNS)))

    def test_output_is_standardized(self):
        matrix, _ = clustering.scale_features(self.table)
        np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(matrix.std(axis=0), 1.0, atol=1e-9)

    def test_scaling_matters_because_the_raw_units_are_incomparable(self):
        """
        Temperature spans tens of degrees while the day-fraction features sit
        below 1. Unscaled, the wide-ranging features would decide every cluster
        on their units alone.
        """
        raw_spread = self.table[config.FEATURE_COLUMNS].std()
        self.assertGreater(raw_spread.max() / raw_spread.min(), 20)

    def test_missing_column_is_reported_clearly(self):
        broken = self.table.drop(columns=["temp_mean_c"])
        with self.assertRaises(KeyError) as context:
            clustering.scale_features(broken)
        self.assertIn("temp_mean_c", str(context.exception))

    def test_nan_is_refused_rather_than_propagated(self):
        broken = self.table.copy()
        broken.loc[broken.index[0], "temp_mean_c"] = np.nan
        with self.assertRaises(ValueError):
            clustering.scale_features(broken)

    def test_infinity_is_refused(self):
        broken = self.table.copy()
        broken.loc[broken.index[0], "precip_log_mm"] = np.inf
        with self.assertRaises(ValueError):
            clustering.scale_features(broken)


class TestEvaluateKRange(unittest.TestCase):
    def setUp(self):
        self.matrix, self.scaler = clustering.scale_features(fixtures.make_feature_table())

    def test_covers_the_configured_range(self):
        diagnostics = clustering.evaluate_k_range(self.matrix)
        self.assertEqual(diagnostics.index.tolist(), list(config.K_RANGE))

    def test_reports_all_three_metrics(self):
        diagnostics = clustering.evaluate_k_range(self.matrix)
        for metric in ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]:
            self.assertIn(metric, diagnostics.columns)

    def test_silhouette_stays_within_bounds(self):
        diagnostics = clustering.evaluate_k_range(self.matrix)
        self.assertTrue(diagnostics["silhouette"].between(-1, 1).all())

    def test_inertia_falls_monotonically_with_k(self):
        """More clusters can only reduce within-cluster spread."""
        diagnostics = clustering.evaluate_k_range(self.matrix)
        self.assertTrue((diagnostics["inertia"].diff().dropna() < 0).all())

    def test_too_few_samples_for_the_requested_k_raises(self):
        with self.assertRaises(ValueError):
            clustering.evaluate_k_range(self.matrix[:5])


class TestChooseK(unittest.TestCase):
    @staticmethod
    def diagnostics_frame(silhouette, calinski, davies_bouldin, ks=(2, 3, 4)):
        return pd.DataFrame(
            {
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies_bouldin,
            },
            index=pd.Index(ks, name="k"),
        )

    # The genuine k=2..10 diagnostics from a 243-station run, used so the tests
    # pin the behaviour of the real decision rather than a contrived one.
    OBSERVED_INERTIA = [1912.70, 1559.43, 1359.39, 1198.56, 1075.88,
                        976.21, 887.96, 823.27, 765.24]
    OBSERVED_SILHOUETTE = [0.2274, 0.2178, 0.2086, 0.2198, 0.2339,
                           0.2172, 0.2255, 0.2249, 0.2345]

    @classmethod
    def observed_diagnostics(cls) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "inertia": cls.OBSERVED_INERTIA,
                "silhouette": cls.OBSERVED_SILHOUETTE,
                "calinski_harabasz": [65.18, 66.99, 62.74, 61.13, 59.66,
                                      58.58, 58.30, 57.09, 56.32],
                "davies_bouldin": [1.630, 1.482, 1.448, 1.413, 1.266,
                                   1.309, 1.241, 1.197, 1.159],
            },
            index=pd.Index(range(2, 11), name="k"),
        )

    def test_delegates_to_the_elbow(self):
        diagnostics = self.observed_diagnostics()
        self.assertEqual(
            clustering.choose_k(diagnostics), clustering.find_elbow(diagnostics)
        )

    def test_a_flat_silhouette_does_not_drag_k_to_the_range_edge(self):
        """
        The failure this rule exists to prevent. On the real data silhouette peaks
        at k=10, beating k=6 by 0.0007 -- noise. The elbow must ignore that and
        return a parsimonious k well inside the range.
        """
        diagnostics = self.observed_diagnostics()
        self.assertEqual(clustering.choose_k_by_silhouette(diagnostics), 10)
        self.assertLess(clustering.choose_k(diagnostics), 7)

    def test_silhouette_only_selection_still_available(self):
        diagnostics = self.diagnostics_frame(
            silhouette=[0.30, 0.71, 0.44],
            calinski=[90.0, 40.0, 55.0],
            davies_bouldin=[0.60, 1.80, 1.20],
        )
        self.assertEqual(clustering.choose_k_by_silhouette(diagnostics), 3)

    def test_recovers_the_true_group_count_on_known_data(self):
        """The fixture holds three separated groups, so k should come out as 3."""
        matrix, _ = clustering.scale_features(fixtures.make_feature_table())
        diagnostics = clustering.evaluate_k_range(matrix)
        self.assertEqual(clustering.choose_k(diagnostics), 3)

    def test_empty_diagnostics_raises(self):
        with self.assertRaises(ValueError):
            clustering.choose_k(pd.DataFrame())

    def test_empty_diagnostics_raises_for_silhouette_selection(self):
        with self.assertRaises(ValueError):
            clustering.choose_k_by_silhouette(pd.DataFrame())


class TestFindElbow(unittest.TestCase):
    @staticmethod
    def frame(inertia, ks=None) -> pd.DataFrame:
        ks = ks or range(2, 2 + len(inertia))
        return pd.DataFrame({"inertia": inertia}, index=pd.Index(ks, name="k"))

    def test_finds_a_sharp_elbow(self):
        """Steep to k=4, then flat. The elbow is 4."""
        elbow = clustering.find_elbow(self.frame([100.0, 60.0, 25.0, 22.0, 21.0, 20.5]))
        self.assertEqual(elbow, 4)

    def test_finds_the_elbow_in_the_real_curve(self):
        diagnostics = TestChooseK.observed_diagnostics()
        self.assertEqual(clustering.find_elbow(diagnostics), 5)

    def test_a_straight_line_has_no_elbow(self):
        """Constant returns per cluster means no k is preferred; take the smallest."""
        self.assertEqual(clustering.find_elbow(self.frame([100.0, 80.0, 60.0, 40.0, 20.0])), 2)

    def test_flat_curve_falls_back_to_the_smallest_k(self):
        self.assertEqual(clustering.find_elbow(self.frame([50.0, 50.0, 50.0])), 2)

    def test_is_independent_of_the_starting_k(self):
        shape = [100.0, 60.0, 25.0, 22.0, 21.0, 20.5]
        self.assertEqual(clustering.find_elbow(self.frame(shape, ks=range(5, 11))), 7)

    def test_too_few_points_returns_the_smallest_k(self):
        self.assertEqual(clustering.find_elbow(self.frame([100.0, 50.0])), 2)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            clustering.find_elbow(pd.DataFrame())


class TestRankDiagnostics(unittest.TestCase):
    def setUp(self):
        self.diagnostics = pd.DataFrame(
            {
                "silhouette": [0.20, 0.40, 0.30],
                "calinski_harabasz": [50.0, 70.0, 60.0],
                "davies_bouldin": [1.50, 0.80, 1.10],
            },
            index=pd.Index([2, 3, 4], name="k"),
        )

    def test_best_score_ranks_first(self):
        ranks = clustering.rank_diagnostics(self.diagnostics)
        self.assertEqual(ranks.loc[3, "silhouette_rank"], 1.0)
        self.assertEqual(ranks.loc[3, "calinski_rank"], 1.0)

    def test_davies_bouldin_is_ranked_ascending_because_lower_is_better(self):
        ranks = clustering.rank_diagnostics(self.diagnostics)
        self.assertEqual(ranks.loc[3, "davies_bouldin_rank"], 1.0)
        self.assertEqual(ranks.loc[2, "davies_bouldin_rank"], 3.0)

    def test_mean_rank_is_the_average_of_the_three(self):
        ranks = clustering.rank_diagnostics(self.diagnostics)
        expected = ranks.loc[4, ["silhouette_rank", "calinski_rank",
                                 "davies_bouldin_rank"]].mean()
        self.assertAlmostEqual(ranks.loc[4, "mean_rank"], expected)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            clustering.rank_diagnostics(pd.DataFrame())


class TestFitKMeans(unittest.TestCase):
    def setUp(self):
        self.table = fixtures.make_feature_table()
        self.matrix, self.scaler = clustering.scale_features(self.table)

    def test_labels_one_per_station(self):
        _, labels = clustering.fit_kmeans(self.matrix, 3)
        self.assertEqual(len(labels), len(self.table))

    def test_produces_exactly_k_clusters(self):
        _, labels = clustering.fit_kmeans(self.matrix, 4)
        self.assertEqual(len(np.unique(labels)), 4)

    def test_is_reproducible_with_a_fixed_seed(self):
        """
        K-Means converges to a local optimum set by its starting centroids. Without
        a fixed seed the reported clusters would change between runs and the
        write-up could not be trusted.
        """
        _, first = clustering.fit_kmeans(self.matrix, 3, seed=42)
        _, second = clustering.fit_kmeans(self.matrix, 3, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_recovers_the_synthetic_groups(self):
        _, labels = clustering.fit_kmeans(self.matrix, 3)
        agreement = adjusted_rand_score(self.table["true_group"], labels)
        self.assertGreater(agreement, 0.95, f"ARI only {agreement:.3f}")


class TestProfileClusters(unittest.TestCase):
    def setUp(self):
        self.table = fixtures.make_feature_table()
        matrix, _ = clustering.scale_features(self.table)
        _, self.labels = clustering.fit_kmeans(matrix, 3)
        self.profile = clustering.profile_clusters(self.table, self.labels)

    def test_one_row_per_cluster(self):
        self.assertEqual(len(self.profile), 3)

    def test_station_counts_add_up(self):
        self.assertEqual(self.profile["n_stations"].sum(), len(self.table))

    def test_includes_withheld_geography_for_validation(self):
        for column in config.GEOGRAPHY_COLUMNS:
            self.assertIn(column, self.profile.columns)

    def test_sorted_by_temperature_for_readable_reporting(self):
        temperatures = self.profile["temp_mean_c"].tolist()
        self.assertEqual(temperatures, sorted(temperatures))

    def test_label_count_mismatch_is_caught(self):
        with self.assertRaises(ValueError):
            clustering.profile_clusters(self.table, self.labels[:-1])


class TestVarianceExplained(unittest.TestCase):
    def test_perfect_separation_gives_one(self):
        table = pd.DataFrame({"abs_latitude": [10.0, 10.0, 60.0, 60.0]})
        result = clustering.variance_explained(table, np.array([0, 0, 1, 1]), "abs_latitude")
        self.assertAlmostEqual(result, 1.0)

    def test_no_relationship_gives_about_zero(self):
        table = pd.DataFrame({"abs_latitude": [10.0, 60.0, 10.0, 60.0]})
        result = clustering.variance_explained(table, np.array([0, 0, 1, 1]), "abs_latitude")
        self.assertAlmostEqual(result, 0.0)

    def test_constant_column_returns_nan(self):
        table = pd.DataFrame({"elevation_m": [5.0, 5.0, 5.0, 5.0]})
        result = clustering.variance_explained(table, np.array([0, 0, 1, 1]), "elevation_m")
        self.assertTrue(np.isnan(result))

    def test_nan_values_are_ignored(self):
        table = pd.DataFrame({"elevation_m": [10.0, np.nan, 60.0, 60.0]})
        result = clustering.variance_explained(table, np.array([0, 0, 1, 1]), "elevation_m")
        self.assertTrue(np.isfinite(result))

    def test_clusters_track_latitude_they_never_saw(self):
        """
        The headline validation. Coordinates are not features, so a high score
        here means the clustering inferred geography from weather alone.
        """
        table = fixtures.make_feature_table()
        matrix, _ = clustering.scale_features(table)
        _, labels = clustering.fit_kmeans(matrix, 3)
        share = clustering.variance_explained(table, labels, "abs_latitude")
        self.assertGreater(share, 0.80, f"only {share:.1%} of latitude variance explained")


class TestRepresentativeStations(unittest.TestCase):
    def setUp(self):
        self.table = fixtures.make_feature_table()
        self.matrix, _ = clustering.scale_features(self.table)
        self.model, self.labels = clustering.fit_kmeans(self.matrix, 3)
        self.representatives = clustering.representative_stations(
            self.table, self.matrix, self.model, self.labels
        )

    def test_one_entry_per_cluster(self):
        self.assertEqual(sorted(self.representatives), [0, 1, 2])

    def test_returns_the_requested_number(self):
        picked = clustering.representative_stations(
            self.table, self.matrix, self.model, self.labels, count=2
        )
        for names in picked.values():
            self.assertEqual(len(names), 2)

    def test_representatives_belong_to_their_own_cluster(self):
        """
        The bug this replaces: listing the first members in table order presented
        a Norwegian station as typical of the tropical cluster.
        """
        for cluster, names in self.representatives.items():
            members = set(self.table["name"].iloc[self.labels == cluster])
            for name in names:
                self.assertIn(name, members)

    def test_the_nearest_station_beats_an_arbitrary_member(self):
        for cluster, names in self.representatives.items():
            members = np.flatnonzero(self.labels == cluster)
            distances = np.linalg.norm(
                self.matrix[members] - self.model.cluster_centers_[cluster], axis=1
            )
            chosen = self.table["name"].iloc[members].tolist().index(names[0])
            self.assertAlmostEqual(distances[chosen], distances.min())

    def test_an_empty_cluster_yields_an_empty_list(self):
        labels = np.zeros(len(self.table), dtype=int)  # everything in cluster 0
        picked = clustering.representative_stations(
            self.table, self.matrix, self.model, labels
        )
        self.assertEqual(picked[1], [])
        self.assertEqual(picked[2], [])


class TestBoundaryFragility(unittest.TestCase):
    def test_well_separated_clusters_are_not_fragile(self):
        table = fixtures.make_feature_table()
        matrix, _ = clustering.scale_features(table)
        model, labels = clustering.fit_kmeans(matrix, 3)
        self.assertLess(clustering.boundary_fragility(matrix, model, labels), 0.05)

    def test_a_straddling_point_is_flagged(self):
        """
        Two tight groups with one station sitting exactly between them. The groups
        alone are unambiguous; adding the straddler introduces a coin flip, and
        the measure has to notice.

        Twenty points per group rather than one, so that the centroids barely move
        toward the straddler -- with a handful of points the centroid follows it
        and the ambiguity disappears.
        """
        groups = np.array([[-5.0]] * 20 + [[5.0]] * 20)
        model, labels = clustering.fit_kmeans(groups, 2)
        self.assertEqual(clustering.boundary_fragility(groups, model, labels), 0.0)

        with_straddler = np.vstack([groups, [[0.0]]])
        model, labels = clustering.fit_kmeans(with_straddler, 2)
        self.assertGreater(clustering.boundary_fragility(with_straddler, model, labels), 0.0)

    def test_returns_a_share_between_zero_and_one(self):
        table = fixtures.make_feature_table()
        matrix, _ = clustering.scale_features(table)
        model, labels = clustering.fit_kmeans(matrix, 4)
        result = clustering.boundary_fragility(matrix, model, labels)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_a_single_cluster_has_no_boundaries(self):
        table = fixtures.make_feature_table()
        matrix, _ = clustering.scale_features(table)
        model, labels = clustering.fit_kmeans(matrix, 1)
        self.assertEqual(clustering.boundary_fragility(matrix, model, labels), 0.0)


class TestCentroidsInRealUnits(unittest.TestCase):
    def setUp(self):
        self.table = fixtures.make_feature_table()
        self.matrix, self.scaler = clustering.scale_features(self.table)
        self.model, self.labels = clustering.fit_kmeans(self.matrix, 3)

    def test_shape_and_columns(self):
        centroids = clustering.centroids_in_real_units(self.model, self.scaler)
        self.assertEqual(centroids.shape, (3, len(config.FEATURE_COLUMNS)))
        self.assertEqual(centroids.columns.tolist(), config.FEATURE_COLUMNS)

    def test_values_land_inside_the_observed_range(self):
        """Inverse-transformed centroids must be physically plausible."""
        centroids = clustering.centroids_in_real_units(self.model, self.scaler)
        for column in config.FEATURE_COLUMNS:
            self.assertGreaterEqual(centroids[column].min(), self.table[column].min() - 1e-6)
            self.assertLessEqual(centroids[column].max(), self.table[column].max() + 1e-6)

    def test_matches_the_group_means(self):
        """A centroid is the mean of its members, so the two must agree."""
        centroids = clustering.centroids_in_real_units(self.model, self.scaler)
        observed = self.table.assign(cluster=self.labels).groupby("cluster")[
            config.FEATURE_COLUMNS
        ].mean()
        np.testing.assert_allclose(
            centroids.to_numpy(), observed.to_numpy(), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
