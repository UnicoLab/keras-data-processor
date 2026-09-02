"""Branch coverage for `ModelAdvisor`, the engine behind `auto_configure`.

The advisor turns statistics into the recommendations and the code snippet a
user copies. Most of its branches had no test, so a recommendation naming an
option the feature ignores -- or one that contradicts its own note -- passed
unnoticed. Every configuration key asserted here is one the target class
actually reads.
"""

import unittest

import pytest

from kdp.model_advisor import ModelAdvisor

# Keys the recommendation may put in `config` that a feature genuinely honours.
REAL_FEATURE_OPTIONS = {
    "name",
    "feature_type",
    "scale",
    "format",
    "add_season",
    "prefered_distribution",
    "preferred_distribution",
    "category_encoding",
    "embedding_size",
    "hash_bucket_size",
    "salt",
    "hash_with_embedding",
    "max_tokens",
    "output_sequence_length",
    "stop_words",
    "bin_boundaries",
    "num_bins",
    "use_embedding",
    "embedding_dim",
}

NUMERIC_STATS = {
    "mean": 5.0,
    "var": 2.0,
    "count": 1000,
    "min": 0.0,
    "max": 10.0,
    "dtype": "float32",
}


def _blank_recommendation():
    return {
        "feature_type": "NumericalFeature",
        "preprocessing": [],
        "config": {},
        "advanced_options": {},
        "notes": [],
    }


@pytest.mark.unit
class TestNumericRecommendationBranches(unittest.TestCase):
    """One case per distribution the recommender knows about."""

    def setUp(self):
        self.advisor = ModelAdvisor({"numeric_stats": {"x": dict(NUMERIC_STATS)}})

    def _recommend(self, dist_type):
        recommendation = _blank_recommendation()
        self.advisor._recommend_numeric_preprocessing(
            recommendation, dist_type, dict(NUMERIC_STATS)
        )
        return recommendation

    def test_every_distribution_produces_a_preprocessing_choice(self):
        """A branch that falls through leaves the feature unconfigured."""
        for dist in (
            "normal",
            "uniform",
            "heavy_tailed",
            "log_normal",
            "periodic",
            "sparse",
            "multimodal",
            "discrete",
        ):
            recommendation = self._recommend(dist)
            self.assertTrue(recommendation["preprocessing"], dist)
            self.assertTrue(recommendation["notes"], f"{dist} has no explanation")

    def test_normal_data_is_standardised(self):
        """The default choice for a symmetric column."""
        self.assertIn("FLOAT_NORMALIZED", self._recommend("normal")["preprocessing"])

    def test_uniform_data_gets_a_usable_scale(self):
        """`FLOAT_RESCALED` is the identity unless `scale` is set.

        The branch used to record only `min` and `max`, which
        `NumericalFeature` does not read, so the recommendation left the column
        untouched while its note claimed rescaling.
        """
        recommendation = self._recommend("uniform")
        self.assertIn("FLOAT_RESCALED", recommendation["preprocessing"])
        self.assertIn("scale", recommendation["config"])
        # min 0, max 10 -> a factor of 1/10.
        self.assertAlmostEqual(recommendation["config"]["scale"], 0.1, places=6)

    def test_a_degenerate_range_does_not_divide_by_zero(self):
        """A constant column has zero span."""
        recommendation = _blank_recommendation()
        self.advisor._recommend_numeric_preprocessing(
            recommendation, "uniform", {"min": 3.0, "max": 3.0}
        )
        self.assertEqual(recommendation["config"]["scale"], 1.0)

    def test_skewed_distributions_ask_for_distribution_aware_encoding(self):
        """These are the shapes a fixed transform handles badly."""
        for dist in ("heavy_tailed", "log_normal", "periodic", "sparse", "multimodal"):
            recommendation = self._recommend(dist)
            self.assertIn("DISTRIBUTION_AWARE", recommendation["preprocessing"], dist)
            self.assertEqual(
                recommendation["config"]["prefered_distribution"], dist, dist
            )


@pytest.mark.unit
class TestCategoricalRecommendationBranches(unittest.TestCase):
    """Cardinality decides the encoding."""

    def _analyse(self, size):
        advisor = ModelAdvisor(
            {
                "categorical_stats": {
                    "c": {
                        "size": size,
                        "vocab": [str(i) for i in range(size)],
                        "dtype": "string",
                    }
                }
            }
        )
        advisor._analyze_categorical_features()
        return advisor.recommendations["c"]

    def test_a_small_vocabulary_is_encoded_directly(self):
        """A handful of categories does not need hashing."""
        recommendation = self._analyse(4)
        self.assertEqual(recommendation["feature_type"], "CategoricalFeature")
        self.assertIn("category_encoding", recommendation["config"])

    def test_a_large_vocabulary_is_handled_too(self):
        """High cardinality must still produce a usable configuration."""
        recommendation = self._analyse(5000)
        self.assertEqual(recommendation["feature_type"], "CategoricalFeature")
        self.assertTrue(recommendation["preprocessing"])

    def test_configuration_keys_are_options_the_feature_reads(self):
        """A recommended key the class ignores misleads the reader."""
        for size in (3, 40, 5000):
            config = self._analyse(size)["config"]
            unknown = set(config) - REAL_FEATURE_OPTIONS
            self.assertEqual(
                unknown, set(), f"size {size} recommends ignored options: {unknown}"
            )


@pytest.mark.unit
class TestAdvisorHelpers(unittest.TestCase):
    """The small calculations the recommendations depend on."""

    def setUp(self):
        self.advisor = ModelAdvisor({})

    def test_embedding_dimension_grows_with_the_data(self):
        """More samples justify a wider embedding, and it stays positive."""
        small = self.advisor._calculate_embedding_dim(100)
        large = self.advisor._calculate_embedding_dim(1_000_000)
        self.assertGreater(small, 0)
        self.assertGreaterEqual(large, small)

    def test_bin_count_grows_with_the_data(self):
        """Same shape of rule for discretisation."""
        small = self.advisor._calculate_num_bins(100)
        large = self.advisor._calculate_num_bins(1_000_000)
        self.assertGreater(small, 0)
        self.assertGreaterEqual(large, small)

    def test_distribution_detection_returns_a_known_type(self):
        """The detector feeds the recommender, so it must never return junk."""
        known = {
            "normal",
            "uniform",
            "heavy_tailed",
            "log_normal",
            "periodic",
            "sparse",
            "multimodal",
            "discrete",
            "exponential",
            "beta",
            "gamma",
            "poisson",
            "cauchy",
            "zero_inflated",
            "bounded",
            "ordinal",
        }
        cases = [
            # skewness, kurtosis, zero_ratio, autocorr, bimodality
            (0.0, 3.0, 0.0, 0.0, 0.0),
            (3.0, 12.0, 0.0, 0.0, 0.0),
            (0.0, 1.5, 0.0, 0.0, 0.9),
            (0.0, 3.0, 0.85, 0.0, 0.0),
            (0.0, 3.0, 0.0, 0.95, 0.0),
        ]
        for skewness, kurtosis, zeros, autocorr, bimodality in cases:
            dist_type, confidence = self.advisor._detect_distribution_type(
                dict(NUMERIC_STATS), skewness, kurtosis, zeros, autocorr, bimodality
            )
            self.assertIn(dist_type, known)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


@pytest.mark.unit
class TestGlobalRecommendations(unittest.TestCase):
    """The model-level settings the advisor suggests."""

    def test_global_config_only_names_real_model_options(self):
        """These go into `PreprocessingModel`, which has no `**kwargs`."""
        import inspect

        from kdp import PreprocessingModel

        advisor = ModelAdvisor(
            {
                "numeric_stats": {f"n{i}": dict(NUMERIC_STATS) for i in range(12)},
                "categorical_stats": {
                    "c": {"size": 50, "vocab": [str(i) for i in range(50)]}
                },
            }
        )
        advisor.analyze_feature_stats()

        real = set(inspect.signature(PreprocessingModel.__init__).parameters) - {"self"}
        # The advisor also reports descriptive keys that are not constructor
        # arguments; only the ones that look like settings must be real.
        suspicious = {
            key
            for key in advisor.global_config
            if key.startswith(("use_", "tabular_", "transfo_", "feature_"))
        }
        self.assertEqual(
            suspicious - real,
            set(),
            f"global config names options PreprocessingModel lacks: {suspicious - real}",
        )


if __name__ == "__main__":
    unittest.main()
