import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import tensorflow as tf

from kdp.model_advisor import ModelAdvisor, recommend_model_configuration


# Custom JSON encoder for handling TensorFlow dtypes
class TFJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tf.dtypes.DType):
            return obj.name
        return super().default(obj)


class TestModelAdvisor(unittest.TestCase):
    """Tests for the ModelAdvisor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample feature statistics
        self.mock_features_stats = {
            "numeric_stats": {
                "num1": {"mean": 0.5, "count": 1000, "var": 1.2, "dtype": tf.float32},
                "num2": {"mean": -0.2, "count": 1000, "var": 4.5, "dtype": tf.float32},
            },
            "categorical_stats": {
                "cat1": {"size": 4, "vocab": ["a", "b", "c", "d"], "dtype": tf.string},
                "cat2": {
                    "size": 100,
                    "vocab": [str(i) for i in range(100)],
                    "dtype": tf.int32,
                },
            },
            "text": {
                "text1": {
                    "size": 2000,
                    "vocab": ["sample", "words", "for", "testing"],
                    "sequence_length": 50,
                    "vocab_size": 2000,
                    "dtype": tf.string,
                }
            },
            "date_stats": {
                "date1": {
                    "mean_year": 2020,
                    "var_year": 2.5,
                    "mean_month_sin": 0.1,
                    "var_month_sin": 0.5,
                    "mean_month_cos": 0.2,
                    "var_month_cos": 0.6,
                    "cyclical_patterns": ["month", "dayofweek"],
                }
            },
        }

        # Initialize the ModelAdvisor with mock stats
        self.advisor = ModelAdvisor(self.mock_features_stats)

    def test_initialization(self):
        """Test that ModelAdvisor initializes correctly."""
        self.assertEqual(self.advisor.features_stats, self.mock_features_stats)
        self.assertEqual(self.advisor.recommendations, {})
        self.assertEqual(self.advisor.global_config, {})

    def test_analyze_numeric_features(self):
        """Test analysis of numeric features."""
        self.advisor._analyze_numeric_features()

        # Check if recommendations were generated for both numeric features
        self.assertIn("num1", self.advisor.recommendations)
        self.assertIn("num2", self.advisor.recommendations)

        # Check that the recommendations have the expected structure
        for feature in ["num1", "num2"]:
            recommendation = self.advisor.recommendations[feature]
            self.assertEqual(recommendation["feature_type"], "NumericalFeature")
            self.assertIsInstance(recommendation["preprocessing"], list)
            self.assertIsInstance(recommendation["config"], dict)
            self.assertIn("detected_distribution", recommendation)
            self.assertIn("distribution_confidence", recommendation)

    def test_analyze_categorical_features(self):
        """Test analysis of categorical features."""
        self.advisor._analyze_categorical_features()

        # Check if recommendations were generated for both categorical features
        self.assertIn("cat1", self.advisor.recommendations)
        self.assertIn("cat2", self.advisor.recommendations)

        # Verify specific recommendations based on vocabulary size
        # cat1 has small vocab (4), should recommend ONE_HOT_ENCODING
        self.assertIn(
            "ONE_HOT_ENCODING", self.advisor.recommendations["cat1"]["preprocessing"]
        )

        # cat2 has large vocab (100), should recommend HASHING
        self.assertIn("HASHING", self.advisor.recommendations["cat2"]["preprocessing"])

    def test_analyze_text_features(self):
        """Test analysis of text features."""
        self.advisor._analyze_text_features()

        # Check if recommendations were generated for text feature
        self.assertIn("text1", self.advisor.recommendations)

        # Verify text feature recommendations
        text_rec = self.advisor.recommendations["text1"]
        self.assertEqual(text_rec["feature_type"], "TextFeature")
        self.assertIn("TEXT_VECTORIZATION", text_rec["preprocessing"])
        self.assertIn("max_tokens", text_rec["config"])
        self.assertIn("embedding_dim", text_rec["config"])

    def test_analyze_date_features(self):
        """The recommendation must name options `DateFeature` actually reads.

        It used to describe `extract`, `timezone_aware` and `cyclical_encoding`.
        `DateFeature` reads only `format` and `add_season`; the components it
        encodes and the cyclical encoding of them are not configurable, so
        recommending them told the reader to write settings that do nothing.
        """
        self.advisor._analyze_date_features()

        self.assertIn("date1", self.advisor.recommendations)
        date_rec = self.advisor.recommendations["date1"]
        self.assertEqual(date_rec["feature_type"], "DateFeature")
        self.assertIn("DATE_ENCODING", date_rec["preprocessing"])

        # Only the two real options may appear.
        self.assertTrue(set(date_rec["config"]) <= {"format", "add_season"})
        self.assertNotIn("extract", date_rec["config"])
        self.assertNotIn("cyclical_encoding", date_rec["advanced_options"])

        # The fixture declares cyclical patterns, so the season one-hot is on.
        self.assertTrue(date_rec["config"].get("add_season"))

    def test_date_statistics_are_read_under_the_key_stats_writes(self):
        """`DatasetStatistics` writes "date"; this looked for "date_stats"."""
        from kdp.model_advisor import ModelAdvisor

        advisor = ModelAdvisor({"date": dict(self.mock_features_stats["date_stats"])})
        advisor._analyze_date_features()
        self.assertIn("date1", advisor.recommendations)

    def test_generate_global_recommendations(self):
        """Test generation of global recommendations."""
        # First analyze features to fill the recommendations dict
        self.advisor._analyze_numeric_features()
        self.advisor._analyze_categorical_features()
        self.advisor._analyze_text_features()
        self.advisor._analyze_date_features()

        # Then generate global recommendations
        self.advisor._generate_global_recommendations()

        # Verify global config recommendations
        self.assertIn("output_mode", self.advisor.global_config)
        self.assertIn("use_distribution_aware", self.advisor.global_config)

        # With mixed feature types and more than 3 features, should recommend tabular attention
        self.assertIn("tabular_attention", self.advisor.global_config)
        self.assertTrue(self.advisor.global_config["tabular_attention"])

        # Should recommend multi-resolution attention for mixed feature types
        self.assertEqual(
            self.advisor.global_config.get("tabular_attention_placement"),
            "multi_resolution",
        )

    def test_code_snippet_generation(self):
        """Test that the generated code snippet has the expected format."""
        # First analyze features
        self.advisor.analyze_feature_stats()

        # Generate code snippet
        code_snippet = self.advisor.generate_code_snippet()

        # Verify code snippet content
        self.assertIn("from kdp.features import (", code_snippet)
        self.assertIn(
            "NumericalFeature, CategoricalFeature, TextFeature, DateFeature,",
            code_snippet,
        )
        self.assertIn("FeatureType, CategoryEncodingOptions", code_snippet)
        self.assertIn("from kdp.processor import PreprocessingModel", code_snippet)
        self.assertIn("features = {", code_snippet)

        # Check that all features are included in the snippet
        for feature in ["num1", "num2", "cat1", "cat2", "text1", "date1"]:
            self.assertIn(f'"{feature}"', code_snippet)

        # Check that the model initialization is included
        self.assertIn("model = PreprocessingModel(", code_snippet)
        self.assertIn("features_specs=features", code_snippet)

    def test_analyze_feature_stats_with_empty_stats(self):
        """Test behavior when no feature statistics are provided."""
        empty_advisor = ModelAdvisor({})
        result = empty_advisor.analyze_feature_stats()
        self.assertEqual(result, {})

    def test_recommend_model_configuration_function(self):
        """Test the recommend_model_configuration function."""
        # Call the function with mock stats
        recommendations = recommend_model_configuration(self.mock_features_stats)

        # Verify the structure of the return value
        self.assertIn("features", recommendations)
        self.assertIn("global_config", recommendations)
        self.assertIn("code_snippet", recommendations)

        # Check that all features have recommendations
        features = recommendations["features"]
        for feature in ["num1", "num2", "cat1", "cat2", "text1", "date1"]:
            self.assertIn(feature, features)

    @patch("builtins.print")
    def test_end_to_end_workflow(self, mock_print):
        """Test the end-to-end workflow of feature analysis and recommendation."""
        # Create a temporary directory for feature statistics
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_path = os.path.join(temp_dir, "features_stats.json")

            # Save mock feature statistics to temporary file with custom encoder
            with open(stats_path, "w") as f:
                json.dump(self.mock_features_stats, f, cls=TFJSONEncoder)

            # Create mock DatasetStatistics with the recommend_model_configuration method
            mock_stats_calculator = MagicMock()
            mock_stats_calculator.features_stats = self.mock_features_stats
            mock_stats_calculator.recommend_model_configuration.return_value = (
                recommend_model_configuration(self.mock_features_stats)
            )

            # Get recommendations
            recommendations = mock_stats_calculator.recommend_model_configuration()

            # Verify recommendations
            self.assertIn("features", recommendations)
            self.assertIn("global_config", recommendations)
            self.assertIn("code_snippet", recommendations)

            # Check that code snippet can be evaluated (basic syntax check)
            code_snippet = recommendations["code_snippet"]
            try:
                compile(code_snippet, "<string>", "exec")
                valid_syntax = True
            except SyntaxError:
                valid_syntax = False

            self.assertTrue(
                valid_syntax, "Generated code snippet should have valid syntax"
            )


if __name__ == "__main__":
    unittest.main()
