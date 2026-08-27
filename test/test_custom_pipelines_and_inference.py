"""Tests for custom preprocessing pipelines and inference data conversion.

Both are documented entry points whose defects only appear on the spellings the
documentation actually recommends.
"""

import unittest

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from kdp import PreprocessingModel
from kdp.features import Feature, FeatureType, NumericalFeature
from kdp.inference.base import InferenceFormatter


def _write_dataset(directory, rows: int = 100):
    """Write a small CSV with one numeric and one categorical column."""
    rng = np.random.default_rng(5)
    csv_path = directory / "data.csv"
    pd.DataFrame(
        {
            "value": rng.normal(10, 2, rows),
            "label": rng.choice(["x", "y"], rows),
        }
    ).to_csv(csv_path, index=False)
    return csv_path


class TestCustomPreprocessorSpellings(unittest.TestCase):
    """A custom pipeline step may be a name, a class, or a built layer."""

    def _build(self, tmp_path, preprocessors, **kwargs):
        preprocessor = PreprocessingModel(
            path_data=str(_write_dataset(tmp_path)),
            features_specs={
                "value": Feature(
                    name="value",
                    feature_type=FeatureType.FLOAT_NORMALIZED,
                    preprocessors=preprocessors,
                    **kwargs,
                ),
            },
            features_stats_path=str(tmp_path / "stats.json"),
            overwrite_stats=True,
        )
        preprocessor.build_preprocessor()
        return preprocessor

    def test_layer_names_as_strings(self):
        """The docs lead with string layer names, which used to raise."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = self._build(
                Path(tmp), ["Rescaling", "Dense"], scale=2.0, units=4
            )
            output = preprocessor.model({"value": tf.constant([[3.0]])})
        self.assertEqual(int(output.shape[-1]), 4)

    def test_layer_classes(self):
        """Class objects keep working."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = self._build(
                Path(tmp),
                [keras.layers.Rescaling, keras.layers.Dense],
                scale=2.0,
                units=3,
            )
            output = preprocessor.model({"value": tf.constant([[3.0]])})
        self.assertEqual(int(output.shape[-1]), 3)

    def test_prebuilt_layer_instances(self):
        """An already-built layer is used as-is rather than re-instantiated."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = self._build(
                Path(tmp), [keras.layers.Rescaling(scale=3.0, name="triple")]
            )
            output = preprocessor.model({"value": tf.constant([[2.0]])})
        np.testing.assert_allclose(np.asarray(output), [[6.0]], rtol=1e-5)

    def test_repeated_layer_type_gets_distinct_names(self):
        """The same layer type twice in one pipeline must not collide."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = self._build(Path(tmp), ["Rescaling", "Rescaling"], scale=2.0)
            output = preprocessor.model({"value": tf.constant([[1.0]])})
        np.testing.assert_allclose(np.asarray(output), [[4.0]], rtol=1e-5)


class TestInferenceTensorConversion(unittest.TestCase):
    """Column types are decided from the values that are present."""

    def setUp(self):
        self.formatter = InferenceFormatter.__new__(InferenceFormatter)

    def _convert(self, values):
        return self.formatter._convert_to_tensors({"column": values})["column"]

    def test_numeric_column_with_missing_values(self):
        """Gaps become NaN and the column stays float."""
        result = self._convert([1.0, 2.0, np.nan])
        self.assertEqual(result.dtype, tf.float32)
        self.assertTrue(np.isnan(np.asarray(result)[-1]))

    def test_categorical_column_with_missing_values(self):
        """A string column with a gap used to raise "mixed types"."""
        result = self._convert(["a", "b", np.nan])
        self.assertEqual(result.dtype, tf.string)
        self.assertEqual(np.asarray(result)[-1], b"")

    def test_string_column_stays_string(self):
        """Dates and categories are not coerced to float."""
        self.assertEqual(self._convert(["2021-01-01", "2021-01-02"]).dtype, tf.string)

    def test_integer_column_becomes_float(self):
        """Numeric columns are normalised to float32."""
        self.assertEqual(self._convert([1, 2, 3]).dtype, tf.float32)

    def test_all_missing_column_is_treated_as_string(self):
        """With nothing present there is no numeric evidence."""
        result = self._convert([np.nan, None])
        self.assertEqual(result.dtype, tf.string)


class TestTimeSeriesValidationAcceptsTensors(unittest.TestCase):
    """The validator must accept what the companion formatter produces."""

    def test_tensor_values_pass_validation(self):
        """Tensors were rejected, so predict() refused the formatter's output."""
        import tempfile
        from pathlib import Path

        from kdp import TimeSeriesFeature

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = 60
            csv_path = tmp_path / "series.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range("2022-01-01", periods=rows).strftime(
                        "%Y-%m-%d"
                    ),
                    "sales": np.linspace(1, 100, rows),
                }
            ).to_csv(csv_path, index=False)

            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "sales": TimeSeriesFeature(
                        name="sales", sort_by="date", lag_config={"lags": [1]}
                    ),
                },
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()

            # A tensor of several time steps is valid history. The sort column
            # has to travel with it, which the validator checks separately.
            self.assertTrue(
                preprocessor._validate_time_series_inference_data(
                    {
                        "sales": tf.constant([[1.0], [2.0], [3.0]]),
                        "date": tf.constant(
                            [["2022-01-01"], ["2022-01-02"], ["2022-01-03"]]
                        ),
                    }
                )
            )

            # A bare scalar still is not.
            with self.assertRaises(ValueError):
                preprocessor._validate_time_series_inference_data(
                    {"sales": 1.0, "date": ["2022-01-01"]}
                )


class TestNumericalFeatureEmbeddingLayer(unittest.TestCase):
    """get_embedding_layer keeps working without the unused input_shape."""

    def test_input_shape_is_optional(self):
        """It is accepted for compatibility but never read."""
        feature = NumericalFeature(name="value", embedding_dim=4)
        with_shape = feature.get_embedding_layer(input_shape=(None, 1))
        without_shape = feature.get_embedding_layer()
        self.assertEqual(with_shape.embedding_dim, without_shape.embedding_dim)


if __name__ == "__main__":
    unittest.main()
