"""Tests for the paths a served model actually takes.

Building a preprocessor and calling it are separate failure surfaces: several
defects only appeared when the model was invoked, and specifically when it was
invoked the way serving does it -- one row at a time.
"""

import keras
import unittest

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import PreprocessingModel
from kdp.features import FeatureType, PassthroughFeature
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.layers.distribution_aware_encoder_layer import DistributionAwareEncoder


def _write_dataset(directory, rows: int = 200):
    """Write a CSV covering the feature types these tests exercise."""
    rng = np.random.default_rng(7)
    csv_path = directory / "data.csv"
    pd.DataFrame(
        {
            "age": rng.normal(40, 12, rows),
            "income": rng.lognormal(10, 1, rows),
            "rating": rng.integers(0, 5, rows).astype(float),
            "city": rng.choice(["paris", "tokyo", "lima"], rows),
            "joined": pd.date_range("2020-01-01", periods=rows).strftime("%Y-%m-%d"),
            "record_id": rng.normal(size=rows),
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _single_row_batch() -> dict:
    """One row of every feature, i.e. what a served model is handed."""
    return {
        "age": tf.constant([[35.0]]),
        "income": tf.constant([[50000.0]]),
        "rating": tf.constant([[2.0]]),
        "city": tf.constant([["paris"]]),
        "joined": tf.constant([["2021-06-15"]]),
        "record_id": tf.constant([[0.5]]),
    }


ALL_SPECS = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "rating": FeatureType.FLOAT_DISCRETIZED,
    "city": FeatureType.STRING_CATEGORICAL,
    "joined": FeatureType.DATE,
}


class TestDateParsingBatchSizes(unittest.TestCase):
    """DateParsingLayer must handle any batch size, one row included."""

    def test_single_row_batch(self):
        """A (1, 1) batch parses instead of collapsing to a scalar."""
        parsed = DateParsingLayer()(tf.constant([["2021-06-15"]])).numpy()
        self.assertEqual(parsed.shape, (1, 4))
        np.testing.assert_array_equal(parsed[0][:3], [2021, 6, 15])

    def test_batch_sizes_agree(self):
        """Parsing one row gives the same answer as parsing it in a larger batch."""
        alone = DateParsingLayer()(tf.constant([["2020-02-29"]])).numpy()
        together = DateParsingLayer()(
            tf.constant([["2020-02-29"], ["1999-12-31"], ["2024-07-04"]])
        ).numpy()
        np.testing.assert_array_equal(alone[0], together[0])

    def test_one_dimensional_input(self):
        """A flat vector of dates is accepted too."""
        parsed = DateParsingLayer()(tf.constant(["2021-06-15", "2022-01-02"])).numpy()
        self.assertEqual(parsed.shape, (2, 4))


class TestDistributionAwareConstantBatch(unittest.TestCase):
    """Distribution detection must survive a batch with no spread."""

    def test_constant_batch_does_not_raise(self):
        """min == max is exactly what a single-row batch looks like."""
        keras.backend.clear_session()
        layer = DistributionAwareEncoder()
        output = layer(tf.constant([[50000.0]]))
        self.assertEqual(output.shape[0], 1)

    def test_large_magnitude_constant_batch(self):
        """A fixed epsilon vanishes at large magnitudes; the range must still widen."""
        keras.backend.clear_session()
        layer = DistributionAwareEncoder()
        output = layer(tf.constant([[1e7], [1e7], [1e7]]))
        self.assertTrue(np.all(np.isfinite(np.asarray(output))))


class TestDiscretizedFeature(unittest.TestCase):
    """FLOAT_DISCRETIZED must produce a callable model from num_bins alone."""

    def test_num_bins_only_is_usable(
        self,
    ):
        """The documented `num_bins` configuration builds *and* runs."""
        keras.backend.clear_session()
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_dataset(tmp_path)
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={"rating": FeatureType.FLOAT_DISCRETIZED},
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()
            output = np.asarray(preprocessor.model({"rating": tf.constant([[2.0]])}))

        # One-hot over the derived bins: exactly one active entry.
        self.assertEqual(output.sum(), 1.0)

    def test_derived_boundaries_are_strictly_increasing(self):
        """Boundaries must be usable even for a feature with no variance."""
        boundaries = PreprocessingModel._derive_bin_boundaries(
            5, {"mean": 3.0, "var": 0.0}
        )
        self.assertEqual(len(boundaries), 4)
        self.assertTrue(all(b < c for b, c in zip(boundaries, boundaries[1:])))

    def test_single_bin_has_no_interior_boundaries(self):
        """A one-bucket request needs no cut points."""
        self.assertEqual(
            PreprocessingModel._derive_bin_boundaries(1, {"mean": 0.0, "var": 1.0}), []
        )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"use_distribution_aware": True},
        {"tabular_attention": True},
        {"output_mode": "dict"},
    ],
    ids=["basic", "distribution_aware", "tabular_attention", "dict_mode"],
)
def test_single_row_prediction(config, tmp_path):
    """A model built from a full feature set answers a one-row request."""
    keras.backend.clear_session()
    csv_path = _write_dataset(tmp_path)
    preprocessor = PreprocessingModel(
        path_data=str(csv_path),
        features_specs=dict(ALL_SPECS),
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
        **config,
    )
    preprocessor.build_preprocessor()

    batch = _single_row_batch()
    batch.pop("record_id")
    output = preprocessor.model(batch)

    if isinstance(output, dict):
        for value in output.values():
            assert int(value.shape[0]) == 1
    else:
        assert int(output.shape[0]) == 1


class TestPassthroughExcludedFromOutput(unittest.TestCase):
    """include_passthrough_in_output=False must still build a valid graph."""

    def _build(self, tmp_path):
        csv_path = _write_dataset(tmp_path)
        preprocessor = PreprocessingModel(
            path_data=str(csv_path),
            features_specs={
                "age": FeatureType.FLOAT_NORMALIZED,
                "record_id": PassthroughFeature(name="record_id", dtype=tf.float32),
            },
            features_stats_path=str(tmp_path / "stats.json"),
            overwrite_stats=True,
            include_passthrough_in_output=False,
        )
        preprocessor.build_preprocessor()
        return preprocessor

    def test_model_builds_and_passes_values_through_untouched(self):
        """A raw Input cannot be an output; the values must still be unchanged."""
        import tempfile
        from pathlib import Path

        keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = self._build(Path(tmp))
            output = preprocessor.model(
                {"age": tf.constant([[35.0]]), "record_id": tf.constant([[0.5]])}
            )

        self.assertIn("processed", output)
        self.assertIn("passthrough", output)
        np.testing.assert_allclose(
            np.asarray(output["passthrough"]["record_id"]), [[0.5]]
        )


class TestTimeSeriesRowAlignment(unittest.TestCase):
    """Time series features must stay concatenable with ordinary ones."""

    def _build(self, tmp_path, **ts_config):
        from kdp import TimeSeriesFeature

        rng = np.random.default_rng(3)
        rows = 120
        csv_path = tmp_path / "series.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=rows).strftime("%Y-%m-%d"),
                "sales": np.linspace(100, 300, rows) + rng.normal(0, 5, rows),
                "temp": rng.normal(20, 3, rows),
            }
        ).to_csv(csv_path, index=False)

        preprocessor = PreprocessingModel(
            path_data=str(csv_path),
            features_specs={
                "sales": TimeSeriesFeature(name="sales", sort_by="date", **ts_config),
                "temp": FeatureType.FLOAT_NORMALIZED,
            },
            features_stats_path=str(tmp_path / "stats.json"),
            overwrite_stats=True,
        )
        preprocessor.build_preprocessor()
        return preprocessor

    def test_time_series_feature_concatenates_with_a_numeric_feature(self):
        """Dropping warm-up rows left the column shorter than its neighbours."""
        import tempfile
        from pathlib import Path

        for config in (
            {"lag_config": {"lags": [1, 7]}},
            {"rolling_stats_config": {"window_size": 7, "statistics": ["mean"]}},
            {"differencing_config": {"order": 1}},
            {"moving_average_config": {"periods": [7]}},
        ):
            with self.subTest(config=next(iter(config))):
                keras.backend.clear_session()
                with tempfile.TemporaryDirectory() as tmp:
                    preprocessor = self._build(Path(tmp), **config)
                    rows = 20
                    output = preprocessor.model(
                        {
                            "sales": tf.constant(
                                np.linspace(100, 200, rows).reshape(-1, 1),
                                dtype=tf.float32,
                            ),
                            "temp": tf.constant(
                                np.full((rows, 1), 20.0), dtype=tf.float32
                            ),
                        }
                    )
                self.assertEqual(int(output.shape[0]), rows)

    def test_explicit_drop_na_is_overridden_for_models(self):
        """An explicit drop_na=True cannot be honoured inside a model."""
        from kdp import TimeSeriesFeature

        feature = TimeSeriesFeature(
            name="sales",
            sort_by="date",
            lag_config={"lags": [1, 3], "drop_na": True},
        )
        model_layers = feature.build_layers()
        self.assertFalse(model_layers[0].drop_na)

        # Driving the layers directly still honours the request.
        standalone = feature.build_layers(row_preserving=False)
        self.assertTrue(standalone[0].drop_na)


if __name__ == "__main__":
    unittest.main()
