"""Tests for the model-level advanced numerical embedding settings.

`PreprocessingModel` accepts `embedding_dim`, `mlp_hidden_units`, `num_bins`,
`init_min`, `init_max`, `dropout_rate` and `use_batch_norm` for the advanced
numerical embedding. They were handed to `add_processing_step`, whose layer
creator was `lambda **kwargs: embedding_layer` -- every one of them was
discarded, and the feature's own defaults applied instead. The output width was
the same no matter what `embedding_dim` was set to.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.features import NumericalFeature

COLUMNS = ("a", "b")


def _dataset(directory, rows: int = 300):
    """Two numeric columns."""
    rng = np.random.default_rng(0)
    csv_path = directory / "numbers.csv"
    pd.DataFrame({c: rng.normal(50, 10, rows) for c in COLUMNS}).to_csv(
        csv_path, index=False
    )
    return csv_path


def _width(tmp_path, specs=None, **kwargs):
    keras.backend.clear_session()
    preprocessor = PreprocessingModel(
        path_data=str(_dataset(tmp_path)),
        features_specs=specs or {c: FeatureType.FLOAT_NORMALIZED for c in COLUMNS},
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
        use_advanced_numerical_embedding=True,
        **kwargs,
    )
    preprocessor.build_preprocessor()
    batch = {c: tf.constant([[50.0]]) for c in COLUMNS}
    return int(preprocessor.model(batch).shape[-1])


@pytest.mark.unit
class TestAdvancedEmbeddingOptions(unittest.TestCase):
    """The model-level settings have to reach the layer."""

    def test_embedding_dim_drives_the_output_width(self):
        """Width is one embedding per feature; it used to be fixed at 16."""
        for dim in (4, 8, 16, 32):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(
                    _width(Path(tmp), embedding_dim=dim), len(COLUMNS) * dim
                )

    def test_a_feature_setting_beats_the_model_setting(self):
        """An explicit per-feature value must not be overwritten."""
        specs = {
            "a": NumericalFeature(
                name="a",
                feature_type=FeatureType.FLOAT_NORMALIZED,
                embedding_dim=4,
            ),
            "b": FeatureType.FLOAT_NORMALIZED,
        }
        with tempfile.TemporaryDirectory() as tmp:
            # 4 for the feature that asked, 16 for the one that did not.
            self.assertEqual(_width(Path(tmp), specs=specs, embedding_dim=16), 4 + 16)

    def test_other_settings_reach_the_layer(self):
        """These changed nothing before; now they build without being dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            width = _width(
                Path(tmp),
                embedding_dim=8,
                mlp_hidden_units=32,
                num_bins=20,
                dropout_rate=0.3,
                use_batch_norm=False,
                init_min=-2.0,
                init_max=2.0,
            )
        self.assertEqual(width, len(COLUMNS) * 8)

    def test_default_is_unchanged(self):
        """Without an explicit dim the previous default still applies."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_width(Path(tmp)), len(COLUMNS) * 8)


if __name__ == "__main__":
    unittest.main()
