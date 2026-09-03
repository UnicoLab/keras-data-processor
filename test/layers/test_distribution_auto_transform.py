"""Tests for `DistributionTransformLayer(transform_type="auto")`.

`auto` picks a transformation from the data's own shape -- whether it is
bounded, has zeros, or has negative values. That selection was untested, so a
transform that silently produced infinities or left a skewed column untouched
would not have been noticed.
"""

import unittest

import keras
import numpy as np
import pytest
import tensorflow as tf

from kdp.layers import DistributionTransformLayer

ROWS = 600


def _apply(data, **kwargs):
    keras.backend.clear_session()
    layer = DistributionTransformLayer(transform_type="auto", **kwargs)
    return np.asarray(layer(tf.constant(data, dtype=tf.float32)))


@pytest.mark.layers
class TestAutoTransformShapes(unittest.TestCase):
    """One case per branch of the candidate selection."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_strictly_positive_data_is_transformed_finitely(self):
        """Log-like transforms on a lognormal must not produce inf or nan."""
        data = self.rng.lognormal(0, 1, (ROWS, 1))
        out = _apply(data)
        self.assertTrue(np.isfinite(out).all())

    def test_a_skewed_column_comes_out_less_skewed(self):
        """That is the point of choosing a transform automatically."""

        def skew(x):
            centred = x - x.mean()
            return float((centred**3).mean() / (x.std() ** 3 + 1e-12))

        data = self.rng.lognormal(0, 1, (ROWS, 1))
        self.assertLess(abs(skew(_apply(data))), abs(skew(data)))

    def test_data_with_zeros_is_handled(self):
        """A log transform would be undefined at zero."""
        data = np.clip(self.rng.normal(0, 1, (ROWS, 1)), 0, None)
        out = _apply(data)
        self.assertTrue(np.isfinite(out).all())

    def test_negative_values_are_handled(self):
        """Candidates for mixed-sign data must not require positivity."""
        data = self.rng.normal(0, 1, (ROWS, 1))
        out = _apply(data)
        self.assertTrue(np.isfinite(out).all())

    def test_bounded_data_is_handled(self):
        """Values inside (0, 1) get the bounded candidates."""
        data = self.rng.uniform(0.01, 0.99, (ROWS, 1))
        out = _apply(data)
        self.assertTrue(np.isfinite(out).all())

    def test_shape_is_preserved(self):
        """The transform is elementwise; it must not change the width."""
        data = self.rng.lognormal(0, 1, (ROWS, 3))
        self.assertEqual(_apply(data).shape, (ROWS, 3))

    def test_explicit_candidates_are_respected(self):
        """Restricting the list must still produce a usable column."""
        data = self.rng.lognormal(0, 1, (ROWS, 1))
        out = _apply(data, auto_candidates=["log", "sqrt"])
        self.assertTrue(np.isfinite(out).all())

    def test_auto_survives_serialization(self):
        """A saved model has to rebuild the same selection."""
        layer = DistributionTransformLayer(
            transform_type="auto", auto_candidates=["log", "sqrt"]
        )
        restored = DistributionTransformLayer.from_config(layer.get_config())
        self.assertEqual(restored.transform_type, "auto")
        self.assertEqual(restored.auto_candidates, ["log", "sqrt"])

    def test_auto_traces_into_a_functional_model(self):
        """It has to work in graph mode, not just eagerly."""
        keras.backend.clear_session()
        inputs = keras.Input(shape=(1,))
        model = keras.Model(
            inputs, DistributionTransformLayer(transform_type="auto")(inputs)
        )
        out = np.asarray(model(tf.constant([[1.0], [2.0], [3.0]])))
        self.assertTrue(np.isfinite(out).all())


@pytest.mark.layers
class TestAutoWithAnEmptyCandidateList(unittest.TestCase):
    """`auto_candidates=[]` reaches a branch nothing else does.

    Passing `None` -- the default -- makes `__init__` fill in the full list, so
    the code that builds candidates from the data's own shape only runs when a
    caller asks for `auto` with an explicitly empty list. That path had never
    been executed.
    """

    CASES = {
        "strictly positive": lambda rng: rng.gamma(2.0, 2.0, (200, 1)),
        "positive with zeros": lambda rng: np.abs(rng.normal(0, 2, (200, 1))),
        "mixed signs": lambda rng: rng.normal(0, 5, (200, 1)),
        "bounded in 0-1": lambda rng: rng.uniform(0.01, 0.99, (200, 1)),
    }

    def test_it_produces_finite_values_of_the_right_shape(self):
        rng = np.random.default_rng(4)
        for label, make in self.CASES.items():
            with self.subTest(case=label):
                data = make(rng).astype("float32")
                layer = DistributionTransformLayer(
                    transform_type="auto",
                    auto_candidates=[],
                )
                output = np.asarray(layer(tf.constant(data), training=True))
                self.assertEqual(output.shape, data.shape)
                self.assertTrue(np.isfinite(output).all())

    def test_it_still_round_trips_through_a_config(self):
        layer = DistributionTransformLayer(
            transform_type="auto",
            auto_candidates=[],
        )
        rebuilt = DistributionTransformLayer.from_config(layer.get_config())
        self.assertEqual(rebuilt.transform_type, "auto")


if __name__ == "__main__":
    unittest.main()
