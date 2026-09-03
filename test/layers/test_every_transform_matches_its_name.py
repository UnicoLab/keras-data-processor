"""Each named transform has to compute the function it is named after.

The other tests here check properties -- bounded output, per-feature scaling,
order preserved. A transform can satisfy every one of those and still not be
the function on the label: `min-max` returned its input untouched and passed a
bounded-output test, because the input happened to be bounded.

So each one is compared against its definition, written out in numpy.
"""

import unittest

import numpy as np
import pytest
import tensorflow as tf

from kdp.layers.distribution_transform_layer import DistributionTransformLayer

EPSILON = 1e-10


def _positive(seed=0, rows=400):
    return (
        np.abs(np.random.default_rng(seed).gamma(2.0, 2.0, (rows, 1))) + 0.5
    ).astype(
        "float32",
    )


def _mixed(seed=1, rows=400):
    return np.random.default_rng(seed).normal(0, 3, (rows, 1)).astype("float32")


def _unit(seed=2, rows=400):
    return np.random.default_rng(seed).uniform(0.05, 0.95, (rows, 1)).astype("float32")


def _yeo_johnson(x, lam):
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    if lam == 0:
        out[positive] = np.log1p(x[positive])
    else:
        out[positive] = ((x[positive] + 1) ** lam - 1) / lam
    negative = ~positive
    if lam == 2:
        out[negative] = -np.log1p(-x[negative])
    else:
        out[negative] = -(((-x[negative] + 1) ** (2 - lam) - 1) / (2 - lam))
    return out


@pytest.mark.layers
class TestEveryTransformMatchesItsName(unittest.TestCase):
    """Every transform against the function it claims to be."""

    def _apply(self, kind, data, **kwargs):
        layer = DistributionTransformLayer(transform_type=kind, **kwargs)
        return np.asarray(layer(tf.constant(data)), dtype=np.float64)

    def _assert_matches(self, kind, actual, expected, tolerance=1e-3):
        actual = np.asarray(actual, dtype=np.float64).ravel()
        expected = np.asarray(expected, dtype=np.float64).ravel()
        self.assertEqual(actual.shape, expected.shape, kind)
        relative = np.max(np.abs(actual - expected) / (np.abs(expected) + 1e-6))
        self.assertLess(
            relative, tolerance, f"{kind}: max relative error {relative:.2e}"
        )

    def test_none_is_the_identity(self):
        data = _mixed()
        self._assert_matches("none", self._apply("none", data), data)

    def test_log(self):
        data = _positive()
        self._assert_matches("log", self._apply("log", data), np.log(data + EPSILON))

    def test_sqrt(self):
        data = _positive()
        self._assert_matches("sqrt", self._apply("sqrt", data), np.sqrt(data))

    def test_arcsinh(self):
        data = _mixed()
        self._assert_matches("arcsinh", self._apply("arcsinh", data), np.arcsinh(data))

    def test_cube_root(self):
        data = _mixed()
        self._assert_matches(
            "cube-root",
            self._apply("cube-root", data),
            np.sign(data) * np.abs(data) ** (1 / 3),
        )

    def test_logit(self):
        data = _unit()
        self._assert_matches(
            "logit",
            self._apply("logit", data),
            np.log(data / (1 - data)),
        )

    def test_box_cox(self):
        data = _positive()
        self._assert_matches(
            "box-cox (lambda 0)",
            self._apply("box-cox", data, lambda_param=0.0),
            np.log(data + EPSILON),
        )
        for lam in (0.5, 1.0, 2.0):
            self._assert_matches(
                f"box-cox (lambda {lam})",
                self._apply("box-cox", data, lambda_param=lam),
                (data**lam - 1) / lam,
            )

    def test_yeo_johnson(self):
        data = _mixed()
        for lam in (0.0, 0.5, 1.0):
            self._assert_matches(
                f"yeo-johnson (lambda {lam})",
                self._apply("yeo-johnson", data, lambda_param=lam),
                _yeo_johnson(data, lam),
            )

    def test_robust_scale(self):
        data = _mixed()
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        self._assert_matches(
            "robust-scale",
            self._apply("robust-scale", data),
            (data - np.median(data)) / iqr,
            tolerance=1e-2,
        )

    def test_min_max(self):
        """The one this file was written for: it returned its input."""
        data = _mixed()
        low, high = -1.0, 1.0
        scaled = (data - data.min()) / (data.max() - data.min())
        self._assert_matches(
            "min-max",
            self._apply("min-max", data, min_value=low, max_value=high),
            low + scaled * (high - low),
            tolerance=1e-2,
        )

    def test_quantile_maps_ranks_onto_the_documented_interval(self):
        """Ranks to [-1, 1], which is what the implementation says it does."""
        data = _mixed()
        actual = self._apply("quantile", data)
        ranks = np.argsort(np.argsort(data.ravel()))
        expected = 2.0 * ((ranks + 0.5) / data.size) - 1.0
        self._assert_matches("quantile", actual, expected, tolerance=1e-2)

    def test_every_named_transform_is_covered_here(self):
        """A transform added later must be measured too, not just listed."""
        layer = DistributionTransformLayer()
        named = set(layer._valid_transforms) - {"auto"}
        covered = {
            "none",
            "log",
            "sqrt",
            "box-cox",
            "yeo-johnson",
            "arcsinh",
            "cube-root",
            "logit",
            "quantile",
            "robust-scale",
            "min-max",
        }
        self.assertEqual(
            named - covered,
            set(),
            "these transforms have no test against their definition",
        )
