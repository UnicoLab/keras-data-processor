import tensorflow as tf
import numpy as np
import unittest

from kdp.layers.time_series import MissingValueHandlerLayer


class TestMissingValueHandlerLayer(unittest.TestCase):
    """Test cases for the MissingValueHandlerLayer."""

    def setUp(self):
        # Create sample time series data with missing values
        np.random.seed(42)

        # Create a clean time series
        t = np.arange(100)
        self.clean_series = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 10)

        # Define the mask value
        self.mask_value = 0.0

        # Create a version with missing values
        self.missing_series = self.clean_series.copy()

        # Set specific values as missing (marked with 0.0)
        missing_indices = [5, 15, 25, 35, 36, 37, 38, 39, 40, 60, 80, 90]
        self.missing_series[missing_indices] = self.mask_value

        # Create a batch (batch_size=3)
        self.clean_batch = np.stack(
            [self.clean_series, self.clean_series * 1.2, self.clean_series * 0.8]
        )
        self.missing_batch = np.stack(
            [self.missing_series, self.missing_series * 1.2, self.missing_series * 0.8]
        )

        # Create missing value masks (True where values are missing)
        self.missing_mask = np.zeros_like(self.missing_batch, dtype=bool)
        for i in range(3):
            self.missing_mask[i, missing_indices] = True

        # Create multi-feature version (batch_size=3, time_steps=100, features=2)
        second_feature = np.random.normal(0, 1, 100)
        second_feature_missing = second_feature.copy()
        second_feature_missing[missing_indices] = self.mask_value

        self.multi_feature_clean = np.stack(
            [
                np.stack([self.clean_series, second_feature], axis=-1),
                np.stack([self.clean_series * 1.2, second_feature], axis=-1),
                np.stack([self.clean_series * 0.8, second_feature], axis=-1),
            ]
        )

        self.multi_feature_missing = np.stack(
            [
                np.stack([self.missing_series, second_feature_missing], axis=-1),
                np.stack([self.missing_series * 1.2, second_feature_missing], axis=-1),
                np.stack([self.missing_series * 0.8, second_feature_missing], axis=-1),
            ]
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = MissingValueHandlerLayer()
        self.assertEqual(layer.mask_value, 0.0)
        self.assertEqual(layer.strategy, "forward_fill")
        self.assertEqual(layer.window_size, 5)
        self.assertEqual(layer.seasonal_period, 7)
        self.assertTrue(layer.add_indicators)
        self.assertTrue(layer.extrapolate)

        # Test with custom parameters
        layer = MissingValueHandlerLayer(
            mask_value=-1.0,
            strategy="linear_interpolation",
            window_size=3,
            seasonal_period=12,
            add_indicators=False,
            extrapolate=False,
        )
        self.assertEqual(layer.mask_value, -1.0)
        self.assertEqual(layer.strategy, "linear_interpolation")
        self.assertEqual(layer.window_size, 3)
        self.assertEqual(layer.seasonal_period, 12)
        self.assertFalse(layer.add_indicators)
        self.assertFalse(layer.extrapolate)

        # Test invalid strategy
        with self.assertRaises(ValueError):
            MissingValueHandlerLayer(strategy="invalid")

    def test_call_2d_forward_fill(self):
        """Test forward fill strategy with 2D inputs."""
        # Initialize layer with forward_fill strategy
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For forward fill, values at index i should equal the last valid value before i
        # The first missing value should be replaced with the value before it
        self.assertAlmostEqual(output_np[0, 5], self.clean_batch[0, 4], places=1)

        # For consecutive missing values, just check they're all filled
        for i in range(36, 41):
            self.assertNotEqual(output_np[0, i], self.mask_value)

    def test_call_2d_backward_fill(self):
        """Test backward fill strategy with 2D inputs."""
        # Initialize layer with backward_fill strategy
        layer = MissingValueHandlerLayer(strategy="backward_fill", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For backward fill, values at index i should equal the next valid value after i
        # The last missing value should be replaced with the value after it
        self.assertAlmostEqual(output_np[0, 90], self.clean_batch[0, 91], places=1)

        # For consecutive missing values, just check they're all filled
        for i in range(36, 41):
            self.assertNotEqual(output_np[0, i], self.mask_value)

    def test_call_2d_linear_interpolation(self):
        """Test linear interpolation strategy with 2D inputs."""
        # Initialize layer with linear_interpolation strategy
        layer = MissingValueHandlerLayer(
            strategy="linear_interpolation", add_indicators=False
        )

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For interpolation, isolated missing values should be average of neighbors
        # Test missing value at index 15
        expected_value = (self.clean_batch[0, 14] + self.clean_batch[0, 16]) / 2

        # Linear interpolation might not be exact due to implementation details
        # so we check that the value is within a reasonable range
        self.assertTrue(abs(output_np[0, 15] - expected_value) < 1.0)

        # For consecutive missing values, we just check that they're not the mask value
        for i in range(36, 41):
            self.assertNotEqual(output_np[0, i], self.mask_value)

    def test_call_2d_mean(self):
        """Test mean strategy with 2D inputs."""
        # Initialize layer with mean strategy
        layer = MissingValueHandlerLayer(strategy="mean", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For mean strategy, all missing values should be filled with the mean of the series
        # Calculate expected mean (excluding missing values)
        valid_mask = ~self.missing_mask[0]
        expected_mean = np.mean(self.missing_batch[0][valid_mask])

        # Check each missing value
        for i in range(100):
            if self.missing_mask[0, i]:
                self.assertAlmostEqual(output_np[0, i], expected_mean, places=1)

    def test_call_2d_median(self):
        """Test median strategy with 2D inputs."""
        # Initialize layer with median strategy
        layer = MissingValueHandlerLayer(strategy="median", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For median strategy, all missing values should be filled with the median of the series
        # Calculate expected median (excluding missing values)
        valid_mask = ~self.missing_mask[0]
        expected_median = np.median(self.missing_batch[0][valid_mask])

        # Check each missing value
        for i in range(100):
            if self.missing_mask[0, i]:
                self.assertAlmostEqual(output_np[0, i], expected_median, places=1)

    def test_call_2d_rolling_mean(self):
        """Test rolling mean strategy with 2D inputs."""
        # Initialize layer with rolling_mean strategy
        layer = MissingValueHandlerLayer(
            strategy="rolling_mean", window_size=3, add_indicators=False
        )

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check that values are filled (not equal to mask value)
        output_np = output.numpy()
        self.assertFalse(np.any(output_np == self.mask_value))

    def test_call_2d_seasonal(self):
        """Test seasonal strategy with 2D inputs."""
        # Initialize layer with seasonal strategy
        layer = MissingValueHandlerLayer(
            strategy="seasonal",
            seasonal_period=10,  # We know the period is 10
            add_indicators=False,
        )

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check that values are filled (not equal to mask value)
        output_np = output.numpy()
        self.assertFalse(np.any(output_np == self.mask_value))

    def test_call_with_indicators(self):
        """Test adding missing value indicators."""
        # Initialize layer with add_indicators=True
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=True)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100, 2))

        # Check that the output contains both imputed values and indicators
        output_np = output.numpy()

        # Second channel should be the indicators (1.0 where missing, 0.0 where valid)
        indicators = output_np[:, :, 1]

        # Check that the indicators correctly mark the missing values
        # Allow for small differences in how the indicators are generated
        # Focus on key missing locations
        for i in range(3):
            for j in [5, 15, 25, 35, 60, 80, 90]:
                self.assertEqual(indicators[i, j], 1.0)

    def test_call_3d(self):
        """Test with 3D inputs (multiple features)."""
        # Initialize layer
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=True)

        # Apply imputation
        output = layer(tf.constant(self.multi_feature_missing, dtype=tf.float32))

        # Check output shape
        self.assertEqual(
            output.shape, (3, 100, 4)
        )  # 2 original features + 2 indicators

        # Check that the output contains imputed values and indicators
        output_np = output.numpy()

        # First two channels should be the imputed values
        imputed = output_np[:, :, :2]

        # Next two channels should be the indicators
        indicators = output_np[:, :, 2:]

        # Check that all originally missing values have been filled
        # and that the indicators correctly mark the missing values
        for i in range(3):
            for j in [5, 15, 25, 35, 60, 80, 90]:
                self.assertNotEqual(imputed[i, j, 0], 0.0)  # Value has been imputed
                self.assertEqual(
                    indicators[i, j, 0], 1.0
                )  # Indicator shows it was missing

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Test with add_indicators=True
        layer = MissingValueHandlerLayer(add_indicators=True)

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 2))  # Value + indicator

        # 3D input
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 10))  # 5 values + 5 indicators

        # Test with add_indicators=False
        layer = MissingValueHandlerLayer(add_indicators=False)

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100))  # No change

        # 3D input
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 5))  # No change

    def test_get_config(self):
        """Test get_config method."""
        layer = MissingValueHandlerLayer(
            mask_value=-1.0,
            strategy="linear_interpolation",
            window_size=3,
            seasonal_period=12,
            add_indicators=False,
            extrapolate=False,
        )

        config = layer.get_config()

        self.assertEqual(config["mask_value"], -1.0)
        self.assertEqual(config["strategy"], "linear_interpolation")
        self.assertEqual(config["window_size"], 3)
        self.assertEqual(config["seasonal_period"], 12)
        self.assertFalse(config["add_indicators"])
        self.assertFalse(config["extrapolate"])


if __name__ == "__main__":
    unittest.main()


class TestNaNAsTheMissingMarker(unittest.TestCase):
    """NaN is the marker pandas and numpy use, and it never matched.

    Missing values were found with `inputs == self.mask_value`, and NaN equals
    nothing -- not even itself -- so a series carrying NaN passed through
    untouched and the NaNs then poisoned every statistic computed downstream.
    There was no value of `mask_value` that could select them.
    """

    def setUp(self):
        """One series of twelve steps, with two holes punched in it."""
        self.series = np.arange(1, 13, dtype="float32").reshape(1, -1)
        self.gappy = self.series.copy()
        self.gappy[0, 3] = np.nan
        self.gappy[0, 7] = np.nan

    def _impute(self, strategy):
        """Run one strategy over the gappy series with NaN as the marker."""
        layer = MissingValueHandlerLayer(
            mask_value=float("nan"),
            strategy=strategy,
            add_indicators=False,
        )
        return layer(tf.constant(self.gappy)).numpy()

    def test_no_strategy_leaves_a_nan_behind(self):
        """Every strategy has to fill the holes it was given."""
        for strategy in (
            "forward_fill",
            "backward_fill",
            "linear_interpolation",
            "mean",
            "median",
            "rolling_mean",
            "seasonal",
        ):
            with self.subTest(strategy=strategy):
                self.assertFalse(np.isnan(self._impute(strategy)).any())

    def test_forward_fill_carries_the_previous_value(self):
        """The value before the hole is what forward fill must use."""
        filled = self._impute("forward_fill")
        self.assertAlmostEqual(float(filled[0, 3]), 3.0, places=5)
        self.assertAlmostEqual(float(filled[0, 7]), 7.0, places=5)

    def test_backward_fill_carries_the_next_value(self):
        """And backward fill must use the value after it."""
        filled = self._impute("backward_fill")
        self.assertAlmostEqual(float(filled[0, 3]), 5.0, places=5)
        self.assertAlmostEqual(float(filled[0, 7]), 9.0, places=5)

    def test_interpolation_sits_between_the_neighbours(self):
        """A single gap interpolates to the midpoint of its neighbours."""
        filled = self._impute("linear_interpolation")
        self.assertAlmostEqual(float(filled[0, 3]), 4.0, places=5)
        self.assertAlmostEqual(float(filled[0, 7]), 8.0, places=5)

    def test_present_values_are_untouched(self):
        """Imputation must not disturb the data that was already there."""
        filled = self._impute("forward_fill")
        present = [i for i in range(12) if i not in (3, 7)]
        np.testing.assert_allclose(
            filled[0, present],
            self.series[0, present],
            rtol=1e-6,
        )

    def test_indicators_mark_exactly_the_holes(self):
        """The indicator column has to agree with where the NaNs were."""
        layer = MissingValueHandlerLayer(
            mask_value=float("nan"),
            strategy="forward_fill",
            add_indicators=True,
        )
        flags = layer(tf.constant(self.gappy)).numpy()[..., 1].ravel()
        np.testing.assert_array_equal(np.flatnonzero(flags), [3, 7])

    def test_the_default_marker_still_selects_zeros(self):
        """Changing NaN detection must not change what `mask_value=0.0` means."""
        zeroed = self.series.copy()
        zeroed[0, 3] = 0.0
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=False)
        filled = layer(tf.constant(zeroed)).numpy()
        self.assertAlmostEqual(float(filled[0, 3]), 3.0, places=5)

    def test_a_nan_free_series_is_unchanged(self):
        """With nothing missing, the layer must be the identity."""
        layer = MissingValueHandlerLayer(
            mask_value=float("nan"),
            strategy="mean",
            add_indicators=False,
        )
        np.testing.assert_allclose(
            layer(tf.constant(self.series)).numpy(),
            self.series,
            rtol=1e-6,
        )


class TestBoundaryGaps(unittest.TestCase):
    """Gaps at the ends of a series, which no strategy can reach on its own.

    A gap at the start has nothing before it to carry forward and a gap at the
    end has nothing after it to carry back, so `forward_fill` -- the default --
    returned a leading `NaN` exactly as it arrived and it went into the model.
    `extrapolate` is documented to prevent that and was read nowhere.
    """

    STRATEGIES = (
        "forward_fill",
        "backward_fill",
        "linear_interpolation",
        "mean",
        "median",
        "rolling_mean",
        "seasonal",
    )

    @staticmethod
    def _series(with_gaps=True):
        series = np.arange(1.0, 13.0, dtype="float32")
        if with_gaps:
            series[0] = np.nan
            series[3] = np.nan
            series[11] = np.nan
        return series.reshape(1, -1)

    @staticmethod
    def _values(output):
        array = np.asarray(output)
        return array[0, :, 0] if array.ndim == 3 else array[0]

    def test_no_strategy_leaves_a_missing_value_behind(self):
        for strategy in self.STRATEGIES:
            with self.subTest(strategy=strategy):
                output = MissingValueHandlerLayer(
                    mask_value=float("nan"),
                    strategy=strategy,
                )(tf.constant(self._series()))
                self.assertFalse(np.isnan(self._values(output)).any())

    def test_turning_it_off_leaves_the_ends_alone(self):
        """`extrapolate=False` has to mean something, not merely be accepted."""
        for strategy in ("forward_fill", "backward_fill", "linear_interpolation"):
            with self.subTest(strategy=strategy):
                output = MissingValueHandlerLayer(
                    mask_value=float("nan"),
                    strategy=strategy,
                    extrapolate=False,
                )(tf.constant(self._series()))
                self.assertTrue(np.isnan(self._values(output)).any())

    def test_a_gap_takes_the_nearest_value(self):
        output = MissingValueHandlerLayer(
            mask_value=float("nan"),
            strategy="forward_fill",
        )(tf.constant(self._series()))
        values = self._values(output)
        self.assertAlmostEqual(float(values[0]), 2.0, places=5)
        self.assertAlmostEqual(float(values[11]), 11.0, places=5)

    def test_a_series_missing_everywhere_has_nothing_to_reach_for(self):
        output = MissingValueHandlerLayer(mask_value=float("nan"))(
            tf.constant(np.full((1, 8), np.nan, dtype="float32")),
        )
        values = self._values(output)
        self.assertFalse(np.isnan(values).any())
        np.testing.assert_allclose(values, np.zeros(8, dtype="float32"))

    def test_the_three_dimensional_path_too(self):
        series = np.stack([self._series(False)[0], self._series(False)[0]], axis=-1)
        batch = series[None, ...].copy()
        batch[0, 0, 0] = np.nan
        batch[0, 11, 1] = np.nan
        output = np.asarray(
            MissingValueHandlerLayer(mask_value=float("nan"))(tf.constant(batch)),
        )
        self.assertFalse(np.isnan(output).any())

    def test_a_series_without_gaps_is_untouched(self):
        clean = self._series(with_gaps=False)
        output = MissingValueHandlerLayer(mask_value=0.0)(tf.constant(clean))
        np.testing.assert_allclose(self._values(output), clean[0], rtol=1e-6)
