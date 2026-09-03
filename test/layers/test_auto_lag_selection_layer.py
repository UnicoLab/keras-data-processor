import tensorflow as tf
import numpy as np
import unittest

from kdp.layers.time_series import AutoLagSelectionLayer


class TestAutoLagSelectionLayer(unittest.TestCase):
    """Test cases for the AutoLagSelectionLayer."""

    def setUp(self):
        # Create sample time series data with known autocorrelation pattern
        # Generate time series where lag 3, 7, and 10 are important
        np.random.seed(42)

        # Create base series with noise
        base = np.random.normal(0, 1, 200)

        # Add lag dependencies
        lag_series = base.copy()
        for i in range(10, 200):
            # Add strong dependency on lag 3
            lag_series[i] += 0.7 * lag_series[i - 3]
            # Add medium dependency on lag 7
            lag_series[i] += 0.5 * lag_series[i - 7]
            # Add weak dependency on lag 10
            lag_series[i] += 0.3 * lag_series[i - 10]

        # Normalize
        lag_series = (lag_series - np.mean(lag_series)) / np.std(lag_series)

        # Create a batch (batch_size=3)
        self.batch_series = np.stack(
            [lag_series, lag_series * 1.2 + 0.5, lag_series * 0.8 - 1.0]
        )

        # Create multi-feature version (batch_size=3, time_steps=200, features=2)
        second_feature = np.random.normal(0, 1, 200)
        multi_feature = np.stack([lag_series, second_feature], axis=-1)
        self.multi_feature_batch = np.stack(
            [multi_feature, multi_feature, multi_feature]
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = AutoLagSelectionLayer()
        self.assertEqual(layer.max_lag, 30)
        self.assertEqual(layer.n_lags, 5)
        self.assertEqual(layer.threshold, 0.2)
        self.assertEqual(layer.method, "top_k")
        self.assertTrue(layer.drop_na)
        self.assertEqual(layer.fill_value, 0.0)
        self.assertTrue(layer.keep_original)

        # Test with custom parameters
        layer = AutoLagSelectionLayer(
            max_lag=15,
            n_lags=3,
            threshold=0.3,
            method="threshold",
            drop_na=False,
            fill_value=-1.0,
            keep_original=False,
        )
        self.assertEqual(layer.max_lag, 15)
        self.assertEqual(layer.n_lags, 3)
        self.assertEqual(layer.threshold, 0.3)
        self.assertEqual(layer.method, "threshold")
        self.assertFalse(layer.drop_na)
        self.assertEqual(layer.fill_value, -1.0)
        self.assertFalse(layer.keep_original)

        # Test invalid method
        with self.assertRaises(ValueError):
            AutoLagSelectionLayer(method="invalid")

    def test_compute_autocorrelation(self):
        """Test autocorrelation computation."""
        # Initialize layer
        layer = AutoLagSelectionLayer(max_lag=15)

        # Convert data to TensorFlow tensor
        data_tensor = tf.constant(self.batch_series, dtype=tf.float32)

        # Compute autocorrelation
        acf = layer._compute_autocorrelation(data_tensor)

        # Check shape
        self.assertEqual(acf.shape, (3, 16))  # batch_size, max_lag+1

        # Check specific values
        acf_np = acf.numpy()

        # Lag 0 autocorrelation should be 1
        np.testing.assert_allclose(acf_np[:, 0], 1.0, rtol=1e-5)

        # Known lags should have higher autocorrelation
        # Lag 3 should have higher autocorrelation than its neighbors
        self.assertGreater(acf_np[0, 3], acf_np[0, 2])
        self.assertGreater(acf_np[0, 3], acf_np[0, 4])

        # Lag 7 should have higher autocorrelation than its neighbors
        self.assertGreater(acf_np[0, 7], acf_np[0, 6])
        self.assertGreater(acf_np[0, 7], acf_np[0, 8])

    def test_select_lags_top_k(self):
        """Test lag selection with top_k method."""
        # Initialize layer with top_k method
        layer = AutoLagSelectionLayer(max_lag=15, n_lags=3, method="top_k")

        # Create sample autocorrelation function with known high values
        # High autocorrelation at lags 3, 7, 10
        acf = np.zeros((2, 16))
        acf[:, 0] = 1.0  # Lag 0
        acf[:, 3] = 0.7  # Lag 3
        acf[:, 7] = 0.5  # Lag 7
        acf[:, 10] = 0.3  # Lag 10
        acf_tensor = tf.constant(acf, dtype=tf.float32)

        # Select lags
        selected_lags = layer._select_lags(acf_tensor)

        # Check shape
        self.assertEqual(selected_lags.shape, (3,))  # n_lags

        # Convert to numpy and sort for comparison
        selected_lags_np = sorted(selected_lags.numpy())

        # Check that the correct lags were selected (3, 7, 10)
        self.assertListEqual(selected_lags_np, [3, 7, 10])

    def test_select_lags_threshold(self):
        """Test lag selection with threshold method."""
        # Initialize layer with threshold method
        layer = AutoLagSelectionLayer(max_lag=15, threshold=0.4, method="threshold")

        # Create sample autocorrelation function with known high values
        acf = np.zeros((2, 16))
        acf[:, 0] = 1.0  # Lag 0
        acf[:, 3] = 0.7  # Lag 3
        acf[:, 7] = 0.5  # Lag 7
        acf[:, 10] = 0.3  # Lag 10
        acf_tensor = tf.constant(acf, dtype=tf.float32)

        # Select lags
        selected_lags = layer._select_lags(acf_tensor)

        # Convert to numpy and sort for comparison
        selected_lags_np = sorted(selected_lags.numpy())

        # Check that lags with autocorrelation > threshold were selected (3, 7)
        self.assertListEqual(selected_lags_np, [3, 7])

    def test_call_2d(self):
        """A 2-D batch keeps its original series alongside the lag columns.

        This was skipped as needing "exact lag feature values that are
        difficult to match". Both of its assertions hold: the output is
        (series, timesteps, 1 + n_lags) and column 0 is the input untouched.
        """
        # Initialize layer
        layer = AutoLagSelectionLayer(
            max_lag=15, n_lags=3, method="top_k", keep_original=True, drop_na=False
        )

        # Apply layer
        output = layer(tf.constant(self.batch_series, dtype=tf.float32))

        # Check output shape
        # With keep_original=True, we get 4 features: original + 3 lags
        self.assertEqual(output.shape, (3, 200, 4))

        # Check that the output contains lag features
        # Original values should be in the first feature
        original = output[:, :, 0].numpy()

        # Verify original values have been preserved
        # Check a few random indices instead of the whole array
        for idx in [0, 10, 50, 100, 150]:
            self.assertAlmostEqual(
                original[0, idx], self.batch_series[0, idx], places=2
            )

        # Each lag column is the series shifted by the lag the layer *chose*,
        # not by its column position: the previous expectation assumed lags
        # 1, 2, 3 while the layer selects them from the autocorrelation, which
        # is why the values never matched.
        selected = [int(lag) for lag in np.asarray(layer.selected_lags)]
        self.assertEqual(len(selected), 3)

        for column, lag in enumerate(selected, start=1):
            lag_feature = output[0, :, column].numpy()
            # The first `lag` positions have no history, so they are padded.
            self.assertEqual(lag_feature[0], 0.0)
            for idx in [20, 50, 100, 150]:
                self.assertAlmostEqual(
                    lag_feature[idx],
                    self.batch_series[0, idx - lag],
                    places=2,
                    msg=f"column {column} should hold lag {lag}",
                )

    def test_call_3d(self):
        """Test layer call with 3D inputs (multiple features)."""
        # Initialize layer
        layer = AutoLagSelectionLayer(
            max_lag=15, n_lags=3, method="top_k", keep_original=True, drop_na=False
        )

        # Apply layer
        output = layer(tf.constant(self.multi_feature_batch, dtype=tf.float32))

        # Check output shape
        # With keep_original=True, we get original features + lag features
        # 2 original features + (2 features * 3 lags)
        self.assertEqual(output.shape, (3, 200, 8))

        # Check that the output contains the original features
        original_features = output[:, :, :2].numpy()
        np.testing.assert_allclose(
            original_features, self.multi_feature_batch, rtol=1e-5
        )

    def test_drop_na(self):
        """`drop_na=True` drops the rows the largest lag consumes.

        This was skipped as "requires a negative batch dimension". It did: the
        declared shape used `rows - max_lag` unclamped while the array itself
        was allocated with `max(1, rows - max_lag)`, so `set_shape` was handed a
        negative dimension whenever there were fewer rows than the lag -- which
        is every batch smaller than the lag, including the default
        configuration on a single series.
        """
        layer = AutoLagSelectionLayer(
            max_lag=15, n_lags=3, method="top_k", keep_original=True, drop_na=True
        )
        layer.selected_lags = tf.constant([3, 7, 10], dtype=tf.int32)

        output = layer(tf.constant(self.batch_series, dtype=tf.float32))

        rows = self.batch_series.shape[0]
        self.assertEqual(output.shape[0], max(1, rows - 10))

    def test_defaults_build(self):
        """The default configuration raised outright before the clamp."""
        series = np.sin(np.arange(200) * 2 * np.pi / 7).astype("float32")
        output = AutoLagSelectionLayer()(tf.constant(series.reshape(1, 200, 1)))
        self.assertTrue(np.isfinite(np.asarray(output)).all())

    def test_drop_na_never_declares_a_negative_dimension(self):
        """A single-row batch is the case that used to raise."""
        series = np.sin(np.arange(120) * 2 * np.pi / 7).astype("float32")
        for max_lag in (5, 10, 20):
            layer = AutoLagSelectionLayer(max_lag=max_lag, n_lags=2, drop_na=True)
            output = layer(tf.constant(series.reshape(1, 120, 1)))
            self.assertGreaterEqual(output.shape[0], 1)

    def test_every_channel_votes_on_the_selected_lags(self):
        """Lag selection used to read channel 0 and discard the rest.

        A noisy first channel then picked the lags for every other channel, so
        simply reordering the columns of the same data changed the output. The
        autocorrelation is now averaged over the channels, which makes the
        choice independent of the column order.
        """
        steps = np.arange(200)
        noise = np.random.default_rng(0).normal(0, 1, 200)
        wave = np.sin(2 * np.pi * steps / 10.0)

        chosen = []
        for stacked in (
            np.stack([noise, wave], axis=-1),
            np.stack([wave, noise], axis=-1),
        ):
            layer = AutoLagSelectionLayer(max_lag=30, n_lags=3, method="top_k")
            layer(tf.constant(stacked[None, ...].astype("float32")), training=True)
            chosen.append(sorted(layer.selected_lags.numpy().ravel().tolist()))

        self.assertEqual(chosen[0], chosen[1])

    def test_a_lone_channel_still_drives_its_own_lags(self):
        """Averaging across channels must not disturb the single-series case."""
        wave = np.sin(2 * np.pi * np.arange(200) / 10.0).astype("float32")

        two_d = AutoLagSelectionLayer(max_lag=30, n_lags=3, method="top_k")
        two_d(tf.constant(wave[None, :]), training=True)

        three_d = AutoLagSelectionLayer(max_lag=30, n_lags=3, method="top_k")
        three_d(tf.constant(wave.reshape(1, 200, 1)), training=True)

        self.assertEqual(
            sorted(two_d.selected_lags.numpy().ravel().tolist()),
            sorted(three_d.selected_lags.numpy().ravel().tolist()),
        )

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Initialize layer with keep_original=True, drop_na=False
        layer = AutoLagSelectionLayer(
            max_lag=15, n_lags=3, keep_original=True, drop_na=False
        )

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 4))  # original + 3 lags

        # 3D input
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 20))  # 5 original + (5 * 3 lags)

        # Test with keep_original=False, drop_na=True
        layer = AutoLagSelectionLayer(
            max_lag=15, n_lags=3, keep_original=False, drop_na=True
        )

        # 2D input with drop_na
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(
            output_shape, (17, 100, 3)
        )  # Lose max_lag rows, 3 lag features

        # 3D input with drop_na
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(
            output_shape, (17, 100, 15)
        )  # Lose max_lag rows, 5 features * 3 lags

    def test_get_config(self):
        """Test get_config method."""
        layer = AutoLagSelectionLayer(
            max_lag=15,
            n_lags=3,
            threshold=0.3,
            method="threshold",
            drop_na=False,
            fill_value=-1.0,
            keep_original=False,
        )

        config = layer.get_config()

        self.assertEqual(config["max_lag"], 15)
        self.assertEqual(config["n_lags"], 3)
        self.assertEqual(config["threshold"], 0.3)
        self.assertEqual(config["method"], "threshold")
        self.assertFalse(config["drop_na"])
        self.assertEqual(config["fill_value"], -1.0)
        self.assertFalse(config["keep_original"])


if __name__ == "__main__":
    unittest.main()
