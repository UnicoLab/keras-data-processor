import tensorflow as tf
import numpy as np
import unittest

from kdp.layers.time_series import TSFreshFeatureLayer


class TestTSFreshFeatureLayer(unittest.TestCase):
    """Test cases for the TSFreshFeatureLayer."""

    def setUp(self):
        # Create sample time series data
        np.random.seed(42)

        # Simple time series: batch_size=3, time_steps=100
        t = np.linspace(0, 4 * np.pi, 100)

        # First example: sin wave with increasing amplitude
        series1 = np.sin(t) * np.linspace(1, 2, 100)

        # Second example: cos wave with noise
        series2 = np.cos(t) + np.random.normal(0, 0.1, size=100)

        # Third example: sawtooth pattern
        series3 = (t % (np.pi / 2)) / (np.pi / 2)

        # Create batch
        self.batch_series = np.stack([series1, series2, series3])

        # Normalize
        self.batch_series = (
            self.batch_series - np.mean(self.batch_series, axis=1, keepdims=True)
        ) / np.std(self.batch_series, axis=1, keepdims=True)

        # Create multi-feature version (batch_size=3, time_steps=100, features=2)
        # Second feature is just random noise
        second_feature = np.random.normal(0, 1, size=(3, 100))
        self.multi_feature_batch = np.stack(
            [self.batch_series, second_feature], axis=-1
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = TSFreshFeatureLayer()
        self.assertEqual(
            layer.features,
            [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "iqr",
                "count_above_mean",
                "count_below_mean",
            ],
        )
        self.assertIsNone(layer.window_size)
        self.assertEqual(layer.stride, 1)
        self.assertTrue(layer.drop_na)
        self.assertFalse(layer.normalize)

        # Test with custom parameters
        layer = TSFreshFeatureLayer(
            features=["mean", "std", "skewness"],
            window_size=10,
            stride=5,
            drop_na=False,
            normalize=True,
        )
        self.assertEqual(layer.features, ["mean", "std", "skewness"])
        self.assertEqual(layer.window_size, 10)
        self.assertEqual(layer.stride, 5)
        self.assertFalse(layer.drop_na)
        self.assertTrue(layer.normalize)

        # Test feature parameter validation
        with self.assertRaises(ValueError):
            TSFreshFeatureLayer(features=["invalid_feature"])

    def test_compute_features(self):
        """Test the _compute_features method."""
        layer = TSFreshFeatureLayer(features=["mean", "std", "min", "max"])

        # Create simple series
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # Calculate features
        features = layer._compute_features(x)

        # Check results
        self.assertEqual(len(features), 4)  # 4 features requested
        self.assertAlmostEqual(features[0], 3.0)  # mean
        self.assertAlmostEqual(features[1], np.std(x))  # std
        self.assertAlmostEqual(features[2], 1.0)  # min
        self.assertAlmostEqual(features[3], 5.0)  # max

    def test_call_2d(self):
        """Test layer call with 2D inputs."""
        # Initialize layer
        layer = TSFreshFeatureLayer(
            features=["mean", "std", "min", "max", "median"], normalize=False
        )

        # Apply layer
        inputs = tf.constant(self.batch_series, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertEqual(output.shape[1], 5)  # Number of features

        # Since we've normalized the data in setup, mean should be close to 0
        # and std close to 1
        output_np = output.numpy()
        self.assertAlmostEqual(output_np[0, 0], 0.0, places=5)  # Mean
        self.assertAlmostEqual(output_np[0, 1], 1.0, places=5)  # Std

    def test_call_3d(self):
        """Test layer call with 3D inputs."""
        # Initialize layer
        layer = TSFreshFeatureLayer(
            features=["mean", "std", "min", "max", "median", "iqr"], normalize=False
        )

        # Apply layer
        inputs = tf.constant(self.multi_feature_batch, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertEqual(output.shape[1], 12)  # 6 features * 2 input features

        # Check some values - first 6 for first feature, next 6 for second feature
        output_np = output.numpy()

        # For the first feature (normalized series)
        self.assertAlmostEqual(output_np[0, 0], 0.0, places=5)  # Mean of first feature
        self.assertAlmostEqual(output_np[0, 1], 1.0, places=5)  # Std of first feature

        # For the second feature (random noise), values will vary but should be within range
        # Just check that they're reasonable
        self.assertTrue(-3.0 <= output_np[0, 6] <= 3.0)  # Mean of second feature
        self.assertTrue(0.5 <= output_np[0, 7] <= 1.5)  # Std of second feature

    def test_windowed_features(self):
        """Test extracting features using windows."""
        # Initialize layer with window
        layer = TSFreshFeatureLayer(
            features=["mean", "std"], window_size=20, stride=1, normalize=False
        )

        # Apply layer
        inputs = tf.constant(self.batch_series, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape - now we have features for each window
        n_windows = self.batch_series.shape[1] - layer.window_size + 1
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertEqual(output.shape[1], n_windows)  # Number of windows
        self.assertEqual(output.shape[2], 2)  # Number of features

        # Check values - just validate shape and some boundary checks
        output_np = output.numpy()

        # Windowed features should have reasonable values
        for w in range(n_windows):
            self.assertTrue(-2.0 <= output_np[0, w, 0] <= 2.0)  # Mean
            self.assertTrue(0.0 <= output_np[0, w, 1] <= 2.0)  # Std

    def test_statistical_features(self):
        """Test extracting statistical features."""
        # Initialize layer with statistical features
        layer = TSFreshFeatureLayer(
            features=["skewness", "kurtosis", "abs_energy"], normalize=False
        )

        # Apply layer
        inputs = tf.constant(self.batch_series, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertEqual(output.shape[1], 3)  # Number of features

        # For normalized data, these should be reasonable values
        output_np = output.numpy()
        self.assertTrue(-5.0 <= output_np[0, 0] <= 5.0)  # Skewness
        self.assertTrue(-5.0 <= output_np[0, 1] <= 5.0)  # Kurtosis
        self.assertTrue(0.0 <= output_np[0, 2] <= 100.0)  # Abs energy

    def test_quantile_features(self):
        """Test extracting quantile features."""
        # Initialize layer with quantile features
        layer = TSFreshFeatureLayer(
            features=["quantile_05", "quantile_95"], normalize=False
        )

        # Apply layer
        inputs = tf.constant(self.batch_series, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertEqual(output.shape[1], 2)  # Number of features

        # For normalized data with mean 0 and std 1:
        # 5% quantile should be negative, 95% quantile should be positive
        output_np = output.numpy()
        self.assertTrue(output_np[0, 0] < 0)  # 5% quantile
        self.assertTrue(output_np[0, 1] > 0)  # 95% quantile

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Test without window
        layer1 = TSFreshFeatureLayer(features=["mean", "std", "min", "max"])

        # For 2D input (batch_size, time_steps)
        input_shape = (32, 100)
        output_shape = layer1.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 4))  # (batch_size, n_features)

        # For 3D input (batch_size, time_steps, n_features)
        input_shape = (32, 100, 3)
        output_shape = layer1.compute_output_shape(input_shape)
        self.assertEqual(
            output_shape, (32, 12)
        )  # (batch_size, n_features * input_features)

        # Test with window
        layer2 = TSFreshFeatureLayer(
            features=["mean", "std", "min", "max"], window_size=20, stride=1
        )

        # For 2D input (batch_size, time_steps)
        input_shape = (32, 100)
        output_shape = layer2.compute_output_shape(input_shape)
        n_windows = 100 - 20 + 1
        self.assertEqual(
            output_shape, (32, n_windows, 4)
        )  # (batch_size, n_windows, n_features)

    def test_get_config(self):
        """Test get_config method."""
        layer = TSFreshFeatureLayer(
            features=["mean", "std", "skewness"],
            window_size=15,
            stride=5,
            drop_na=False,
            normalize=True,
        )

        config = layer.get_config()

        self.assertEqual(config["features"], ["mean", "std", "skewness"])
        self.assertEqual(config["window_size"], 15)
        self.assertEqual(config["stride"], 5)
        self.assertFalse(config["drop_na"])
        self.assertTrue(config["normalize"])

    def test_a_window_longer_than_the_series(self):
        """The extraction clamps the window to the series; the shape did not.

        `n_windows` came out negative and `set_shape` refused it, so a
        `window_size` above the series length raised instead of running.
        """
        data = tf.constant(np.arange(1, 41, dtype="float32").reshape(4, 10))
        for window_size in (3, 10, 15, 40):
            layer = TSFreshFeatureLayer(
                features=["mean", "std"],
                window_size=window_size,
            )
            output = layer(data)
            self.assertEqual(
                tuple(output.shape),
                tuple(layer.compute_output_shape((4, 10))),
            )
            self.assertGreaterEqual(int(output.shape[1]), 1)
            self.assertTrue(np.isfinite(np.asarray(output)).all())

    def test_the_declared_shape_matches_the_values(self):
        """Windowed and whole-series output, checked against what comes back."""
        data = tf.constant(np.arange(1, 61, dtype="float32").reshape(4, 15))
        for window_size, stride in ((None, 1), (5, 1), (5, 3), (4, 2)):
            layer = TSFreshFeatureLayer(
                features=["mean", "std", "min"],
                window_size=window_size,
                stride=stride,
            )
            output = layer(data)
            self.assertEqual(
                tuple(output.shape),
                tuple(layer.compute_output_shape((4, 15))),
            )


if __name__ == "__main__":
    unittest.main()
