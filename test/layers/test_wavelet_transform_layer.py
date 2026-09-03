import tensorflow as tf
import numpy as np
import unittest

import keras

from kdp.layers.time_series import WaveletTransformLayer


class TestWaveletTransformLayer(unittest.TestCase):
    """Test cases for the WaveletTransformLayer."""

    def setUp(self):
        # Create sample time series data
        np.random.seed(42)

        # Create a simple sine wave with noise
        t = np.linspace(0, 4 * np.pi, 256)
        signal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.25 * np.sin(3 * t)
        noise = np.random.normal(0, 0.1, size=len(t))
        series = signal + noise

        # Normalize
        series = (series - np.mean(series)) / np.std(series)

        # Create a batch (batch_size=3)
        self.batch_series = np.stack(
            [
                series,
                series * 1.2 + 0.5,  # Scaled and shifted
                series * 0.8 - 0.3,  # Scaled and shifted
            ]
        )

        # Create multi-feature version (batch_size=3, time_steps=256, features=2)
        second_feature = np.random.normal(0, 1, size=len(t))
        multi_feature = np.stack([series, second_feature], axis=-1)
        self.multi_feature_batch = np.stack(
            [multi_feature, multi_feature, multi_feature]
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = WaveletTransformLayer()
        self.assertEqual(layer.levels, 3)
        self.assertEqual(layer.keep_levels, "all")
        self.assertEqual(layer.window_sizes, None)
        self.assertTrue(layer.flatten_output)
        self.assertTrue(layer.drop_na)

        # Test with custom parameters
        layer = WaveletTransformLayer(
            levels=4,
            keep_levels="approx",
            window_sizes=[2, 4, 8, 16],
            flatten_output=False,
            drop_na=False,
        )
        self.assertEqual(layer.levels, 4)
        self.assertEqual(layer.keep_levels, "approx")
        self.assertEqual(layer.window_sizes, [2, 4, 8, 16])
        self.assertFalse(layer.flatten_output)
        self.assertFalse(layer.drop_na)

        # Test invalid keep_levels
        with self.assertRaises(ValueError):
            WaveletTransformLayer(keep_levels="invalid_option")

    def test_call_2d(self):
        """Test layer call with 2D inputs."""
        # Initialize layer
        layer = WaveletTransformLayer(
            levels=3, window_sizes=[4, 8, 16], flatten_output=True, drop_na=False
        )

        # Apply layer
        inputs = tf.constant(self.batch_series, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertTrue(output.shape[1] > 0)  # Features

        # Check values aren't all zeros
        output_np = output.numpy()
        self.assertFalse(np.allclose(output_np, 0))

    def test_call_3d(self):
        """Test layer call with 3D inputs (multiple features)."""
        # Initialize layer
        layer = WaveletTransformLayer(
            levels=2, window_sizes=[4, 8], flatten_output=True, drop_na=False
        )

        # Apply layer
        inputs = tf.constant(self.multi_feature_batch, dtype=tf.float32)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape[0], 3)  # Batch size
        self.assertTrue(output.shape[1] > 0)  # Features

        # Check values aren't all zeros
        output_np = output.numpy()
        self.assertFalse(np.allclose(output_np, 0))

    def test_keep_levels_options(self):
        """Test different options for keep_levels."""
        inputs = tf.constant(self.batch_series, dtype=tf.float32)

        # Test 'all' option
        layer_all = WaveletTransformLayer(
            levels=3,
            window_sizes=[4, 8, 16],
            keep_levels="all",
            flatten_output=True,
            drop_na=False,
        )
        output_all = layer_all(inputs)

        # Test 'approx' option
        layer_approx = WaveletTransformLayer(
            levels=3,
            window_sizes=[4, 8, 16],
            keep_levels="approx",
            flatten_output=True,
            drop_na=False,
        )
        output_approx = layer_approx(inputs)

        # Test specific levels
        layer_specific = WaveletTransformLayer(
            levels=3,
            window_sizes=[4, 8, 16],
            keep_levels=[0, 1],
            flatten_output=True,
            drop_na=False,
        )
        output_specific = layer_specific(inputs)

        # Check output shapes
        self.assertTrue(output_all.shape[1] > output_approx.shape[1])
        self.assertTrue(output_all.shape[1] > output_specific.shape[1])

    def test_moving_average(self):
        """Test the moving average function."""
        layer = WaveletTransformLayer()

        # Test with simple sequence
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3

        ma = layer._moving_average(series, window_size)

        # Expected output: [2, 3, 4, 5, 6, 7, 8, 9]
        # (1+2+3)/3, (2+3+4)/3, ..., (8+9+10)/3
        expected = np.array([2, 3, 4, 5, 6, 7, 8, 9])

        np.testing.assert_array_almost_equal(ma, expected)

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Initialize layer
        layer = WaveletTransformLayer(
            levels=3, window_sizes=[4, 8, 16], flatten_output=True, drop_na=False
        )

        # 2D input
        input_shape = (32, 256)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape[0], 32)  # Batch size
        self.assertTrue(output_shape[1] > 0)  # Features

        # 3D input
        input_shape = (32, 256, 2)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape[0], 32)  # Batch size
        self.assertTrue(output_shape[1] > 0)  # Features

    def test_get_config(self):
        """Test get_config method."""
        layer = WaveletTransformLayer(
            levels=4,
            keep_levels="approx",
            window_sizes=[2, 4, 8, 16],
            flatten_output=False,
            drop_na=False,
        )

        config = layer.get_config()

        self.assertEqual(config["levels"], 4)
        self.assertEqual(config["keep_levels"], "approx")
        self.assertEqual(config["window_sizes"], [2, 4, 8, 16])
        self.assertFalse(config["flatten_output"])
        self.assertFalse(config["drop_na"])

    def test_unflattened_output_carries_the_same_coefficients(self):
        """`flatten_output=False` returned `np.zeros(...)` -- every coefficient
        thrown away -- and then declared a rank-2 shape for a rank-3 tensor, so
        `set_shape` raised and the option could not run on any input at all.
        """
        rng = np.random.default_rng(11)
        for shape in ((1, 32), (1, 32, 1), (2, 32, 3)):
            data = tf.constant(rng.normal(0, 1, shape).astype("float32"))

            flat = np.asarray(
                WaveletTransformLayer(levels=2, flatten_output=True)(data),
            )
            layer = WaveletTransformLayer(levels=2, flatten_output=False)
            deep = np.asarray(layer(data))

            self.assertFalse(np.all(deep == 0))
            self.assertEqual(len(deep.shape), 3)
            np.testing.assert_allclose(deep.reshape(flat.shape), flat)
            self.assertEqual(
                tuple(deep.shape),
                tuple(layer.compute_output_shape(shape)),
            )

    def test_unflattened_output_builds_in_a_graph(self):
        """The declared shape has to survive functional model construction."""
        inputs = keras.Input(shape=(32, 3))
        model = keras.Model(
            inputs,
            WaveletTransformLayer(levels=2, flatten_output=False)(inputs),
        )
        rng = np.random.default_rng(12)
        result = model(tf.constant(rng.normal(0, 1, (2, 32, 3)).astype("float32")))
        self.assertEqual(tuple(result.shape)[1:], model.output_shape[1:])


if __name__ == "__main__":
    unittest.main()
