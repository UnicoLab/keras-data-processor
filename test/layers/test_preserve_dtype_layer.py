import keras
import unittest
import numpy as np
import tensorflow as tf
import pytest

from kdp.layers.preserve_dtype import PreserveDtypeLayer


@pytest.mark.layers
@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.micro
class TestPreserveDtypeLayer(unittest.TestCase):
    """Test cases for PreserveDtypeLayer."""

    def test_preserve_original_dtype(self):
        """Test that the layer preserves original dtype when target_dtype is None."""
        layer = PreserveDtypeLayer()

        # Test with string input
        string_input = tf.constant(["hello", "world"])
        output = layer(string_input)
        self.assertEqual(output.dtype, tf.string)
        np.testing.assert_array_equal(output.numpy(), string_input.numpy())

        # Test with int input
        int_input = tf.constant([1, 2, 3])
        output = layer(int_input)
        self.assertEqual(output.dtype, tf.int32)
        np.testing.assert_array_equal(output.numpy(), int_input.numpy())

        # Test with float input
        float_input = tf.constant([1.5, 2.7, 3.9])
        output = layer(float_input)
        self.assertEqual(output.dtype, tf.float32)
        np.testing.assert_array_equal(output.numpy(), float_input.numpy())

    def test_cast_to_target_dtype(self):
        """Test that the layer casts to target_dtype when specified."""
        # Test casting float to int
        layer = PreserveDtypeLayer(target_dtype=tf.int32)
        float_input = tf.constant([1.5, 2.7, 3.9])
        output = layer(float_input)
        self.assertEqual(output.dtype, tf.int32)
        np.testing.assert_array_equal(output.numpy(), [1, 2, 3])

        # Test casting int to float
        layer = PreserveDtypeLayer(target_dtype=tf.float32)
        int_input = tf.constant([1, 2, 3])
        output = layer(int_input)
        self.assertEqual(output.dtype, tf.float32)
        np.testing.assert_array_equal(output.numpy(), [1.0, 2.0, 3.0])

        # Test casting to float64
        layer = PreserveDtypeLayer(target_dtype=tf.float64)
        float_input = tf.constant([1.5, 2.7, 3.9])
        output = layer(float_input)
        self.assertEqual(output.dtype, tf.float64)
        np.testing.assert_array_almost_equal(output.numpy(), [1.5, 2.7, 3.9])

    def test_batch_processing(self):
        """Test that the layer works correctly with batched inputs."""
        layer = PreserveDtypeLayer()

        # Test with 2D input
        batch_input = tf.constant([[1, 2], [3, 4], [5, 6]])
        output = layer(batch_input)
        self.assertEqual(output.dtype, tf.int32)
        self.assertEqual(output.shape, (3, 2))
        np.testing.assert_array_equal(output.numpy(), batch_input.numpy())

        # Test with 3D input
        layer = PreserveDtypeLayer(target_dtype=tf.float32)
        batch_input = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        output = layer(batch_input)
        self.assertEqual(output.dtype, tf.float32)
        self.assertEqual(output.shape, (2, 2, 2))

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        layer = PreserveDtypeLayer(target_dtype=tf.int32, name="test_layer")

        # Serialize
        config = layer.get_config()

        # Deserialize
        new_layer = PreserveDtypeLayer.from_config(config)

        # Test that they behave the same
        input_tensor = tf.constant([1.5, 2.7, 3.9])
        original_output = layer(input_tensor)
        new_output = new_layer(input_tensor)

        self.assertEqual(original_output.dtype, new_output.dtype)
        np.testing.assert_array_equal(original_output.numpy(), new_output.numpy())
        self.assertEqual(layer.name, new_layer.name)
        self.assertEqual(layer.target_dtype, new_layer.target_dtype)

    def test_model_integration(self):
        """Test that the layer works correctly within a Keras model."""
        layer = PreserveDtypeLayer(target_dtype=tf.float32)

        # Create a simple model
        inputs = keras.Input(shape=(3,), dtype=tf.int32)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test the model
        test_input = tf.constant([[1, 2, 3], [4, 5, 6]])
        output = model(test_input)

        self.assertEqual(output.dtype, tf.float32)
        self.assertEqual(output.shape, (2, 3))
        np.testing.assert_array_equal(
            output.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )


if __name__ == "__main__":
    unittest.main()
