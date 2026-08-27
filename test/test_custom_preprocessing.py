import keras
import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
import sys
from pathlib import Path
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline

# Add the project root to the Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration for testing
tf.random.set_seed(42)
np.random.seed(42)


# Custom scaling layer for testing
class CustomScalingLayer(keras.layers.Layer):
    def __init__(self, scaling_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        # Add __name__ attribute for compatibility with PreprocessingModel
        self.__name__ = "CustomScalingLayer"

    def call(self, inputs):
        return inputs * self.scaling_factor

    def get_config(self):
        config = super().get_config()
        config.update({"scaling_factor": self.scaling_factor})
        return config


# Custom normalization layer for testing
class CustomNormalizationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add __name__ attribute for compatibility with PreprocessingModel
        self.__name__ = "CustomNormalizationLayer"

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0)
        std = tf.math.reduce_std(inputs, axis=0)
        return (inputs - mean) / (std + 1e-5)

    def get_config(self):
        return super().get_config()


class DynamicPreprocessingPipelineTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files if needed
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_basic_pipeline(self):
        """Test basic DynamicPreprocessingPipeline with custom layers."""
        # Create custom layers with unique names
        scaling_layer = CustomScalingLayer(scaling_factor=2.0, name="scaling")
        norm_layer = CustomNormalizationLayer(name="norm")

        # Create a simple dataset with the required input keys for each layer
        data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices({"scaling": data, "norm": data})

        # Create pipeline with custom layers
        pipeline = DynamicPreprocessingPipeline([scaling_layer, norm_layer])

        # Process dataset through pipeline
        processed_dataset = pipeline.process(dataset)

        # Get first element of processed dataset
        for element in processed_dataset.take(1):
            # Verify that both layers were applied
            self.assertIn("scaling", element)
            self.assertIn("norm", element)

            # Verify the output values
            # For scaling layer, input 1.0 should be scaled to 2.0
            self.assertAlmostEqual(element["scaling"].numpy()[0], 2.0, places=5)

    def test_pipeline_with_dependencies(self):
        """Test DynamicPreprocessingPipeline with layer dependencies."""
        # Create a pipeline with a sequence of layers where second depends on first
        # First layer doubles the input
        scaling_layer = CustomScalingLayer(scaling_factor=2.0, name="scaling")
        # Second layer normalizes the output of the first layer
        norm_layer = CustomNormalizationLayer(name="norm")

        # Create a simple dataset
        data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices({"scaling": data})

        # Create pipeline with layers in sequence
        pipeline = DynamicPreprocessingPipeline([scaling_layer, norm_layer])

        # Process dataset through pipeline
        processed_dataset = pipeline.process(dataset)

        # Verify that both layers were applied correctly
        for element in processed_dataset.take(1):
            self.assertIn("scaling", element)
            self.assertIn("norm", element)


if __name__ == "__main__":
    unittest.main()
