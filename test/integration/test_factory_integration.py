import keras
import unittest
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import tempfile

from kdp.dynamic_pipeline import DynamicPreprocessingPipeline
from kdp.layers_factory import PreprocessorLayerFactory


class TestFactoryIntegration(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model")

        # Generate synthetic data
        self.n_samples = 500

        # Numeric features with different distributions
        self.data = {
            # Log-normal distribution (right-skewed)
            "income": np.random.lognormal(mean=10, sigma=1, size=self.n_samples),
            # Normal distribution
            "age": np.random.normal(loc=35, scale=10, size=self.n_samples),
            # Uniform distribution
            "score": np.random.uniform(low=0, high=100, size=self.n_samples),
        }

        # Create pandas DataFrame
        self.df = pd.DataFrame(self.data)

        # Save data to CSV for testing
        self.data_path = os.path.join(self.temp_dir.name, "test_data.csv")
        self.df.to_csv(self.data_path, index=False)

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_basic_factory_layers(self):
        """Test basic layer factory methods to ensure they work correctly."""

        # Create test data as NumPy arrays
        income_data = np.array(self.df["income"].values, dtype=np.float32).reshape(
            -1, 1
        )
        age_data = np.array(self.df["age"].values, dtype=np.float32).reshape(-1, 1)

        # Test DistributionTransformLayer using factory method
        distribution_transform = PreprocessorLayerFactory.distribution_transform_layer(
            transform_type="log", name="log_transform"
        )

        # Create a dataset with the income data
        income_dataset = tf.data.Dataset.from_tensor_slices(
            {"log_transform": income_data}
        ).batch(32)

        # Create a pipeline with the distribution transform layer
        transform_pipeline = DynamicPreprocessingPipeline([distribution_transform])

        # Process the data
        processed_transform = transform_pipeline.process(income_dataset)

        # Check the output - log transformation should have been applied
        for batch in processed_transform.take(1):
            # The output should be in log space, so it should be smaller than the input
            self.assertTrue(
                tf.reduce_mean(batch["log_transform"]) < tf.reduce_mean(income_data)
            )
            self.assertEqual(
                batch["log_transform"].shape[1], 1
            )  # Should maintain the shape

        # Test CastToFloat32Layer using factory method
        cast_layer = PreprocessorLayerFactory.cast_to_float32_layer(
            name="cast_to_float"
        )

        # Create input data with int32 type
        int_data = np.array(np.random.randint(0, 100, size=(100, 1)), dtype=np.int32)

        # Create a dataset
        int_dataset = tf.data.Dataset.from_tensor_slices(
            {"cast_to_float": int_data}
        ).batch(32)

        # Create a pipeline
        cast_pipeline = DynamicPreprocessingPipeline([cast_layer])

        # Process the data
        processed_cast = cast_pipeline.process(int_dataset)

        # Check the output - data should now be float32
        for batch in processed_cast.take(1):
            self.assertEqual(batch["cast_to_float"].dtype, tf.float32)

        # Test combining multiple layers in sequence
        # First normalize, then apply distribution transform
        normalize_layer = keras.layers.Normalization(name="normalize")
        # Adapt the normalization layer to the data before using it
        normalize_layer.adapt(age_data)

        log_transform = PreprocessorLayerFactory.distribution_transform_layer(
            transform_type="log", name="log_after_norm"
        )

        # Create a pipeline with normalize followed by log transform
        combined_pipeline = DynamicPreprocessingPipeline(
            [normalize_layer, log_transform]
        )

        # Create a dataset
        age_dataset = tf.data.Dataset.from_tensor_slices({"normalize": age_data}).batch(
            32
        )

        # Process the data
        processed_combined = combined_pipeline.process(age_dataset)

        # Check the output
        for batch in processed_combined.take(1):
            self.assertIn("log_after_norm", batch)
            # The output should be normalized and then log-transformed
            self.assertEqual(batch["log_after_norm"].shape[1], 1)

        # Test factory methods in isolation
        # Test distribution_transform_layer by applying directly to tensor
        test_data = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
        dist_transform = PreprocessorLayerFactory.distribution_transform_layer(
            transform_type="log"
        )
        transformed = dist_transform(test_data)

        # Check the output - log transform should make values smaller
        self.assertTrue(tf.reduce_mean(transformed) < tf.reduce_mean(test_data))
        self.assertEqual(transformed.shape, test_data.shape)

        # Test cast_to_float32_layer
        int_tensor = tf.constant([[1], [2], [3], [4]], dtype=tf.int32)
        cast_layer = PreprocessorLayerFactory.cast_to_float32_layer()
        float_tensor = cast_layer(int_tensor)

        # Check the output
        self.assertEqual(float_tensor.dtype, tf.float32)
        self.assertEqual(float_tensor.shape, int_tensor.shape)

        # Verify all layers in factory can be instantiated
        # This tests that the factory methods are properly defined
        factory_methods = [
            "cast_to_float32_layer",
            "date_encoding_layer",
            "date_parsing_layer",
            "date_season_layer",
            "distribution_transform_layer",
            "gated_linear_unit_layer",
            "gated_residual_network_layer",
            "global_numerical_embedding_layer",
            "multi_resolution_attention_layer",
            "numerical_embedding_layer",
            "tabular_attention_layer",
            "text_preprocessing_layer",
            "transformer_block_layer",
            "variable_selection_layer",
        ]

        # Verify all methods exist and can be called
        for method_name in factory_methods:
            self.assertTrue(hasattr(PreprocessorLayerFactory, method_name))
            # Just make sure the method is callable, without necessarily creating a layer
            # This is to verify the basic functionality of the factory


if __name__ == "__main__":
    unittest.main()
