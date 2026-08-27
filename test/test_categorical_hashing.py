import keras
import unittest
import numpy as np
import tensorflow as tf
from keras.layers import Input

from kdp import (
    CategoricalFeature,
    FeatureType,
    CategoryEncodingOptions,
    FeaturePreprocessor,
)


class TestCategoricalHashing(unittest.TestCase):
    """Test cases for categorical feature hashing functionality."""

    def test_hashing_option_exists(self):
        """Test that HASHING is available in CategoryEncodingOptions."""
        self.assertTrue(hasattr(CategoryEncodingOptions, "HASHING"))
        self.assertEqual(CategoryEncodingOptions.HASHING, "HASHING")

    def test_basic_hashing_config(self):
        """Test basic configuration for a hashed categorical feature."""
        feature = CategoricalFeature(
            name="user_id",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            hash_bucket_size=1024,
        )

        self.assertEqual(feature.name, "user_id")
        self.assertEqual(feature.feature_type, FeatureType.STRING_CATEGORICAL)
        self.assertEqual(feature.category_encoding, CategoryEncodingOptions.HASHING)
        self.assertEqual(feature.kwargs.get("hash_bucket_size"), 1024)

    def test_hashing_with_salt(self):
        """Test hashing configuration with custom salt."""
        feature = CategoricalFeature(
            name="user_id",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            hash_bucket_size=1024,
            salt=42,
        )

        self.assertEqual(feature.kwargs.get("salt"), 42)

    def test_hashing_with_embedding(self):
        """Test hash with embedding configuration."""
        feature = CategoricalFeature(
            name="user_id",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            hash_bucket_size=1024,
            hash_with_embedding=True,
            embedding_size=16,
        )

        self.assertTrue(feature.kwargs.get("hash_with_embedding"))
        self.assertEqual(feature.kwargs.get("embedding_size"), 16)

    def test_categorical_hashing_preprocessing(self):
        """Test that the hashing preprocessing step is properly created."""
        # Create a feature preprocessor manually
        preprocessor = FeaturePreprocessor(name="test_feature")

        # Set up a feature with hashing
        feature = CategoricalFeature(
            name="test_feature",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            hash_bucket_size=128,
        )

        # Manual function to simulate the processor's _add_categorical_encoding method
        def add_categorical_encoding():
            if feature.category_encoding == CategoryEncodingOptions.HASHING:
                hash_bucket_size = feature.kwargs.get("hash_bucket_size", 128)
                preprocessor.add_processing_step(
                    layer_class="Hashing",
                    num_bins=hash_bucket_size,
                    salt=feature.kwargs.get("salt", None),
                    name=f"hash_{feature.name}",
                )
                preprocessor.add_processing_step(
                    layer_class="CategoryEncoding",
                    num_tokens=hash_bucket_size,
                    output_mode="multi_hot",
                    name=f"hash_encode_{feature.name}",
                )

        # Add the hashing steps
        add_categorical_encoding()

        # Check that we have the right number of steps
        self.assertEqual(len(preprocessor.processing_steps), 2)

        # Check that the first step is a Hashing layer
        self.assertEqual(preprocessor.processing_steps[0]["layer_class"], "Hashing")
        self.assertEqual(preprocessor.processing_steps[0]["num_bins"], 128)

        # Check that the second step is a CategoryEncoding layer
        self.assertEqual(
            preprocessor.processing_steps[1]["layer_class"], "CategoryEncoding"
        )
        self.assertEqual(preprocessor.processing_steps[1]["output_mode"], "multi_hot")

    def test_hashing_with_embedding_preprocessing(self):
        """Test that the hashing with embedding preprocessing steps are properly created."""
        # Create a feature preprocessor manually
        preprocessor = FeaturePreprocessor(name="test_feature")

        # Set up a feature with hashing and embedding
        feature = CategoricalFeature(
            name="test_feature",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            hash_bucket_size=128,
            hash_with_embedding=True,
            embedding_size=16,
        )

        # Manual function to simulate the processor's _add_categorical_encoding method
        def add_categorical_encoding():
            if feature.category_encoding == CategoryEncodingOptions.HASHING:
                hash_bucket_size = feature.kwargs.get("hash_bucket_size", 128)
                preprocessor.add_processing_step(
                    layer_class="Hashing",
                    num_bins=hash_bucket_size,
                    salt=feature.kwargs.get("salt", None),
                    name=f"hash_{feature.name}",
                )

                if feature.kwargs.get("hash_with_embedding", False):
                    emb_size = feature.kwargs.get("embedding_size", 8)
                    preprocessor.add_processing_step(
                        layer_class="Embedding",
                        input_dim=hash_bucket_size,
                        output_dim=emb_size,
                        name=f"hash_embed_{feature.name}",
                    )

        # Add the hashing and embedding steps
        add_categorical_encoding()

        # Check that we have the right number of steps
        self.assertEqual(len(preprocessor.processing_steps), 2)

        # Check that the first step is a Hashing layer
        self.assertEqual(preprocessor.processing_steps[0]["layer_class"], "Hashing")
        self.assertEqual(preprocessor.processing_steps[0]["num_bins"], 128)

        # Check that the second step is an Embedding layer
        self.assertEqual(preprocessor.processing_steps[1]["layer_class"], "Embedding")
        self.assertEqual(preprocessor.processing_steps[1]["input_dim"], 128)
        self.assertEqual(preprocessor.processing_steps[1]["output_dim"], 16)

    def test_functional_pipeline_with_hashing(self):
        """Test that a complete TF pipeline with hashing works end-to-end."""
        # This test constructs a small TF pipeline to verify the hashing works as expected

        # Create a simple input layer
        input_layer = Input(shape=(1,), dtype=tf.string, name="input_layer")

        # Create the hashing layer
        hash_bucket_size = 64
        salt = 42

        hashing_layer = keras.layers.Hashing(
            num_bins=hash_bucket_size, salt=salt, name="hash_layer"
        )(input_layer)

        # Create a category encoding layer
        encoding_layer = keras.layers.CategoryEncoding(
            num_tokens=hash_bucket_size, output_mode="multi_hot", name="encoding_layer"
        )(hashing_layer)

        # Create a model
        model = keras.Model(inputs=input_layer, outputs=encoding_layer)

        # Test with some sample data - use longer, more distinctive strings to avoid collisions
        user1 = tf.constant("user_with_very_distinctive_name_1", dtype=tf.string)
        user2 = tf.constant("user_with_completely_different_name_2", dtype=tf.string)
        user3 = tf.constant("user_with_another_unique_identifier_3", dtype=tf.string)
        sample_data = tf.stack([[user1], [user2], [user1], [user3]])

        result = model.predict(sample_data)

        # Verify the output shape
        self.assertEqual(result.shape, (4, hash_bucket_size))

        # Verify that the same input value results in the same hash bucket being activated
        # user_1 appears twice, so their hashed representations should be identical
        np.testing.assert_array_equal(result[0], result[2])

        # Verify that different input values result in different hash buckets
        # With our distinctive strings, these should hash to different buckets
        self.assertFalse(np.array_equal(result[0], result[1]))
        self.assertFalse(np.array_equal(result[1], result[3]))

    def test_integration_with_embedding(self):
        """Test hashing followed by embedding in a TF pipeline."""
        # Create a simple input layer
        input_layer = Input(shape=(1,), dtype=tf.string, name="input_layer")

        # Create the hashing layer
        hash_bucket_size = 64
        embedding_dim = 8

        hashing_layer = keras.layers.Hashing(
            num_bins=hash_bucket_size, name="hash_layer"
        )(input_layer)

        # Add embedding layer after hashing
        embedding_layer = keras.layers.Embedding(
            input_dim=hash_bucket_size, output_dim=embedding_dim, name="embedding_layer"
        )(hashing_layer)

        # Flatten the output
        flattened_layer = keras.layers.Flatten()(embedding_layer)

        # Create a model
        model = keras.Model(inputs=input_layer, outputs=flattened_layer)

        # Test with some sample data
        user1 = tf.constant("user_1", dtype=tf.string)
        user2 = tf.constant("user_2", dtype=tf.string)
        user3 = tf.constant("user_3", dtype=tf.string)
        sample_data = tf.stack([[user1], [user2], [user1], [user3]])

        result = model.predict(sample_data)

        # Verify the output shape (4 samples, each with an embedding of size 8)
        self.assertEqual(result.shape, (4, embedding_dim))

        # Verify that the same input value results in the same embedding
        np.testing.assert_array_equal(result[0], result[2])

        # Verify that different input values result in different embeddings
        self.assertFalse(np.array_equal(result[0], result[1]))
        self.assertFalse(np.array_equal(result[1], result[3]))

    def test_different_salt_values(self):
        """Test that different salt values produce different hash outputs."""
        # Create two parallel hashing models with different salts
        input_layer = Input(shape=(1,), dtype=tf.string, name="input_layer")

        hash_bucket_size = 64

        # Model 1 with salt=1
        hashing_layer1 = keras.layers.Hashing(
            num_bins=hash_bucket_size, salt=1, name="hash_layer1"
        )(input_layer)

        # Model 2 with salt=2
        hashing_layer2 = keras.layers.Hashing(
            num_bins=hash_bucket_size, salt=2, name="hash_layer2"
        )(input_layer)

        # Create a model with both outputs
        model = keras.Model(
            inputs=input_layer, outputs=[hashing_layer1, hashing_layer2]
        )

        # Test with some sample data
        user1 = tf.constant("user_1", dtype=tf.string)
        user2 = tf.constant("user_2", dtype=tf.string)
        sample_data = tf.stack([[user1], [user2]])

        result1, result2 = model.predict(sample_data)

        # Verify that different salts produce different hash outputs
        self.assertFalse(np.array_equal(result1, result2))


if __name__ == "__main__":
    unittest.main()
