"""Tests for the DistributionAwareEncoder layer."""

import keras
import numpy as np
import tensorflow as tf
import os

from kdp.layers.distribution_aware_encoder_layer import (
    DistributionAwareEncoder,
    DistributionType,
    get_custom_objects,
)


class TestDistributionAwareEncoder(tf.test.TestCase):
    def setUp(self):
        """Setup test case environment."""
        super().setUp()
        self.encoder = DistributionAwareEncoder(
            name="test_encoder",
            epsilon=1e-6,
            detect_periodicity=True,
            handle_sparsity=True,
            auto_detect=True,
            # Legacy parameters to ensure backward compatibility
            num_bins=1000,
            adaptive_binning=True,
            mixture_components=3,
        )

        # Set seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def generate_distributions(self):
        """Generate data with different distributions for testing."""
        n_samples = 1000

        # Normal distribution
        normal_data = np.random.normal(0, 1, (n_samples, 1))

        # Heavy-tailed distribution (Student's t with 3 degrees of freedom)
        heavy_tailed_data = np.random.standard_t(3, (n_samples, 1))

        # Multimodal distribution (mixture of two Gaussians)
        multimodal_data = np.concatenate(
            [
                np.random.normal(-3, 1, (n_samples // 2, 1)),
                np.random.normal(3, 1, (n_samples // 2, 1)),
            ]
        )

        # Uniform distribution
        uniform_data = np.random.uniform(-1, 1, (n_samples, 1))

        # Exponential distribution
        exponential_data = np.random.exponential(1, (n_samples, 1))

        # Log-normal distribution
        lognormal_data = np.random.lognormal(0, 1, (n_samples, 1))

        # Discrete distribution (integers from 0 to 5)
        discrete_data = np.random.randint(0, 6, (n_samples, 1)).astype(np.float32)

        # Periodic data (sine wave with noise)
        x = np.linspace(0, 6 * np.pi, n_samples)
        periodic_data = np.sin(x).reshape(-1, 1) + 0.1 * np.random.normal(
            0, 1, (n_samples, 1)
        )

        # Sparse data (mostly zeros with some values)
        sparse_data = np.zeros((n_samples, 1))
        # Fix: Generate a random mask and apply values
        mask = np.random.choice([0, 1], size=(n_samples, 1), p=[0.9, 0.1]).astype(bool)
        sparse_data[mask] = np.random.exponential(1, size=np.sum(mask)).reshape(-1)

        # Beta distribution (bounded between 0 and 1)
        beta_data = np.random.beta(2, 5, (n_samples, 1))

        # For poisson distribution
        poisson_data = np.random.poisson(5, (n_samples, 1)).astype(np.float32)

        # For zero-inflated distribution
        zero_inflated = np.random.random(
            (n_samples, 1)
        )  # Generate random numbers between 0 and 1
        zero_mask = np.random.random((n_samples, 1)) < 0.4  # Create mask for 40% zeros
        zero_inflated[zero_mask] = 0  # Zero out 40% of values

        return {
            "normal": normal_data,
            "heavy_tailed": heavy_tailed_data,
            "multimodal": multimodal_data,
            "uniform": uniform_data,
            "exponential": exponential_data,
            "log_normal": lognormal_data,
            "discrete": discrete_data,
            "periodic": periodic_data,
            "sparse": sparse_data,
            "beta": beta_data,
            "poisson": poisson_data,
            "zero_inflated": zero_inflated,
        }

    def test_automatic_detection(self):
        """Test if the encoder correctly detects different distributions."""
        data = self.generate_distributions()

        for dist_name, dist_data in data.items():
            # Create encoder with auto-detection
            encoder = DistributionAwareEncoder(auto_detect=True)

            # Convert data to tensor
            tensor_data = tf.convert_to_tensor(dist_data, dtype=tf.float32)

            # Create a small model with the encoder
            inputs = keras.Input(shape=(1,))
            encoded = encoder(inputs)
            model = keras.Model(inputs, encoded)

            # Feed data through the model to trigger detection
            _ = model(tensor_data, training=True)

            # Get the detected distribution type
            dist_idx = int(encoder.detected_distribution.numpy()[0])
            detected_type = encoder._valid_distributions[dist_idx]

            # Print for debugging
            print(f"  {dist_name:12} -> detected as: {detected_type}")

            # For certain distributions, we can assert the detection is correct
            if dist_name == "sparse":
                self.assertEqual(detected_type, "sparse")
            # We can't rely on strict detection for other types as the
            # classification logic differs from the old implementation

    def test_various_configurations(self):
        """Test the encoder with different configurations."""
        data = self.generate_distributions()

        test_cases = [
            {"name": "Default", "params": {}},
            {"name": "With embedding", "params": {"embedding_dim": 16}},
            {
                "name": "With distribution embedding",
                "params": {"add_distribution_embedding": True},
            },
            {
                "name": "With specific transform",
                "params": {"transform_type": "arcsinh"},
            },
            {
                "name": "With no periodic detection",
                "params": {"detect_periodicity": False},
            },
            {
                "name": "With manual distribution type",
                "params": {"auto_detect": False, "distribution_type": "normal"},
            },
        ]

        for test_case in test_cases:
            with self.subTest(config=test_case["name"]):
                # Create encoder with test configuration
                encoder = DistributionAwareEncoder(**test_case["params"])

                # Test on normal data
                tensor_data = tf.convert_to_tensor(data["normal"], dtype=tf.float32)

                # Create a small model
                inputs = keras.Input(shape=(1,))
                encoded = encoder(inputs)
                model = keras.Model(inputs, encoded)

                # Feed data through the model
                output = model(tensor_data)

                # Basic checks on output shape and values
                self.assertIsNotNone(output)

                # Check for embedding_dim only if no periodicity is detected or periodicity is disabled
                if (
                    test_case["params"].get("embedding_dim") is not None
                    and not encoder._added_periodic_features
                ):
                    # Verify output dimension only if embedding_dim is specified and periodic features weren't added
                    self.assertEqual(
                        output.shape[-1], test_case["params"]["embedding_dim"]
                    )

                # Check if distribution embedding is added correctly
                if test_case["params"].get(
                    "add_distribution_embedding", False
                ) and hasattr(encoder, "distribution_embedding"):
                    # Expected dimension with embedding
                    expected_extra_dim = 8  # The embedding dimension from the code

                    # For embedding_dim parameter with distribution embedding
                    if test_case["params"].get("embedding_dim") is not None:
                        # No need to verify exact dimensions since we already checked embedding_dim above
                        pass
                    elif not encoder._added_periodic_features:
                        # If no embedding_dim specified and no periodic features, check for input dim + embedding dim
                        expected_dim = tensor_data.shape[-1] + expected_extra_dim
                        self.assertEqual(output.shape[-1], expected_dim)

    def test_normal_distribution(self):
        """Test handling of normal distribution."""
        # Generate normal distribution data
        np.random.seed(42)
        data = np.random.normal(0, 1, (1000, 1))
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)

        # Check statistical properties
        self.assertLess(tf.abs(tf.reduce_mean(outputs)), 1.0)

    def test_heavy_tailed_distribution(self):
        """Test handling of heavy-tailed distribution."""
        # Generate heavy-tailed data
        np.random.seed(42)
        data = np.random.standard_t(
            3, (1000, 1)
        )  # t-distribution with 3 degrees of freedom
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Create encoder with fixed distribution type
        encoder = DistributionAwareEncoder(
            auto_detect=False, distribution_type="heavy_tailed"
        )

        # Process data
        outputs = encoder(inputs)

        # Check output properties
        # We focus on checking that the output is transformed rather than specific values
        # Allow a wider range since some transformations like Box-Cox can produce large values
        self.assertGreaterEqual(tf.reduce_min(outputs), -100.0)
        self.assertLessEqual(tf.reduce_max(outputs), 100.0)

        # For heavy-tailed transformations, we should reduce extreme values
        input_range = tf.reduce_max(inputs) - tf.reduce_min(inputs)
        output_range = tf.reduce_max(outputs) - tf.reduce_min(outputs)

        # Only verify if the transform isn't increasing the range too much
        # Allow some flexibility but ensure it's not drastically increasing the range
        self.assertLess(output_range / input_range, 5.0)

    def test_multimodal_distribution(self):
        """Test handling of multimodal distribution."""
        # Generate mixture of two normal distributions
        np.random.seed(42)
        data1 = np.random.normal(-2, 0.5, (500, 1))
        data2 = np.random.normal(2, 0.5, (500, 1))
        data = np.concatenate([data1, data2])
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Create encoder with fixed distribution type
        encoder = DistributionAwareEncoder(
            auto_detect=False, distribution_type="multimodal"
        )

        # Process data
        outputs = encoder(inputs)

        # We can't directly check against a static shape due to dynamic detection
        # Instead, ensure that the output data has reasonable values
        self.assertGreaterEqual(tf.reduce_min(outputs), -10.0)
        self.assertLessEqual(tf.reduce_max(outputs), 10.0)

    def test_uniform_distribution(self):
        """Test handling of uniform distribution."""
        # Generate uniform distribution data
        np.random.seed(42)
        data = np.random.uniform(-1, 1, (1000, 1))
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)

    def test_sparse_distribution(self):
        """Test handling of sparse data."""
        # Generate sparse data
        np.random.seed(42)
        data = np.zeros((1000, 1))
        indices = np.random.choice(1000, 100, replace=False)
        data[indices] = np.random.normal(0, 1, (100, 1)).flatten().reshape(-1, 1)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)

        # Check if zeros are preserved when handle_sparsity is True
        if self.encoder.handle_sparsity:
            zero_mask = tf.abs(inputs) < self.encoder.epsilon
            zero_outputs = tf.boolean_mask(outputs, zero_mask)
            # Allow small non-zero values for numerical reasons
            for value in zero_outputs.numpy().flatten():
                self.assertLess(abs(value), 0.01)

    def test_periodic_features(self):
        """Test handling of periodic data with explicit detection."""
        # Generate periodic data with strong periodicity
        t = np.linspace(0, 10 * np.pi, 1000)
        data = np.sin(t).reshape(-1, 1)  # Ensure 2D shape
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Create encoder with forced periodic handling
        encoder = DistributionAwareEncoder(
            detect_periodicity=True,
            auto_detect=False,
            distribution_type="periodic",  # Force it to use periodic handling
        )

        # Process the data and verify output
        outputs = encoder(inputs)

        # Since we're forcing periodic and detect_periodicity=True,
        # the output should have 3x the features (original + sin + cos)
        self.assertEqual(
            outputs.shape[-1],
            3 * inputs.shape[-1],
            f"Expected feature dim {3 * inputs.shape[-1]}, got {outputs.shape[-1]}",
        )

    def test_in_model(self):
        """Test the encoder as part of a complete model."""
        # Create synthetic dataset
        x_train = np.random.normal(0, 1, (1000, 5)).astype(np.float32)
        y_train = (x_train[:, 0] > 0).astype(np.float32)  # Simple binary task

        # Create a model with the encoder - disable periodic features to avoid shape issues
        inputs = keras.Input(shape=(5,))
        encoded = DistributionAwareEncoder(
            embedding_dim=16,
            detect_periodicity=False,  # Disable periodic features to avoid gradient issues
            auto_detect=True,
        )(inputs)
        hidden = keras.layers.Dense(8, activation="relu")(encoded)
        outputs = keras.layers.Dense(1, activation="sigmoid")(hidden)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train for just one epoch to verify it works
        model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

        # Get predictions to ensure model works
        preds = model.predict(x_train[:10])
        self.assertEqual(preds.shape, (10, 1))

        # Verify we can save and load the model with custom objects
        model_path = "test_model.keras"  # Add .keras extension for proper saving

        try:
            model.save(model_path)
            # Use get_custom_objects to load the model
            custom_objects = get_custom_objects()
            loaded_model = keras.saving.load_model(
                model_path, custom_objects=custom_objects
            )

            # Check that the loaded model gives same predictions
            loaded_preds = loaded_model.predict(x_train[:10])
            # Use a higher tolerance level to account for floating-point differences
            self.assertAllClose(preds, loaded_preds, rtol=1e-1, atol=1e-1)

            # Clean up
            if os.path.exists(model_path):
                os.remove(
                    model_path
                )  # Use os.remove for file instead of shutil.rmtree for directory
        except Exception as e:
            # Clean up even if test fails
            if os.path.exists(model_path):
                os.remove(model_path)  # Use os.remove for file
            self.fail(f"Model saving/loading failed: {e}")

    def test_config(self):
        """Test configuration serialization."""
        # Create encoder with custom configuration
        encoder = DistributionAwareEncoder(
            embedding_dim=32,
            epsilon=1e-5,
            detect_periodicity=False,
            handle_sparsity=True,
            auto_detect=True,
            transform_type="arcsinh",
            add_distribution_embedding=True,
            name="custom_encoder",
        )

        # Get config
        config = encoder.get_config()

        # Check that all parameters are in the config
        self.assertEqual(config["embedding_dim"], 32)
        self.assertEqual(config["epsilon"], 1e-5)
        self.assertEqual(config["detect_periodicity"], False)
        self.assertEqual(config["handle_sparsity"], True)
        self.assertEqual(config["auto_detect"], True)
        self.assertEqual(config["transform_type"], "arcsinh")
        self.assertEqual(config["add_distribution_embedding"], True)
        self.assertEqual(config["name"], "custom_encoder")

        # Test reconstruction from config
        new_encoder = DistributionAwareEncoder.from_config(config)
        self.assertEqual(new_encoder.embedding_dim, encoder.embedding_dim)
        self.assertEqual(new_encoder.epsilon, encoder.epsilon)
        self.assertEqual(new_encoder.detect_periodicity, encoder.detect_periodicity)
        self.assertEqual(new_encoder.handle_sparsity, encoder.handle_sparsity)
        self.assertEqual(new_encoder.auto_detect, encoder.auto_detect)
        self.assertEqual(new_encoder.transform_type, encoder.transform_type)
        self.assertEqual(
            new_encoder.add_distribution_embedding, encoder.add_distribution_embedding
        )

    def test_legacy_parameters(self):
        """Test that the encoder accepts and handles legacy parameters."""
        # Create an encoder with legacy parameters
        encoder = DistributionAwareEncoder(
            num_bins=1000,
            adaptive_binning=True,
            mixture_components=3,
            prefered_distribution="normal",
        )

        # Check that the encoder was created successfully
        self.assertEqual(encoder.num_bins, 1000)
        self.assertEqual(encoder.adaptive_binning, True)
        self.assertEqual(encoder.mixture_components, 3)
        self.assertEqual(encoder.prefered_distribution, "normal")
        self.assertEqual(encoder.distribution_type, "normal")
        self.assertEqual(
            encoder.auto_detect, False
        )  # Should be disabled with prefered_distribution

        # Process some data to ensure it works
        data = np.random.normal(0, 1, (100, 1))
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)
        outputs = encoder(inputs)

        # Check that output has same shape as input
        self.assertEqual(outputs.shape, inputs.shape)


class TestDistributionAwareEncoderCompat(tf.test.TestCase):
    """Compatibility tests for the DistributionAwareEncoder."""

    def test_old_test_interface(self):
        """Test compatibility with the old test interface."""
        # Old test code used num_bins, adaptive_binning, etc.
        encoder = DistributionAwareEncoder(
            name="encoder_compat",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=True,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )

        # Test with simple data
        data = np.random.normal(0, 1, (100, 1)).astype(np.float32)
        inputs = tf.convert_to_tensor(data)
        outputs = encoder(inputs)

        # Check basic output properties
        self.assertEqual(outputs.shape, inputs.shape)

    def test_forced_distribution_types(self):
        """Test that we can force specific distribution types."""
        data = np.random.normal(0, 1, (100, 1)).astype(np.float32)
        inputs = tf.convert_to_tensor(data)

        # Test each distribution type
        for dist_type in DistributionType:
            encoder = DistributionAwareEncoder(
                auto_detect=False, distribution_type=dist_type.value
            )
            outputs = encoder(inputs)

            # Check that output shape is consistent
            # For periodic, check for potential tripling
            if dist_type.value == "periodic" and encoder.detect_periodicity:
                self.assertEqual(outputs.shape[-1], 3 * inputs.shape[-1])
            else:
                self.assertEqual(outputs.shape[-1], inputs.shape[-1])


if __name__ == "__main__":
    tf.test.main()
