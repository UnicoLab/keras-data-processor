import keras
# import numpy as np
# import pytest
# import tensorflow as tf

# from kdp.layers.distribution_aware_encoder_layer import DistributionAwareEncoder, DistributionType


# @pytest.fixture
# def encoder():
#     """Create a DistributionAwareEncoder instance for testing."""
#     return DistributionAwareEncoder(
#         num_bins=10, detect_periodicity=True, handle_sparsity=True,
#     )


# def test_normal_distribution(encoder):
#     """Test that normal distribution is correctly identified and transformed."""
#     # Generate normal distribution data
#     np.random.seed(42)
#     data = np.random.normal(0, 1, (100, 1))

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite and in a reasonable range
#     assert np.all(np.isfinite(transformed))
#     assert -2.0 <= np.min(transformed) <= 2.0
#     assert -2.0 <= np.max(transformed) <= 2.0


# def test_heavy_tailed_distribution(encoder):
#     """Test that heavy-tailed distribution is correctly identified and transformed."""
#     # Generate t-distribution data (heavy-tailed)
#     np.random.seed(42)
#     data = np.random.standard_t(df=3, size=(100, 1))

#     # Force heavy-tailed distribution type
#     encoder.prefered_distribution = DistributionType.HEAVY_TAILED

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite and in a reasonable range
#     assert np.all(np.isfinite(transformed))
#     assert 0.0 <= np.min(transformed) <= 1.0
#     assert 0.0 <= np.max(transformed) <= 1.0


# def test_multimodal_distribution(encoder):
#     """Test that multimodal distribution is correctly identified and transformed."""
#     # Generate bimodal distribution
#     np.random.seed(42)
#     data = np.concatenate(
#         [np.random.normal(-3, 1, (50, 1)), np.random.normal(3, 1, (50, 1))]
#     )

#     # Force multimodal distribution type
#     encoder.prefered_distribution = DistributionType.MULTIMODAL

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite and in a reasonable range
#     assert np.all(np.isfinite(transformed))
#     assert 0.0 <= np.min(transformed) <= 1.0
#     assert 0.0 <= np.max(transformed) <= 1.0


# def test_uniform_distribution(encoder):
#     """Test that uniform distribution is correctly identified and transformed."""
#     # Generate uniform distribution data
#     np.random.seed(42)
#     data = np.random.uniform(-1, 1, (100, 1))

#     # Force uniform distribution type
#     encoder.prefered_distribution = DistributionType.UNIFORM

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite and in a reasonable range
#     assert np.all(np.isfinite(transformed))
#     assert 0.0 <= np.min(transformed) <= 1.0
#     assert 0.0 <= np.max(transformed) <= 1.0


# def test_discrete_distribution(encoder):
#     """Test that discrete distribution is correctly identified and transformed."""
#     # Generate discrete data
#     data = np.array([[1], [2], [3], [1], [2], [3], [1], [2], [3]])

#     # Force discrete distribution type
#     encoder.prefered_distribution = DistributionType.DISCRETE

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite and in a reasonable range
#     assert np.all(np.isfinite(transformed))
#     assert 0.0 <= np.min(transformed) <= 1.0
#     assert 0.0 <= np.max(transformed) <= 1.0

#     # Check that the discrete values are mapped to distinct values
#     unique_values = np.unique(transformed)
#     assert len(unique_values) == 3


# def test_sparse_distribution(encoder):
#     """Test that sparse distribution is correctly identified and transformed."""
#     # Generate sparse data (mostly zeros)
#     np.random.seed(42)
#     data = np.zeros((100, 1))
#     data[np.random.choice(100, 10)] = np.random.exponential(1, 10)

#     # Force sparse distribution type
#     encoder.prefered_distribution = DistributionType.SPARSE

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite
#     assert np.all(np.isfinite(transformed))

#     # Check that zeros in input remain zeros in output
#     zero_indices = np.where(np.abs(data) < 1e-6)[0]
#     assert np.all(np.abs(transformed[zero_indices]) < 1e-6)


# def test_periodic_distribution(encoder):
#     """Test that periodic distribution is correctly identified and transformed."""
#     # Generate periodic data
#     x = np.linspace(0, 4 * np.pi, 100).reshape(-1, 1)
#     data = np.sin(x)

#     # Force periodic distribution type
#     encoder.prefered_distribution = DistributionType.PERIODIC

#     # Transform the data
#     transformed = encoder(data)

#     # Check that the output is finite
#     assert np.all(np.isfinite(transformed))

#     # Check that the output has the expected shape (should be 2D for sine/cosine features)
#     assert transformed.shape[1] == 2


# def test_graph_mode_compatibility(encoder):
#     """Test that the encoder works in graph mode."""
#     # Create a simple model with the encoder
#     inputs = keras.layers.Input(shape=(1,))
#     encoded = encoder(inputs)
#     outputs = keras.layers.Dense(1)(encoded)
#     model = keras.Model(inputs=inputs, outputs=outputs)

#     # Compile the model
#     model.compile(optimizer="adam", loss="mse")

#     # Generate some data
#     np.random.seed(42)
#     data = np.random.normal(0, 1, (100, 1))
#     targets = np.random.normal(0, 1, (100, 1))

#     # Train for one step to ensure graph compatibility
#     model.fit(data, targets, epochs=1, verbose=0)

#     # If we got here without errors, the test passes
#     assert True
