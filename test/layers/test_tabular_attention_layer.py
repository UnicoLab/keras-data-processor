import keras
import tensorflow as tf
import numpy as np
import pytest

from kdp.layers.tabular_attention_layer import TabularAttention


def test_tabular_attention_layer_init():
    """Test initialization of TabularAttention layer."""
    layer = TabularAttention(num_heads=4, d_model=64)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.dropout_rate == 0.1  # default value


def test_tabular_attention_layer_config():
    """Test get_config and from_config methods."""
    original_layer = TabularAttention(
        num_heads=4, d_model=64, dropout_rate=0.2, name="test_attention"
    )

    config = original_layer.get_config()
    restored_layer = TabularAttention.from_config(config)

    assert restored_layer.num_heads == original_layer.num_heads
    assert restored_layer.d_model == original_layer.d_model
    assert restored_layer.dropout_rate == original_layer.dropout_rate
    assert restored_layer.name == original_layer.name


def test_tabular_attention_computation():
    """Test the computation performed by TabularAttention layer."""
    batch_size = 32
    num_samples = 10
    num_features = 8
    d_model = 16

    # Create a layer instance
    layer = TabularAttention(num_heads=2, d_model=d_model)

    # Create input data
    inputs = tf.random.normal((batch_size, num_samples, num_features))

    # Call the layer
    outputs = layer(inputs, training=True)

    # Check output shape - output will have d_model dimension
    assert outputs.shape == (batch_size, num_samples, d_model)


def test_tabular_attention_training():
    """Test TabularAttention layer in training vs inference modes."""
    batch_size = 16
    num_samples = 5
    num_features = 4

    layer = TabularAttention(num_heads=2, d_model=8, dropout_rate=0.5)
    inputs = tf.random.normal((batch_size, num_samples, num_features))

    # Test in training mode
    outputs_training = layer(inputs, training=True)

    # Test in inference mode
    outputs_inference = layer(inputs, training=False)

    # The outputs should be different due to dropout
    assert not np.allclose(outputs_training.numpy(), outputs_inference.numpy())


def test_tabular_attention_invalid_inputs():
    """Test TabularAttention layer with invalid inputs."""
    layer = TabularAttention(num_heads=2, d_model=8)

    # Test with wrong input shape
    with pytest.raises(ValueError, match="Input tensor must be 3-dimensional"):
        # Missing batch dimension
        inputs = tf.random.normal((5, 4))
        layer(inputs)

    with pytest.raises(ValueError):
        # Wrong rank
        inputs = tf.random.normal((16, 5, 4, 2))
        layer(inputs)


def test_tabular_attention_end_to_end_simple():
    """Test TabularAttention in a simple end-to-end model."""
    batch_size = 16
    num_samples = 5
    num_features = 4

    # Create a simple model with TabularAttention
    inputs = keras.Input(shape=(num_samples, num_features))
    x = TabularAttention(num_heads=2, d_model=8)(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Create some dummy data
    X = tf.random.normal((batch_size, num_samples, num_features))
    y = tf.random.normal((batch_size, num_samples, 1))

    # Train for one epoch
    history = model.fit(X, y, epochs=1, verbose=0)

    # Check if loss was computed
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_tabular_attention_shapes():
    """Test that TabularAttention produces correct output shapes."""
    # Setup
    batch_size = 32
    num_samples = 10
    num_features = 8
    d_model = 16
    num_heads = 4

    layer = TabularAttention(num_heads=num_heads, d_model=d_model)

    # Create sample inputs
    inputs = tf.random.normal((batch_size, num_samples, num_features))

    # Process features
    outputs = layer(inputs, training=True)

    # Check shapes
    assert outputs.shape == (batch_size, num_samples, d_model)

    # Test with different input shapes
    inputs_2d = tf.random.normal((batch_size, num_features))
    with pytest.raises(ValueError):
        layer(inputs_2d)  # Should raise error for 2D input


def test_tabular_attention_training_modes():
    """Test TabularAttention behavior in training vs inference modes."""
    batch_size = 16
    num_samples = 8
    num_features = 12
    d_model = 24
    num_heads = 3
    dropout_rate = 0.5  # High dropout for visible effect

    layer = TabularAttention(
        num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate
    )

    # Create inputs
    inputs = tf.random.normal((batch_size, num_samples, num_features))

    # Get outputs in training mode
    train_output = layer(inputs, training=True)

    # Get outputs in inference mode
    infer_output = layer(inputs, training=False)

    # Check that outputs are different due to dropout
    assert not np.allclose(train_output.numpy(), infer_output.numpy())


def test_tabular_attention_feature_interactions():
    """Test that TabularAttention captures feature interactions."""
    batch_size = 8
    num_samples = 4
    num_features = 6
    d_model = 12
    num_heads = 2

    layer = TabularAttention(num_heads=num_heads, d_model=d_model)

    # Create correlated features
    base_feature = tf.random.normal((batch_size, num_samples, 1))
    correlated_features = tf.concat(
        [
            base_feature,
            base_feature * 2
            + tf.random.normal((batch_size, num_samples, 1), stddev=0.1),
            tf.random.normal((batch_size, num_samples, num_features - 2)),
        ],
        axis=-1,
    )

    # Process features
    output_correlated = layer(correlated_features)

    # Create uncorrelated features
    uncorrelated_features = tf.random.normal((batch_size, num_samples, num_features))
    output_uncorrelated = layer(uncorrelated_features)

    # The attention patterns should be different
    assert not np.allclose(
        output_correlated.numpy(), output_uncorrelated.numpy(), rtol=1e-3
    )


def test_tabular_attention_config():
    """Test configuration saving and loading."""
    original_layer = TabularAttention(num_heads=4, d_model=32, dropout_rate=0.2)

    config = original_layer.get_config()
    restored_layer = TabularAttention.from_config(config)

    assert restored_layer.num_heads == original_layer.num_heads
    assert restored_layer.d_model == original_layer.d_model
    assert restored_layer.dropout_rate == original_layer.dropout_rate


def test_tabular_attention_end_to_end():
    """Test TabularAttention in a simple end-to-end model."""
    batch_size = 16
    num_samples = 6
    num_features = 8
    d_model = 16
    num_heads = 2

    # Create a simple model
    inputs = keras.Input(shape=(num_samples, num_features))
    attention_layer = TabularAttention(num_heads=num_heads, d_model=d_model)

    x = attention_layer(inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer="adam", loss="mse")

    # Create dummy data
    X = tf.random.normal((batch_size, num_samples, num_features))
    y = tf.random.normal((batch_size, 1))

    # Train for one epoch
    history = model.fit(X, y, epochs=1, verbose=0)

    # Check if loss was computed
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_tabular_attention_masking():
    """Test TabularAttention with masked inputs."""
    batch_size = 8
    num_samples = 5
    num_features = 4
    d_model = 8
    num_heads = 2

    layer = TabularAttention(num_heads=num_heads, d_model=d_model)

    # Create inputs with masked values
    inputs = tf.random.normal((batch_size, num_samples, num_features))
    mask = tf.random.uniform((batch_size, num_samples)) > 0.3
    masked_inputs = tf.where(tf.expand_dims(mask, -1), inputs, tf.zeros_like(inputs))

    # Process both masked and unmasked inputs
    output_masked = layer(masked_inputs)
    output_unmasked = layer(inputs)

    # Outputs should be different
    assert not np.allclose(output_masked.numpy(), output_unmasked.numpy())
