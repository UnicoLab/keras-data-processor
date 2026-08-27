import keras
import numpy as np
import tensorflow as tf

from kdp.layers.multi_resolution_tabular_attention_layer import (
    MultiResolutionTabularAttention,
)
from kdp.layers_factory import PreprocessorLayerFactory


def test_multi_resolution_attention_layer_init():
    """Test initialization of MultiResolutionTabularAttention layer."""
    layer = MultiResolutionTabularAttention(num_heads=4, d_model=64, embedding_dim=32)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.embedding_dim == 32
    assert layer.dropout_rate == 0.1  # default value


def test_multi_resolution_attention_layer_config():
    """Test get_config and from_config methods for MultiResolutionTabularAttention."""
    original_layer = MultiResolutionTabularAttention(
        num_heads=4,
        d_model=64,
        embedding_dim=32,
        dropout_rate=0.2,
        name="test_multi_attention",
    )

    config = original_layer.get_config()
    restored_layer = MultiResolutionTabularAttention.from_config(config)

    assert restored_layer.num_heads == original_layer.num_heads
    assert restored_layer.d_model == original_layer.d_model
    assert restored_layer.embedding_dim == original_layer.embedding_dim
    assert restored_layer.dropout_rate == original_layer.dropout_rate
    assert restored_layer.name == original_layer.name


def test_multi_resolution_attention_computation():
    """Test the computation performed by MultiResolutionTabularAttention layer."""
    batch_size = 32
    num_numerical = 8
    num_categorical = 5
    numerical_dim = 16
    categorical_dim = 8

    # Create a layer instance
    layer = MultiResolutionTabularAttention(
        num_heads=2, d_model=numerical_dim, embedding_dim=categorical_dim
    )

    # Create input data
    numerical_features = tf.random.normal((batch_size, num_numerical, numerical_dim))
    categorical_features = tf.random.normal(
        (batch_size, num_categorical, categorical_dim)
    )

    # Call the layer
    numerical_output, categorical_output = layer(
        numerical_features, categorical_features, training=True
    )

    # Check output shapes
    assert numerical_output.shape == (batch_size, num_numerical, numerical_dim)
    assert categorical_output.shape == (batch_size, num_categorical, numerical_dim)

    # Test with different batch sizes
    numerical_features_2 = tf.random.normal((64, num_numerical, numerical_dim))
    categorical_features_2 = tf.random.normal((64, num_categorical, categorical_dim))
    num_out_2, cat_out_2 = layer(
        numerical_features_2, categorical_features_2, training=False
    )
    assert num_out_2.shape == (64, num_numerical, numerical_dim)
    assert cat_out_2.shape == (64, num_categorical, numerical_dim)


def test_multi_resolution_attention_training():
    """Test MultiResolutionTabularAttention layer in training vs inference modes."""
    batch_size = 16
    num_numerical = 4
    num_categorical = 3
    numerical_dim = 8
    categorical_dim = 4

    layer = MultiResolutionTabularAttention(
        num_heads=2,
        d_model=numerical_dim,
        embedding_dim=categorical_dim,
        dropout_rate=0.5,
    )

    numerical_features = tf.random.normal((batch_size, num_numerical, numerical_dim))
    categorical_features = tf.random.normal(
        (batch_size, num_categorical, categorical_dim)
    )

    # Test in training mode
    num_train, cat_train = layer(
        numerical_features, categorical_features, training=True
    )

    # Test in inference mode
    num_infer, cat_infer = layer(
        numerical_features, categorical_features, training=False
    )

    # The outputs should be different due to dropout
    assert not np.allclose(num_train.numpy(), num_infer.numpy())
    assert not np.allclose(cat_train.numpy(), cat_infer.numpy())


def test_multi_resolution_attention_factory():
    """Test creation of MultiResolutionTabularAttention layer through PreprocessorLayerFactory."""
    layer = PreprocessorLayerFactory.multi_resolution_attention_layer(
        num_heads=4,
        d_model=64,
        embedding_dim=32,
        name="test_multi_attention",
        dropout_rate=0.2,
    )

    assert isinstance(layer, MultiResolutionTabularAttention)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.embedding_dim == 32
    assert layer.dropout_rate == 0.2
    assert layer.name == "test_multi_attention"


def test_multi_resolution_attention_end_to_end_complex():
    """Test MultiResolutionTabularAttention layer in a simple end-to-end model."""
    batch_size = 16
    num_numerical = 4
    num_categorical = 3
    numerical_dim = 8
    categorical_dim = 4
    output_dim = 1

    # Create inputs
    numerical_inputs = keras.Input(shape=(num_numerical, numerical_dim))
    categorical_inputs = keras.Input(shape=(num_categorical, categorical_dim))

    # Apply multi-resolution attention
    num_attended, cat_attended = MultiResolutionTabularAttention(
        num_heads=2, d_model=numerical_dim, embedding_dim=categorical_dim
    )(numerical_inputs, categorical_inputs)

    # Combine outputs
    combined = keras.layers.Concatenate(axis=1)([num_attended, cat_attended])
    outputs = keras.layers.Dense(output_dim)(combined)

    # Create model
    model = keras.Model(inputs=[numerical_inputs, categorical_inputs], outputs=outputs)

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Create dummy data
    X_num = tf.random.normal((batch_size, num_numerical, numerical_dim))
    X_cat = tf.random.normal((batch_size, num_categorical, categorical_dim))
    y = tf.random.normal((batch_size, num_numerical + num_categorical, output_dim))

    # Train for one epoch
    history = model.fit([X_num, X_cat], y, epochs=1, verbose=0)

    # Check if loss was computed
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_multi_resolution_attention_shapes():
    """Test that MultiResolutionTabularAttention produces correct output shapes."""
    # Setup
    batch_size = 32
    num_numerical = 10
    num_categorical = 5
    numerical_features_num = 8
    categorical_features_num = 7
    d_model = 16
    embedding_dim = 16
    num_heads = 4

    layer = MultiResolutionTabularAttention(
        num_heads=num_heads, d_model=d_model, embedding_dim=embedding_dim
    )

    # Create sample inputs
    numerical_features = tf.random.normal(
        (batch_size, num_numerical, numerical_features_num)
    )
    categorical_features = tf.random.normal(
        (batch_size, num_categorical, categorical_features_num)
    )

    # Process features
    numerical_output, categorical_output = layer(
        numerical_features, categorical_features, training=True
    )

    # Check shapes
    assert numerical_output.shape == (batch_size, num_numerical, d_model)
    assert categorical_output.shape == (batch_size, num_categorical, d_model)


def test_multi_resolution_attention_training_simple():
    """Test MultiResolutionTabularAttention behavior in training vs inference modes."""
    # Setup
    batch_size = 16
    num_numerical = 8
    num_categorical = 4
    numerical_dim = 24
    categorical_dim = 6
    d_model = 24
    embedding_dim = 12
    num_heads = 3
    dropout_rate = 0.5  # High dropout for visible effect

    layer = MultiResolutionTabularAttention(
        num_heads=num_heads,
        d_model=d_model,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
    )

    # Create inputs
    numerical_features = tf.random.normal((batch_size, num_numerical, numerical_dim))
    categorical_features = tf.random.normal(
        (batch_size, num_categorical, categorical_dim)
    )

    # Get outputs in training mode
    num_train, cat_train = layer(
        numerical_features, categorical_features, training=True
    )

    # Get outputs in inference mode
    num_infer, cat_infer = layer(
        numerical_features, categorical_features, training=False
    )

    # Check that outputs are different due to dropout
    assert not np.allclose(num_train.numpy(), num_infer.numpy())
    assert not np.allclose(cat_train.numpy(), cat_infer.numpy())


def test_multi_resolution_attention_cross_attention():
    """Test that cross-attention is working between numerical and categorical features."""

    # Setup
    batch_size = 8
    num_numerical = 4
    num_categorical = 2
    numerical_dim = 8
    categorical_dim = 4
    d_model = 8
    embedding_dim = 8
    num_heads = 2

    layer = MultiResolutionTabularAttention(
        num_heads=num_heads,
        d_model=d_model,
        embedding_dim=embedding_dim,
        dropout_rate=0.0,  # Disable dropout for deterministic testing
    )

    # Create numerical features
    numerical_features = tf.random.normal((batch_size, num_numerical, numerical_dim))

    # Create contrasting categorical patterns using string colors
    colors1 = tf.constant([["blue", "green"] for _ in range(batch_size)])  # Warm colors
    colors2 = tf.constant([["red", "yellow"] for _ in range(batch_size)])  # Cool colors

    # Convert strings to one-hot encodings
    all_colors = ["red", "blue", "green", "yellow"]
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            all_colors, tf.range(len(all_colors), dtype=tf.int64)
        ),
        default_value=-1,
    )

    categorical_pattern1 = tf.one_hot(table.lookup(colors1), categorical_dim)
    categorical_pattern2 = tf.one_hot(table.lookup(colors2), categorical_dim)

    # Process with contrasting categorical patterns
    num_output1, cat_output1 = layer(
        numerical_features, categorical_pattern1, training=False
    )
    num_output2, cat_output2 = layer(
        numerical_features, categorical_pattern2, training=False
    )

    # Check numerical outputs are different due to contrasting categorical patterns
    num_mean_diff = tf.reduce_mean(tf.abs(num_output1 - num_output2))
    assert (
        num_mean_diff > 1e-3
    ), "Numerical outputs are too similar - cross attention may not be working"

    # Check categorical outputs are different
    cat_mean_diff = tf.reduce_mean(tf.abs(cat_output1 - cat_output2))
    assert (
        cat_mean_diff > 1e-3
    ), "Categorical outputs are too similar - cross attention may not be working"

    # Check shapes are correct
    assert cat_output1.shape == cat_output2.shape
    assert cat_output1.shape[0] == batch_size
    assert cat_output1.shape[1] == num_categorical
    assert cat_output1.shape[2] == d_model

    # Check outputs are in reasonable range
    assert tf.reduce_all(
        tf.abs(cat_output1) < 10
    ), "Categorical outputs 1 have unexpectedly large values"
    assert tf.reduce_all(
        tf.abs(cat_output2) < 10
    ), "Categorical outputs 2 have unexpectedly large values"


def test_multi_resolution_attention_config():
    """Test configuration saving and loading."""
    original_layer = MultiResolutionTabularAttention(
        num_heads=4, d_model=32, embedding_dim=16, dropout_rate=0.2
    )

    config = original_layer.get_config()
    restored_layer = MultiResolutionTabularAttention.from_config(config)

    assert restored_layer.num_heads == original_layer.num_heads
    assert restored_layer.d_model == original_layer.d_model
    assert restored_layer.embedding_dim == original_layer.embedding_dim
    assert restored_layer.dropout_rate == original_layer.dropout_rate


def test_multi_resolution_attention_end_to_end_simple():
    """Test MultiResolutionTabularAttention in a simple end-to-end model."""
    # Setup
    batch_size = 16
    num_numerical = 100
    num_categorical = 10
    numerical_dim = 16
    categorical_dim = 4
    d_model = 8
    embedding_dim = 8
    num_heads = 2

    # Create a simple model
    numerical_inputs = keras.Input(shape=(num_numerical, numerical_dim))
    categorical_inputs = keras.Input(shape=(num_categorical, categorical_dim))

    attention_layer = MultiResolutionTabularAttention(
        num_heads=num_heads, d_model=d_model, embedding_dim=embedding_dim
    )

    num_output, cat_output = attention_layer(numerical_inputs, categorical_inputs)
    combined = keras.layers.Concatenate(axis=1)([num_output, cat_output])
    outputs = keras.layers.Dense(1)(combined)

    model = keras.Model(inputs=[numerical_inputs, categorical_inputs], outputs=outputs)

    # Compile model
    model.compile(optimizer="adam", loss="mse")

    # Create some data
    X_num = tf.random.normal((batch_size, num_numerical, numerical_dim))
    X_cat = tf.random.normal((batch_size, num_categorical, categorical_dim))
    y = tf.random.normal((batch_size, num_numerical + num_categorical, 1))

    # Train for one epoch
    history = model.fit([X_num, X_cat], y, epochs=1, verbose=0)

    # Check if loss was computed
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
