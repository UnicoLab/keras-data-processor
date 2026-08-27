"""
Example script for using DynamicPreprocessingPipeline with custom preprocessing layers.

This script demonstrates how to use the DynamicPreprocessingPipeline to create
a flexible pipeline of preprocessing layers, with customizable transformations.
"""
# ruff: noqa: E402

import numpy as np
import tensorflow as tf
import logging
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdp.dynamic_pipeline import DynamicPreprocessingPipeline
from kdp.layers_factory import PreprocessorLayerFactory

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Example 1: Basic Custom Layers
class ScalingLayer(tf.keras.layers.Layer):
    """Custom layer to scale numeric input by a factor."""

    def __init__(self, scaling_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor

    def call(self, inputs):
        return inputs * self.scaling_factor

    def get_config(self):
        config = super().get_config()
        config.update({"scaling_factor": self.scaling_factor})
        return config


class NormalizationLayer(tf.keras.layers.Layer):
    """Custom layer to normalize input to have mean 0 and std 1."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0)
        std = tf.math.reduce_std(inputs, axis=0)
        return (inputs - mean) / (std + 1e-5)  # Add epsilon for numerical stability

    def get_config(self):
        return super().get_config()


class LogTransformLayer(tf.keras.layers.Layer):
    """Custom layer to apply log transformation to input."""

    def __init__(self, offset=1.0, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def call(self, inputs):
        return tf.math.log(inputs + self.offset)

    def get_config(self):
        config = super().get_config()
        config.update({"offset": self.offset})
        return config


def example_1_basic_pipeline():
    """Example of a basic pipeline with custom layers."""
    print("\n=== Example 1: Basic Pipeline with Custom Layers ===")

    # Create custom layers
    scaling_layer = ScalingLayer(scaling_factor=3.0, name="scaling")
    normalization_layer = NormalizationLayer(name="normalization")

    # Create the pipeline
    pipeline = DynamicPreprocessingPipeline([scaling_layer, normalization_layer])

    # Create sample data
    data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    # Input features need to have keys matching layer names for processing
    dataset = tf.data.Dataset.from_tensor_slices(
        {"scaling": data, "normalization": data},
    ).batch(2)

    # Process the data
    processed_dataset = pipeline.process(dataset)

    # Display results
    print("Original data:", data.flatten())
    for batch in processed_dataset:
        print("Scaled data:", batch["scaling"].numpy().flatten())
        print("Normalized data:", batch["normalization"].numpy().flatten())
        break


# Example 2: Complex Pipeline with Dependencies
def example_2_dependency_pipeline():
    """Example of a pipeline with layers that depend on each other's outputs."""
    print("\n=== Example 2: Pipeline with Dependencies ===")

    # Create layers with dependencies
    # Layer 1: Apply scaling first
    scaling_layer = ScalingLayer(scaling_factor=2.0, name="scaling")
    # Layer 2: Apply log transform to scaled data
    log_layer = LogTransformLayer(name="log_transform")
    # Layer 3: Normalize the log-transformed data
    norm_layer = NormalizationLayer(name="normalization")

    # Create pipeline with dependency order
    pipeline = DynamicPreprocessingPipeline([scaling_layer, log_layer, norm_layer])

    # Create sample data
    data = np.array([[1.0], [5.0], [10.0], [50.0], [100.0]], dtype=np.float32)
    # Only need to provide the initial input
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "scaling": data,
        },
    ).batch(2)

    # Process the data
    processed_dataset = pipeline.process(dataset)

    # Display results
    print("Original data:", data.flatten())
    for batch in processed_dataset:
        print("Scaled data:", batch["scaling"].numpy().flatten())
        print("Log-transformed data:", batch["log_transform"].numpy().flatten())
        print("Normalized data:", batch["normalization"].numpy().flatten())
        break


# Example 3: Processing Multiple Features
class EncodingLayer(tf.keras.layers.Layer):
    """Custom layer for encoding categorical features using one-hot encoding."""

    def __init__(self, vocabulary=None, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary or []

    def build(self, input_shape):
        # Create a lookup table for categorical values
        self.lookup_table = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary,
            mask_token=None,
            num_oov_indices=1,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Convert categorical values to indices
        indices = self.lookup_table(inputs)
        # One-hot encode the indices
        return tf.one_hot(indices, depth=len(self.vocabulary) + 1)

    def get_config(self):
        config = super().get_config()
        config.update({"vocabulary": self.vocabulary})
        return config


def example_3_multiple_features():
    """Example of processing multiple different features in different pipelines."""
    print("\n=== Example 3: Processing Multiple Features ===")

    # Create layers for numeric features
    numeric_scaling = ScalingLayer(scaling_factor=2.0, name="numeric_scaling")
    numeric_pipeline = DynamicPreprocessingPipeline([numeric_scaling])

    # Create layers for categorical features
    categorical_encoding = EncodingLayer(
        vocabulary=["A", "B", "C"],
        name="categorical_encoding",
    )
    categorical_pipeline = DynamicPreprocessingPipeline([categorical_encoding])

    # Create sample data with different feature types
    numeric_data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    # Create string data with bytes_ type for TensorFlow compatibility
    categorical_data = np.array([["A"], ["B"], ["C"], ["D"]], dtype=np.object_)

    # Create separate datasets for each feature type
    numeric_dataset = tf.data.Dataset.from_tensor_slices(
        {"numeric_scaling": numeric_data},
    ).batch(2)

    categorical_dataset = tf.data.Dataset.from_tensor_slices(
        {"categorical_encoding": categorical_data},
    ).batch(2)

    # Process each dataset through its respective pipeline
    processed_numeric = numeric_pipeline.process(numeric_dataset)
    processed_categorical = categorical_pipeline.process(categorical_dataset)

    # Display results
    print("Original numeric data:", numeric_data.flatten())
    print("Original categorical data:", [s[0] for s in categorical_data])

    print("\nProcessed numeric data:")
    for batch in processed_numeric:
        print("Scaled numeric data:", batch["numeric_scaling"].numpy().flatten())
        break

    print("\nProcessed categorical data:")
    for batch in processed_categorical:
        print(
            "Encoded categorical data shape:",
            batch["categorical_encoding"].numpy().shape,
        )
        print("Encoded categorical data sample:")
        print(batch["categorical_encoding"].numpy())
        break


# Example 4: Integration with Keras Model
def example_4_keras_integration():
    """Example showing how to integrate DynamicPreprocessingPipeline with a Keras model."""
    print("\n=== Example 4: Integration with Keras Model ===")

    # Create preprocessing layers
    scaling_layer = ScalingLayer(scaling_factor=2.0, name="scaling")
    normalization_layer = NormalizationLayer(name="normalization")

    # Create preprocessing pipeline
    preprocess_pipeline = DynamicPreprocessingPipeline(
        [scaling_layer, normalization_layer],
    )

    # Create a simple Keras model
    inputs = tf.keras.Input(shape=(1,), name="model_input")
    dense1 = tf.keras.layers.Dense(10, activation="relu")(inputs)
    dense2 = tf.keras.layers.Dense(5, activation="relu")(dense1)
    outputs = tf.keras.layers.Dense(1)(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Create sample data
    data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    targets = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]], dtype=np.float32)

    # Create dataset and preprocess
    dataset = tf.data.Dataset.from_tensor_slices(
        {"scaling": data, "normalization": data, "y": targets},
    ).batch(2)

    processed_dataset = preprocess_pipeline.process(dataset)

    # Define a data generator that prepares data for the model
    def data_generator():
        for batch in processed_dataset:
            # Use the normalized data as model input
            x = batch["normalization"]
            y = batch["y"]
            yield x, y

    # Create a dataset from the generator
    train_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        ),
    )

    # Train the model (for demonstration, using just a few steps)
    model.fit(train_dataset, epochs=2, steps_per_epoch=3)

    # Make a prediction with preprocessing
    test_data = np.array([[6.0]], dtype=np.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        {"scaling": test_data, "normalization": test_data},
    ).batch(1)

    processed_test = preprocess_pipeline.process(test_dataset)

    for batch in processed_test:
        normalized_input = batch["normalization"]
        prediction = model.predict(normalized_input)
        print(f"Input: {test_data[0][0]}")
        print(f"Normalized input: {normalized_input.numpy()[0][0]}")
        print(f"Prediction: {prediction[0][0]}")


# Example 5: Normalize and Transform Pipeline
def example_5_normalize_transform():
    """Create a pipeline that normalizes data and then applies a log transform."""
    print("\n=== Example 5: Normalize and Transform Pipeline ===")

    # Generate random data - lognormal distribution (right-skewed)
    data = np.random.lognormal(mean=0, sigma=1, size=(1000, 1)).astype(np.float32)

    # Create a normalization layer
    normalize_layer = tf.keras.layers.Normalization(name="normalize")
    normalize_layer.adapt(data)

    # Create a log transform layer using our factory
    log_transform = PreprocessorLayerFactory.distribution_transform_layer(
        transform_type="log",
        name="log_transform",
    )

    # Create our pipeline with both layers
    pipeline = DynamicPreprocessingPipeline([normalize_layer, log_transform])

    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices({"normalize": data}).batch(32)

    # Process the data
    processed_data = pipeline.process(dataset)

    # Examine the results
    for batch in processed_data.take(1):
        original_mean = np.mean(data)
        transformed_mean = batch["log_transform"].numpy().mean()

        print(f"Original data mean: {original_mean:.4f}")
        print(f"Transformed data mean: {transformed_mean:.4f}")

        # Visualize the transformation
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(data, bins=50, alpha=0.7)
        plt.title("Original Data Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(batch["log_transform"].numpy(), bins=50, alpha=0.7)
        plt.title("Normalized + Log Transformed Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    return pipeline


if __name__ == "__main__":
    example_1_basic_pipeline()
    example_2_dependency_pipeline()
    example_3_multiple_features()
    example_4_keras_integration()
    example_5_normalize_transform()
