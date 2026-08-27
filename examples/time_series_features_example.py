#!/usr/bin/env python

"""
Example of using the new time series feature layers in keras-data-processor.

This example demonstrates how to use the WaveletTransformLayer and TSFreshFeatureLayer
for extracting features from time series data.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Concatenate

from kdp.layers.time_series import (
    WaveletTransformLayer,
    TSFreshFeatureLayer,
    LagFeatureLayer,
)


def generate_sample_data(n_samples=1000, n_features=1):
    """Generate a sample time series dataset."""
    np.random.seed(42)

    # Time steps
    t = np.linspace(0, 10 * np.pi, n_samples)

    # Base sine wave with increasing frequency
    base_signal = np.sin(t * (1 + t / (10 * np.pi)))

    # Add trends and seasonality for complexity
    trend = 0.3 * t / (10 * np.pi)
    seasonality = 0.5 * np.sin(0.5 * t)

    # Create signal with noise
    signal = base_signal + trend + seasonality + np.random.normal(0, 0.2, n_samples)

    # Normalize
    signal = (signal - np.mean(signal)) / np.std(signal)

    # For multiple features, create variations
    if n_features > 1:
        signals = [signal]
        for i in range(1, n_features):
            # Create different variations with phase shifts and scaling
            variation = np.sin(t * (1 + t / (10 * np.pi) + i * 0.2)) + trend * (
                1.0 + 0.1 * i
            )
            variation = (variation - np.mean(variation)) / np.std(variation)
            signals.append(variation)
        signal = np.column_stack(signals)

    # Target variable: the next value in the series. Predicting one step ahead
    # means the final row has no target, so it is dropped from the inputs too --
    # otherwise X and y differ by one row and Keras rejects the pair.
    targets = signal[1:, 0] if n_features > 1 else signal[1:]
    features = signal[:-1]

    # Create test/train split
    train_size = int(0.8 * len(features))
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = targets[:train_size], targets[train_size:]

    return X_train, y_train, X_test, y_test


def build_model_with_feature_layers(input_shape):
    """Build a model that uses various time series feature layers."""
    inputs = Input(shape=input_shape)

    # 1. Extract wavelet transform features
    wavelet_features = WaveletTransformLayer(
        levels=3,
        window_sizes=[4, 8, 16],
        flatten_output=True,
    )(inputs)

    # 2. Extract statistical features using TSFreshFeatureLayer
    tsfresh_features = TSFreshFeatureLayer(
        features=["mean", "std", "min", "max", "median", "skewness", "kurtosis"],
        normalize=True,
    )(inputs)

    # 3. Extract lag features for temporal patterns
    lag_features = LagFeatureLayer(
        lag_indices=[1, 2, 3, 5, 7, 14, 21],
        drop_na=False,  # We'll get zeros for missing values
    )(inputs)

    # Combine all features
    combined_features = Concatenate()(
        [wavelet_features, tsfresh_features, lag_features],
    )

    # Dense layers for prediction
    x = Dense(64, activation="relu")(combined_features)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def main():
    """Run the example."""
    # Generate sample data
    X_train, y_train, X_test, y_test = generate_sample_data(
        n_samples=1000,
        n_features=2,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Reshape for the model (add batch dimension if not already present)
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

    # Build model
    model = build_model_with_feature_layers(input_shape=(X_train.shape[1],))

    # Print model summary
    model.summary()

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    plt.tight_layout()
    plt.savefig("time_series_features_training.png")
    print("Training plot saved as 'time_series_features_training.png'")

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Make predictions and plot
    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title("Time Series Prediction with Feature Layers")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("time_series_features_prediction.png")
    print("Prediction plot saved as 'time_series_features_prediction.png'")


if __name__ == "__main__":
    main()
