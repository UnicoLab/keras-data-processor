#!/usr/bin/env python
import keras
from pathlib import Path

from kdp.layers.time_series.lag_feature_layer import LagFeatureLayer
from kdp.layers.time_series.moving_average_layer import MovingAverageLayer
from kdp.layers.time_series.differencing_layer import DifferencingLayer
from kdp.layers.time_series.rolling_stats_layer import RollingStatsLayer

# Setup output directory
OUTPUT_DIR = Path("docs/features/imgs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_simple_model(name, layer):
    """Create a simple model with a single time series layer for diagram generation."""
    # Create a simple input
    inputs = keras.Input(shape=(1,), name="sales")

    # Apply normalization for better visualization
    norm = keras.layers.Normalization()(inputs)

    # Apply the time series layer
    outputs = layer(norm)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"time_series_{name}")

    # Generate diagram
    filename = f"{name}.png"
    output_path = OUTPUT_DIR / filename

    keras.utils.plot_model(
        model,
        to_file=str(output_path),
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )

    print(f"Generated time series diagram: {output_path}")
    return output_path


def create_combined_model(name, layers):
    """Create a model with multiple time series layers in sequence."""
    # Create a simple input
    inputs = keras.Input(shape=(1,), name="sales")

    # Apply normalization for better visualization
    x = keras.layers.Normalization()(inputs)

    # Apply all the time series layers sequentially
    for layer in layers:
        x = layer(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name=f"time_series_{name}")

    # Generate diagram
    filename = f"{name}.png"
    output_path = OUTPUT_DIR / filename

    keras.utils.plot_model(
        model,
        to_file=str(output_path),
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )

    print(f"Generated time series diagram: {output_path}")
    return output_path


def main():
    print("Generating time series diagrams...")

    # Basic time series
    create_simple_model(
        "basic_time_series",
        keras.layers.LayerNormalization(name="time_series_norm"),
    )

    # Time series with lag features
    create_simple_model(
        "time_series_with_lags",
        LagFeatureLayer(
            lag_indices=[1, 2, 3],
            keep_original=True,
            name="lag_features",
            drop_na=False,
        ),
    )

    # Time series with moving averages
    create_simple_model(
        "time_series_moving_average",
        MovingAverageLayer(
            periods=[3, 7, 14],
            keep_original=True,
            name="moving_average",
            drop_na=False,
        ),
    )

    # Time series with differencing
    create_simple_model(
        "time_series_differencing",
        DifferencingLayer(
            order=1,
            keep_original=True,
            name="differencing",
            drop_na=False,
        ),
    )

    # Time series with rolling statistics
    create_simple_model(
        "time_series_rolling_stats",
        RollingStatsLayer(
            window_size=7,
            statistics=["mean", "std"],
            keep_original=True,
            name="rolling_stats",
            drop_na=False,
        ),
    )

    # Time series with all features
    create_combined_model(
        "time_series_all_features",
        [
            LagFeatureLayer(
                lag_indices=[1, 2],
                keep_original=True,
                name="lag_features",
                drop_na=False,
            ),
            MovingAverageLayer(
                periods=[7],
                keep_original=True,
                name="moving_average",
                drop_na=False,
            ),
            DifferencingLayer(
                order=1,
                keep_original=True,
                name="differencing",
                drop_na=False,
            ),
            RollingStatsLayer(
                window_size=5,
                statistics=["mean"],
                keep_original=True,
                name="rolling_stats",
                drop_na=False,
            ),
        ],
    )

    print("All time series diagrams generated successfully!")


if __name__ == "__main__":
    main()
