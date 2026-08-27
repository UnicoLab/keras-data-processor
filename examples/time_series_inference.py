#!/usr/bin/env python

"""
Example script showing how to use the InferenceDataFormatter to prepare data for time series inference.
This demonstrates how to handle single-point inference, batch inference, forecasting, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from kdp.features import FeatureType, TimeSeriesFeature
from kdp.processor import PreprocessingModel
from kdp.time_series.inference import TimeSeriesInferenceFormatter


def generate_sample_data(num_stores=3, days_per_store=30, add_noise=True):
    """Generate sample time series data for multiple stores."""
    np.random.seed(42)

    all_data = []
    base_date = datetime(2023, 1, 1)

    for store_id in range(num_stores):
        # Each store has a different sales pattern
        if store_id == 0:
            # Store 0: Linear increase
            base_sales = 100
            growth = 2
        elif store_id == 1:
            # Store 1: Linear decrease
            base_sales = 300
            growth = -1.5
        else:
            # Store 2: Sinusoidal pattern
            base_sales = 200
            growth = 0

        for day in range(days_per_store):
            date = base_date + timedelta(days=day)

            # Calculate sales based on pattern
            if store_id < 2:
                # Linear pattern
                sales = base_sales + (day * growth)
            else:
                # Sinusoidal pattern
                sales = base_sales + 50 * np.sin(day * 0.2)

            # Add noise if requested
            if add_noise:
                sales += np.random.normal(0, 5)

            all_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": f"Store_{store_id}",
                    "sales": sales,
                },
            )

    return pd.DataFrame(all_data)


def train_preprocessor(train_data):
    """Train a preprocessor on the sample data."""
    # Define feature specs with time series features
    features_specs = {
        "sales": TimeSeriesFeature(
            name="sales",
            feature_type=FeatureType.TIME_SERIES,
            sort_by="date",
            sort_ascending=True,
            group_by="store_id",
            lag_config={"lags": [1, 7], "keep_original": True, "drop_na": False},
            rolling_stats_config={
                "window_size": 5,
                "statistics": ["mean"],
                "drop_na": False,
            },
        ),
        "date": FeatureType.DATE,
        "store_id": FeatureType.STRING_CATEGORICAL,
    }

    # Create a preprocessor with dict output to see results
    preprocessor = PreprocessingModel(
        path_data=train_data,
        features_specs=features_specs,
        output_mode="dict",
    )

    # Build the preprocessor
    preprocessor.build_preprocessor()

    return preprocessor


def example_single_point_inference_failure(preprocessor, formatter):
    """Example showing how single-point inference fails with time series features."""
    print("\n=== Single-Point Inference with Time Series Features ===")

    # Create a single data point
    single_point = {
        "date": "2023-02-01",
        "store_id": "Store_0",
        "sales": 150.0,
    }

    try:
        # This should fail because time series features need historical context
        formatter.prepare_inference_data(single_point)
    except ValueError as e:
        print(f"As expected, single-point inference failed: {e}")
        print("This is why we need the InferenceDataFormatter!")


def example_with_historical_context(preprocessor, formatter, train_data):
    """Example showing how to use historical context for inference."""
    print("\n=== Inference with Historical Context ===")

    # Get the requirements for inference
    print(formatter.describe_requirements())

    # Use the last 10 days of data as historical context
    historical_data = train_data.iloc[-10:].copy()

    # Create a new day to predict
    new_date = (
        pd.to_datetime(historical_data["date"].iloc[-1]) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    new_point = {
        "date": new_date,
        "store_id": "Store_0",  # Just predict for store 0
        "sales": np.nan,  # This is what we want to predict
    }

    # Prepare the data with historical context and convert to tensors
    formatted_data = formatter.prepare_inference_data(
        new_point,
        historical_data,
        to_tensors=True,  # Automatically convert to TensorFlow tensors
    )

    print(f"Historical data shape: {historical_data.shape}")
    print(f"Formatted data has {len(formatted_data['sales'])} data points")

    # Make the prediction (formatted_data already contains TensorFlow tensors)
    prediction = preprocessor.predict(formatted_data)

    if isinstance(prediction, dict):
        print(f"Predicted sales: {prediction['sales'][-1]}")
    else:
        print(f"Predicted sales: {prediction[-1]}")


def example_multi_step_forecast(preprocessor, formatter, train_data):
    """Example showing how to generate a multi-step forecast."""
    print("\n=== Multi-Step Forecasting ===")

    # Use the last 14 days of Store_0 as history
    store_0_data = train_data[train_data["store_id"] == "Store_0"].iloc[-14:].copy()

    # Create future dates for forecasting (7 days)
    last_date = pd.to_datetime(store_0_data["date"].iloc[-1])
    future_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(7)
    ]

    # Manually implement multi-step forecast
    forecast_rows = []
    history = store_0_data.copy()

    for future_date in future_dates:
        # Create next row to predict
        next_row = {
            "date": future_date,
            "store_id": "Store_0",
            "sales": np.nan,  # To be predicted
        }

        # Prepare data for prediction with historical context (automatically converts to tensors)
        formatted_data = formatter.format_for_incremental_prediction(
            history,
            next_row,
            to_tensors=True,  # Automatically convert to TensorFlow tensors
        )

        # Make prediction
        prediction = preprocessor.predict(formatted_data)

        # Extract the prediction value (last value in the sales array)
        predicted_value = (
            prediction["sales"][-1] if isinstance(prediction, dict) else prediction[-1]
        )

        # Create a result row for the forecast
        forecast_row = {
            "date": future_date,
            "store_id": "Store_0",
            "sales": predicted_value,
        }
        forecast_rows.append(forecast_row)

        # Add the prediction to history for the next step
        history = pd.concat([history, pd.DataFrame([forecast_row])], ignore_index=True)

    forecast = pd.DataFrame(forecast_rows)

    print(f"Generated a {len(forecast)} day forecast:")
    print(forecast)

    # Optional: Visualize the forecast
    try:
        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(
            pd.to_datetime(store_0_data["date"]),
            store_0_data["sales"],
            marker="o",
            linestyle="-",
            label="Historical",
        )

        # Plot forecast
        plt.plot(
            pd.to_datetime(forecast["date"]),
            forecast["sales"],
            marker="x",
            linestyle="--",
            color="red",
            label="Forecast",
        )

        plt.title("Sales Forecast")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig("forecast_example.png")
        print("Forecast visualization saved as 'forecast_example.png'")
    except Exception as e:
        print(f"Couldn't create visualization: {e}")


def example_batch_inference(preprocessor, formatter, train_data):
    """Example showing batch inference with new data points for multiple stores."""
    print("\n=== Batch Inference for Multiple Stores ===")

    # Use last 10 days as historical data
    historical_data = train_data.iloc[-30:].copy()

    # Create new data points for all 3 stores
    new_date = (
        pd.to_datetime(historical_data["date"].iloc[-1]) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    new_points = {
        "date": [new_date, new_date, new_date],
        "store_id": ["Store_0", "Store_1", "Store_2"],
        "sales": [np.nan, np.nan, np.nan],  # These are what we want to predict
    }

    # Prepare the data with historical context and convert to tensors
    formatted_data = formatter.prepare_inference_data(
        new_points,
        historical_data,
        to_tensors=True,  # Automatically convert to TensorFlow tensors
    )

    # Make the prediction (formatted_data already contains TensorFlow tensors)
    prediction = preprocessor.predict(formatted_data)

    if isinstance(prediction, dict):
        # Find the indices of the new points in the original (non-tensor) data
        store_indices = {"Store_0": [], "Store_1": [], "Store_2": []}

        # First convert back to regular Python lists for processing
        store_id_list = [
            s.decode("utf-8") if isinstance(s, bytes) else s
            for s in formatted_data["store_id"].numpy().tolist()
        ]

        for i, store in enumerate(store_id_list):
            if store in store_indices:
                store_indices[store].append(i)

        # Get the last index for each store
        for store in ["Store_0", "Store_1", "Store_2"]:
            if store_indices[store]:
                last_idx = store_indices[store][-1]
                print(
                    f"Predicted sales for {store}: {prediction['sales'][last_idx].numpy()}",
                )
    else:
        print(
            "Prediction result:",
            prediction[-3:],
        )  # Last 3 values are the predictions


def main():
    """Main function to run the examples."""
    # Generate sample data
    train_data = generate_sample_data()
    print(f"Generated sample data with {len(train_data)} records")

    # Train the preprocessor
    preprocessor = train_preprocessor(train_data)

    # Create the formatter
    formatter = TimeSeriesInferenceFormatter(preprocessor)

    # Example 1: Single-point inference (will fail, showing why we need the formatter)
    example_single_point_inference_failure(preprocessor, formatter)

    # Example 2: Inference with historical context
    example_with_historical_context(preprocessor, formatter, train_data)

    # Example 3: Multi-step forecast
    example_multi_step_forecast(preprocessor, formatter, train_data)

    # Example 4: Batch inference for multiple stores
    example_batch_inference(preprocessor, formatter, train_data)


if __name__ == "__main__":
    main()
