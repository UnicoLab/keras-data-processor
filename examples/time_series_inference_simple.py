#!/usr/bin/env python

"""
Simplified example showing how to use the TimeSeriesInferenceFormatter to prepare data for time series inference.
This demonstrates the core functionality without requiring actual model prediction.
"""

import numpy as np
import pandas as pd
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


def create_preprocessor(train_data):
    """Create a preprocessor with time series features."""
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

    # Create a preprocessor
    return PreprocessingModel(
        path_data=train_data,
        features_specs=features_specs,
    )

    # We don't need to build the preprocessor for this simplified example
    # We just need the features_specs and validation methods


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
        print("This is why we need the TimeSeriesInferenceFormatter!")


def example_with_historical_context(formatter, train_data):
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

    # Prepare the data with historical context (no tensor conversion for simplified example)
    formatted_data = formatter.prepare_inference_data(new_point, historical_data)

    print(f"Historical data shape: {historical_data.shape}")
    print(f"Formatted data has {len(formatted_data['sales'])} data points")

    # In a real example, we would now call preprocessor.predict(formatted_data)
    # but for this simplified example, we'll just show that the data is ready for prediction
    print(
        f"Data is ready for prediction! Last data point date: {formatted_data['date'][-1]}",
    )


def example_inspect_requirements(formatter):
    """Example showing how to inspect the requirements for time series inference."""
    print("\n=== Inspect Time Series Requirements ===")

    # Get detailed requirements
    requirements = formatter.min_history_requirements

    print("Requirements for each time series feature:")
    for feature, reqs in requirements.items():
        print(f"\nFeature: {feature}")
        for key, value in reqs.items():
            print(f"  {key}: {value}")

    # Get human-readable description
    print("\nHuman-readable description:")
    print(formatter.describe_requirements())


def example_multi_step_data_preparation(formatter, train_data):
    """Example showing how to prepare data for multi-step forecasting."""
    print("\n=== Multi-Step Forecast Data Preparation ===")

    # Use the last 14 days of Store_0 as history
    store_0_data = train_data[train_data["store_id"] == "Store_0"].iloc[-14:].copy()

    # Create future dates for forecasting (7 days)
    last_date = pd.to_datetime(store_0_data["date"].iloc[-1])
    future_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(7)
    ]

    print(f"Historical data: {len(store_0_data)} data points")
    print(f"Future dates to forecast: {future_dates}")

    # Prepare data for first prediction step
    next_row = {"date": future_dates[0], "store_id": "Store_0", "sales": np.nan}

    # Format data for first prediction step
    formatted_data = formatter.format_for_incremental_prediction(store_0_data, next_row)

    print("Data prepared for first prediction step.")
    print(f"Formatted data has {len(formatted_data['sales'])} data points")
    print(f"The last point (to predict) has date: {formatted_data['date'][-1]}")

    # In a real prediction scenario, this would be followed by:
    # 1. Make prediction for this step
    # 2. Add the prediction to history
    # 3. Prepare next step's data
    # 4. Repeat for all future dates


def main():
    """Main function to run the examples."""
    # Generate sample data
    train_data = generate_sample_data()
    print(f"Generated sample data with {len(train_data)} records")

    # Create the preprocessor (simplified without building the model)
    preprocessor = create_preprocessor(train_data)

    # Create the formatter
    formatter = TimeSeriesInferenceFormatter(preprocessor)

    # Example 1: Single-point inference (will fail, showing why we need the formatter)
    example_single_point_inference_failure(preprocessor, formatter)

    # Example 2: Inference with historical context
    example_with_historical_context(formatter, train_data)

    # Example 3: Inspect requirements
    example_inspect_requirements(formatter)

    # Example 4: Multi-step forecast data preparation
    example_multi_step_data_preparation(formatter, train_data)


if __name__ == "__main__":
    main()
