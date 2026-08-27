# Time Series Inference Guide

This guide explains how to properly use time series features for inference in keras-data-processor, including handling the unique requirements and challenges they present.

## Understanding Time Series Inference Requirements

Time series features have special requirements that differ from other feature types:

1. **Historical Context**: Unlike standard features which can operate on single data points, time series features require historical context to compute transformations like lags, moving averages, etc.

2. **Temporal Ordering**: Data must be properly ordered chronologically for time series features to work correctly.

3. **Group Integrity**: When using group-based time series (like store-level sales), the data for each group must maintain its integrity.

4. **Minimum History Length**: Each transformation requires a specific minimum history length:
   - Lag features need at least `max(lags)` historical points
   - Rolling windows need at least `window_size` historical points
   - Differencing needs at least `order` historical points

## The TimeSeriesInferenceFormatter

The `TimeSeriesInferenceFormatter` class helps bridge the gap between raw time series data and the format required by the preprocessor during inference. It:

1. **Analyzes Requirements**: Examines your preprocessor to determine the exact requirements for each time series feature
2. **Validates Data**: Checks if your inference data meets these requirements
3. **Formats Data**: Combines historical and new data, sorts by time and group
4. **Converts to Tensors**: Automatically converts the data to TensorFlow tensors for prediction

### Basic Usage

```python
from kdp.time_series.inference import TimeSeriesInferenceFormatter

# Create a formatter with your trained preprocessor
formatter = TimeSeriesInferenceFormatter(preprocessor)

# Get human-readable description of requirements
print(formatter.describe_requirements())

# Prepare data for inference
formatted_data = formatter.prepare_inference_data(
    data=new_data,                 # The data point(s) to predict
    historical_data=historical_df, # Historical context for time series features
    to_tensors=True                # Convert output to TensorFlow tensors
)

# Make a prediction
prediction = preprocessor.predict(formatted_data)
```

### Understanding Requirements

To understand what your model needs for inference:

```python
# Check if the preprocessor has time series features
has_ts_features = formatter.is_time_series_preprocessor()

# Get detailed requirements
requirements = formatter.min_history_requirements

# For each time series feature
for feature, reqs in requirements.items():
    print(f"Feature: {feature}")
    print(f"  Minimum history: {reqs['min_history']} data points")
    print(f"  Sort by: {reqs['sort_by']}")
    print(f"  Group by: {reqs['group_by']}")
```

### Common Inference Scenarios

#### Single-Point Inference (Will Fail)

This will fail for time series features because they need historical context:

```python
single_point = {
    "date": "2023-02-01",
    "store_id": "Store_A",
    "sales": np.nan,  # What we want to predict
}

# This will raise a ValueError about insufficient history
formatter.prepare_inference_data(single_point)
```

#### Inference with Historical Context

```python
# Historical data (past 14 days)
historical_data = df.loc[df["date"] >= (prediction_date - pd.Timedelta(days=14))]

# New point to predict
new_point = {
    "date": prediction_date.strftime("%Y-%m-%d"),
    "store_id": "Store_A",
    "sales": np.nan,  # What we want to predict
}

# Prepare the data with historical context
formatted_data = formatter.prepare_inference_data(
    new_point,
    historical_data,
    to_tensors=True
)

# Make prediction
prediction = preprocessor.predict(formatted_data)
```

#### Multi-Step Forecasting

For multi-step forecasting, you need to:
1. Make the first prediction
2. Add that prediction to the history
3. Move forward and repeat

```python
# Start with historical data
history = historical_df.copy()
forecasts = []

# Generate 7-day forecast
for i in range(7):
    # Calculate the next date to predict
    next_date = (pd.to_datetime(history["date"].iloc[-1]) +
                pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Create the next point to predict
    next_point = {
        "date": next_date,
        "store_id": "Store_A",
        "sales": np.nan,  # To be predicted
    }

    # Format data for prediction
    formatted_data = formatter.format_for_incremental_prediction(
        history,
        next_point,
        to_tensors=True
    )

    # Make prediction
    prediction = preprocessor.predict(formatted_data)
    predicted_value = prediction["sales"][-1].numpy()

    # Record the forecast
    forecasts.append({
        "date": next_date,
        "store_id": "Store_A",
        "sales": predicted_value
    })

    # Add prediction to history for next step
    history = pd.concat([
        history,
        pd.DataFrame([{"date": next_date, "store_id": "Store_A", "sales": predicted_value}])
    ], ignore_index=True)
```

## Best Practices for Time Series Inference

1. **Provide Ample History**: Always provide more history than the minimum required - this improves prediction quality.

2. **Maintain Data Format**: Keep the same data format between training and inference:
   - Same column names and types
   - Same temporal granularity (daily, hourly, etc.)
   - Same grouping structure

3. **Handle Edge Cases**:
   - New groups that weren't in training data
   - Gaps in historical data
   - Irregularly sampled time series

4. **Use the Formatter Methods**:
   - `describe_requirements()` to understand what's needed
   - `prepare_inference_data()` for one-off predictions
   - `format_for_incremental_prediction()` for step-by-step forecasting

## Troubleshooting

Common errors and their solutions:

### "Feature requires historical context"
- **Problem**: You're trying to use a single data point with time series features
- **Solution**: Provide historical data as context

### "Requires at least X data points"
- **Problem**: You don't have enough history for the time series transformations
- **Solution**: Provide more historical points (at least the minimum required)

### "Requires grouping by X"
- **Problem**: Missing the column used for grouping in time series features
- **Solution**: Ensure your data includes all required grouping columns

### "Requires sorting by X"
- **Problem**: Missing the column used for sorting (usually a date/time column)
- **Solution**: Ensure your data includes all required sorting columns

## Advanced Usage

For more complex scenarios, the formatter provides additional options:

```python
# When you need more control over data preparation
formatted_data = formatter.prepare_inference_data(
    data=new_data,
    historical_data=historical_data,
    fill_missing=True,   # Inert; see the API docs. Missing values inside
                         # sufficient history are handled by
                         # missing_value_config on the feature itself.
    to_tensors=False     # Keep as Python/NumPy types for inspection
)

# Manual control of tensor conversion
tf_data = formatter._convert_to_tensors(formatted_data)

# Getting generated multi-step forecast
forecast_df = formatter.generate_multi_step_forecast(
    history=historical_data,
    future_dates=future_dates_list,
    group_id="Store_A",
    steps=7  # Generate 7 steps ahead
)
```

## Example Code

See the full examples in:
- `examples/time_series_inference_simple.py` for a simplified example
- `examples/time_series_inference.py` for a complete example with model prediction
