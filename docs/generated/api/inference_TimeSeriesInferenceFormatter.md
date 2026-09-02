# TimeSeriesInferenceFormatter

Specialized formatter for time series inference data.

This class helps bridge the gap between raw time series data and the format required
by the preprocessor during inference. It handles the unique requirements of time series
features such- **as**: 1. Historical context requirements (lags, windows, etc.)
2. Temporal ordering of data
3. Proper grouping of time series
4. Data validation and formatting

For non-time series data, this formatter falls back to basic data conversion.


## Constructor

```python
__init__(self, preprocessor)
```

Initialize the TimeSeriesInferenceFormatter.

### Parameters- **preprocessor**: The trained preprocessor model to prepare data for


---

## describe_requirements

```python
describe_requirements(self) -> str
```

Generate a human-readable description of the requirements for time series inference.

### Returns

    String with requirements description


---

## format_for_incremental_prediction

```python
format_for_incremental_prediction(self, current_history: dict, new_row: dict, to_tensors: bool = False) -> dict | dict[str, tensorflow.python.framework.tensor.Tensor]
```

Format data for incremental time series prediction.

This is useful for forecasting scenarios where each new prediction
becomes part of the history for the next prediction.

### Parameters- **current_history**: Current historical data
- **new_row**: New data row to predict
- **to_tensors**: Whether to convert output to TensorFlow tensors

### Returns

    Properly formatted data for making the prediction


---

## generate_multi_step_forecast

```python
generate_multi_step_forecast(self, history: dict, future_dates: list, group_id: str | None = None, steps: int | None = None) -> pandas.core.frame.DataFrame
```

Generate a placeholder frame for multi-step forecasting.

The returned frame carries one row per forecast step, with the sort
column filled from ``future_dates`` and every time series feature set to
NaN. Callers fill each row in turn with their model's prediction, so the
row becomes part of the history for the following step.

### Parameters- **history**: Historical data dictionary or DataFrame. It is validated
        against the minimum history each configured feature needs.
- **future_dates**: List of dates for future predictions.
- **group_id**: Optional group identifier (e.g. store_id) if using grouped
        time series.
- **steps**: Number of steps to forecast. Defaults to every date in
        ``future_dates``.

### Returns

    DataFrame with placeholder rows for each future step.

### Raises
- **ValueError**: If the preprocessor has no time series features, if the
        feature has no sort column, if ``steps`` asks for more rows than
        ``future_dates`` provides, or if ``history`` is too short for the
        configured lookback.


---

## is_time_series_preprocessor

```python
is_time_series_preprocessor(self) -> bool
```

Check if the preprocessor has time series features.

### Returns- **bool**: True if time series features are present, False otherwise


---

## prepare_inference_data

```python
prepare_inference_data(self, data: dict | pandas.core.frame.DataFrame, historical_data: dict | pandas.core.frame.DataFrame | None = None, fill_missing: bool = True, to_tensors: bool = False) -> dict | dict[str, tensorflow.python.framework.tensor.Tensor]
```

Prepare time series data for inference based on preprocessor requirements.

### Parameters- **data**: The new data to make predictions on
- **historical_data**: Optional historical data to provide context for time series
- **fill_missing**: Accepted for backwards compatibility and currently
        inert. This formatter never fabricates history: when the data is
        too short for the configured lookback it raises so the caller can
        supply real context. Missing *values* inside otherwise sufficient
        history are handled in-graph by `MissingValueHandlerLayer`, via
        the feature's `missing_value_config`.
- **to_tensors**: Whether to convert the output to TensorFlow tensors

### Returns

    Dict with properly formatted data for inference, either as Python types or as TensorFlow tensors

### Raises
- **ValueError**: If the data cannot be formatted to meet time series requirements


---

