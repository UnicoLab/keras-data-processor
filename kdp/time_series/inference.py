import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Any

from kdp.features import FeatureType, TimeSeriesFeature
from kdp.inference.base import InferenceFormatter


class TimeSeriesInferenceFormatter(InferenceFormatter):
    """Specialized formatter for time series inference data.

    This class helps bridge the gap between raw time series data and the format required
    by the preprocessor during inference. It handles the unique requirements of time series
    features such as:

    1. Historical context requirements (lags, windows, etc.)
    2. Temporal ordering of data
    3. Proper grouping of time series
    4. Data validation and formatting

    For non-time series data, this formatter falls back to basic data conversion.
    """

    def __init__(self, preprocessor):
        """Initialize the TimeSeriesInferenceFormatter.

        Args:
            preprocessor: The trained preprocessor model to prepare data for
        """
        super().__init__(preprocessor)
        self.time_series_features = self._identify_time_series_features()
        self.min_history_requirements = self._calculate_min_history_requirements()

    def is_time_series_preprocessor(self) -> bool:
        """Check if the preprocessor has time series features.

        Returns:
            bool: True if time series features are present, False otherwise
        """
        return len(self.time_series_features) > 0

    def _identify_time_series_features(self) -> dict[str, TimeSeriesFeature]:
        """Identify all time series features in the preprocessor.

        Returns:
            Dict mapping feature names to TimeSeriesFeature objects
        """
        time_series_features = {}

        for name, feature in self.preprocessor.features_specs.items():
            if (
                hasattr(feature, "feature_type")
                and feature.feature_type == FeatureType.TIME_SERIES
            ):
                time_series_features[name] = feature

        return time_series_features

    def _calculate_min_history_requirements(self) -> dict[str, dict[str, Any]]:
        """Calculate minimum history requirements for each time series feature.

        Returns:
            Dict with feature names mapping to requirements dict
        """
        requirements = {}

        for feature_name, feature in self.time_series_features.items():
            feature_req = {
                "min_history": 1,  # Default minimum
                "sort_by": getattr(feature, "sort_by", None),
                "sort_ascending": getattr(feature, "sort_ascending", True),
                "group_by": getattr(feature, "group_by", None),
            }

            # Calculate minimum required history
            # Check lag features
            if hasattr(feature, "lag_config") and feature.lag_config:
                lags = feature.lag_config.get("lags", [])
                if lags:
                    feature_req["min_history"] = max(
                        feature_req["min_history"],
                        max(lags),
                    )

            # Check rolling statistics
            if (
                hasattr(feature, "rolling_stats_config")
                and feature.rolling_stats_config
            ):
                window_size = feature.rolling_stats_config.get("window_size", 1)
                feature_req["min_history"] = max(
                    feature_req["min_history"],
                    window_size,
                )

            # Check differencing
            if hasattr(feature, "differencing_config") and feature.differencing_config:
                order = feature.differencing_config.get("order", 1)
                feature_req["min_history"] = max(feature_req["min_history"], order)

            # Check moving averages
            if (
                hasattr(feature, "moving_average_config")
                and feature.moving_average_config
            ):
                periods = feature.moving_average_config.get("periods", [])
                if periods:
                    feature_req["min_history"] = max(
                        feature_req["min_history"],
                        max(periods),
                    )

            # Check wavelet transform
            if (
                hasattr(feature, "wavelet_transform_config")
                and feature.wavelet_transform_config
            ):
                levels = feature.wavelet_transform_config.get("levels", 3)
                feature_req["min_history"] = max(feature_req["min_history"], 2**levels)

            requirements[feature_name] = feature_req

        return requirements

    def prepare_inference_data(
        self,
        data: dict | pd.DataFrame,
        historical_data: dict | pd.DataFrame | None = None,
        fill_missing: bool = True,  # noqa: ARG002 - inert, see docstring
        to_tensors: bool = False,
    ) -> dict | dict[str, tf.Tensor]:
        """Prepare time series data for inference based on preprocessor requirements.

        Args:
            data: The new data to make predictions on
            historical_data: Optional historical data to provide context for time series
            fill_missing: Accepted for backwards compatibility and currently
                inert. This formatter never fabricates history: when the data is
                too short for the configured lookback it raises so the caller can
                supply real context. Missing *values* inside otherwise sufficient
                history are handled in-graph by `MissingValueHandlerLayer`, via
                the feature's `missing_value_config`.
            to_tensors: Whether to convert the output to TensorFlow tensors

        Returns:
            Dict with properly formatted data for inference, either as Python types or as TensorFlow tensors

        Raises:
            ValueError: If the data cannot be formatted to meet time series requirements
        """
        # Convert inputs to consistent format
        inference_data = self._convert_to_dict(data)

        # If no time series features, use basic formatting from parent class
        if not self.time_series_features:
            return super().prepare_inference_data(inference_data, to_tensors=to_tensors)

        # If we have time series features, we need proper formatting
        if historical_data is not None:
            historical_dict = self._convert_to_dict(historical_data)
            # Combine historical and new data
            combined_data = self._combine_historical_and_new(
                historical_dict,
                inference_data,
            )
        else:
            # Check if inference data itself has enough history
            self._check_inference_data_sufficiency(inference_data)
            combined_data = inference_data

        # Sort data by time for each group if needed
        formatted_data = self._sort_by_time_and_group(combined_data)

        # Final validation
        self.preprocessor._validate_time_series_inference_data(formatted_data)

        # Convert to tensors if requested
        if to_tensors:
            return self._convert_to_tensors(formatted_data)

        return formatted_data

    def _check_inference_data_sufficiency(self, data: dict) -> None:
        """Check if inference data itself has enough history for each feature.

        Args:
            data: Inference data dictionary

        Raises:
            ValueError: If data doesn't have sufficient history
        """
        for feature_name, requirements in self.min_history_requirements.items():
            if feature_name not in data:
                raise ValueError(
                    f"Time series feature '{feature_name}' is missing from input data",
                )

            # Check that data length is sufficient
            data_length = len(data[feature_name])
            if data_length < requirements["min_history"]:
                raise ValueError(
                    f"Time series feature '{feature_name}' requires at least "
                    f"{requirements['min_history']} data points, but only "
                    f"{data_length} were provided. Please provide historical data.",
                )

    def _combine_historical_and_new(self, historical: dict, new_data: dict) -> dict:
        """Combine historical and new data for time series features.

        Args:
            historical: Historical data dictionary
            new_data: New data dictionary for prediction

        Returns:
            Combined data dictionary
        """
        combined = {}

        # First, copy all keys from new_data
        for key in new_data:
            combined[key] = new_data[key]

        # Now add/combine historical data for time series features
        for feature_name in self.time_series_features:
            if feature_name in historical and feature_name in new_data:
                # Combine historical and new values
                combined[feature_name] = (
                    historical[feature_name] + new_data[feature_name]
                )

                # If we have group_by column, we need to combine that too
                group_by = self.min_history_requirements[feature_name]["group_by"]
                if group_by and group_by in historical and group_by in new_data:
                    combined[group_by] = historical[group_by] + new_data[group_by]

                # If we have sort_by column, we need to combine that too
                sort_by = self.min_history_requirements[feature_name]["sort_by"]
                if sort_by and sort_by in historical and sort_by in new_data:
                    combined[sort_by] = historical[sort_by] + new_data[sort_by]

        return combined

    def _sort_by_time_and_group(self, data: dict) -> dict:
        """Sort time series data by time and group.

        Args:
            data: Input data dictionary

        Returns:
            Sorted data dictionary
        """
        # Check if any time series feature requires sorting
        needs_sorting = False
        sort_columns = set()
        group_columns = set()

        for requirements in self.min_history_requirements.values():
            if requirements["sort_by"]:
                needs_sorting = True
                sort_columns.add(requirements["sort_by"])
            if requirements["group_by"]:
                group_columns.add(requirements["group_by"])

        if not needs_sorting:
            return data

        # Convert to DataFrame for easier sorting
        df = pd.DataFrame(data)
        sorted_dfs = []

        # Handle the case of multiple different sort and group requirements
        for feature_name, requirements in self.min_history_requirements.items():
            if requirements["sort_by"]:
                # Filter columns relevant to this feature
                relevant_cols = [feature_name, requirements["sort_by"]]
                if requirements["group_by"]:
                    relevant_cols.append(requirements["group_by"])

                # Ensure all required columns exist
                if all(col in df.columns for col in relevant_cols):
                    # Sort the data
                    feature_df = df[relevant_cols].sort_values(
                        by=[requirements["group_by"], requirements["sort_by"]]
                        if requirements["group_by"]
                        else requirements["sort_by"],
                        ascending=requirements["sort_ascending"],
                    )
                    sorted_dfs.append((feature_name, feature_df))

        # If we sorted any features, update the data dict
        if sorted_dfs:
            # Start with original data
            result_dict = data.copy()

            # Update with sorted data for each feature
            for feature_name, sorted_df in sorted_dfs:
                result_dict[feature_name] = sorted_df[feature_name].tolist()

                # Update sort and group columns if needed
                requirements = self.min_history_requirements[feature_name]
                if requirements["sort_by"]:
                    result_dict[requirements["sort_by"]] = sorted_df[
                        requirements["sort_by"]
                    ].tolist()
                if requirements["group_by"]:
                    result_dict[requirements["group_by"]] = sorted_df[
                        requirements["group_by"]
                    ].tolist()

            return result_dict

        return data

    def describe_requirements(self) -> str:
        """Generate a human-readable description of the requirements for time series inference.

        Returns:
            String with requirements description
        """
        if not self.time_series_features:
            return "No time series features detected. Data can be provided as single points."

        requirements = []
        requirements.append("Time Series Features Requirements:")

        for feature_name, reqs in self.min_history_requirements.items():
            feature_req = [f"  - {feature_name}:"]
            feature_req.append(
                f"    * Minimum history: {reqs['min_history']} data points",
            )

            if reqs["sort_by"]:
                feature_req.append(
                    f"    * Must be sorted by: {reqs['sort_by']} "
                    f"({'ascending' if reqs['sort_ascending'] else 'descending'})",
                )

            if reqs["group_by"]:
                feature_req.append(f"    * Must be grouped by: {reqs['group_by']}")

            requirements.extend(feature_req)

        return "\n".join(requirements)

    def format_for_incremental_prediction(
        self,
        current_history: dict,
        new_row: dict,
        to_tensors: bool = False,
    ) -> dict | dict[str, tf.Tensor]:
        """Format data for incremental time series prediction.

        This is useful for forecasting scenarios where each new prediction
        becomes part of the history for the next prediction.

        Args:
            current_history: Current historical data
            new_row: New data row to predict
            to_tensors: Whether to convert output to TensorFlow tensors

        Returns:
            Properly formatted data for making the prediction
        """
        # Ensure all inputs are in the right format
        history_dict = self._convert_to_dict(current_history)
        new_dict = self._convert_to_dict(new_row)

        # Combine and prepare the data
        return self.prepare_inference_data(
            new_dict,
            history_dict,
            to_tensors=to_tensors,
        )

    def generate_multi_step_forecast(
        self,
        history: dict,
        future_dates: list,
        group_id: str | None = None,
        steps: int | None = None,
    ) -> pd.DataFrame:
        """Generate a placeholder frame for multi-step forecasting.

        The returned frame carries one row per forecast step, with the sort
        column filled from ``future_dates`` and every time series feature set to
        NaN. Callers fill each row in turn with their model's prediction, so the
        row becomes part of the history for the following step.

        Args:
            history: Historical data dictionary or DataFrame. It is validated
                against the minimum history each configured feature needs.
            future_dates: List of dates for future predictions.
            group_id: Optional group identifier (e.g. store_id) if using grouped
                time series.
            steps: Number of steps to forecast. Defaults to every date in
                ``future_dates``.

        Returns:
            DataFrame with placeholder rows for each future step.

        Raises:
            ValueError: If the preprocessor has no time series features, if the
                feature has no sort column, if ``steps`` asks for more rows than
                ``future_dates`` provides, or if ``history`` is too short for the
                configured lookback.
        """
        if not self.time_series_features:
            raise ValueError("No time series features found in the preprocessor")

        future_dates = list(future_dates)
        if steps is not None:
            if steps > len(future_dates):
                raise ValueError(
                    f"Requested {steps} forecast steps but only "
                    f"{len(future_dates)} future date(s) were supplied.",
                )
            future_dates = future_dates[:steps]

        # The history has to satisfy the same lookback the preprocessor needs,
        # otherwise the frame built here cannot be fed back through it.
        self._check_history_requirements(self._convert_to_dict(history))

        # Get the first time series feature to determine sort and group columns
        feature_name = next(iter(self.time_series_features))
        requirements = self.min_history_requirements[feature_name]

        if not requirements["sort_by"]:
            raise ValueError(
                f"Time series feature '{feature_name}' has no sort_by column specified",
            )

        # Create a DataFrame of future dates
        sort_col = requirements["sort_by"]
        group_col = requirements["group_by"]

        future_data = {sort_col: future_dates}

        # Add group column if specified
        if group_col and group_id:
            future_data[group_col] = [group_id] * len(future_dates)

        # Add placeholder values for each time series feature
        for ts_feature in self.time_series_features:
            future_data[ts_feature] = [np.nan] * len(future_dates)

        # Convert to DataFrame and return
        return pd.DataFrame(future_data)
