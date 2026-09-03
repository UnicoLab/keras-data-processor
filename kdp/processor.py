"""Preprocessor Module for Keras Data Processor.

This module provides a preprocessing model that can handle various types of features
and transformations for machine learning pipelines.
"""
import os
import time
import gc
import tensorflow as tf
import keras
from collections import OrderedDict
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import wraps
from typing import Any
from pathlib import Path
import json
import numpy as np

from loguru import logger

from kdp.layers.global_numerical_embedding_layer import GlobalNumericalEmbedding
from kdp.features import (
    CategoricalFeature,
    FeatureSpaceConverter,
    CategoryEncodingOptions,
    Feature,
    FeatureType,
    NumericalFeature,
    PassthroughFeature,
    # Re-exported: `kdp.processor` used to define a second class of this name
    # whose members were different values, so an import from the wrong module
    # produced an option that silently did nothing.
    TextVectorizerOutputOptions,
)
from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor
from kdp.stats import DatasetStatistics
from kdp.moe import (
    FeatureMoE,
    PadFeatureLayer,
    PerFeatureDense,
    StackFeaturesLayer,
    UnstackLayer,
)


def _to_tensor_mapping(data: dict) -> dict:
    """Convert a mapping of feature values to tensors where possible.

    Keras rejects NumPy string arrays outright ("Invalid dtype: str32"), so any
    feature dictionary holding raw Python or NumPy values has to be converted
    before it reaches the model. Values that cannot be converted are passed
    through untouched so the model raises its own, clearer error.

    Args:
        data: Mapping of feature names to values.

    Returns:
        The same mapping with convertible values turned into tensors.
    """
    converted = {}
    for key, value in data.items():
        if isinstance(value, tf.Tensor) or tf.is_tensor(value):
            converted[key] = value
            continue
        try:
            converted[key] = tf.convert_to_tensor(value)
        except (ValueError, TypeError, tf.errors.OpError):
            converted[key] = value
    return converted


class CallableDict(dict):
    """A dictionary that can be called like a function.

    This class extends the built-in dict class and adds a __call__ method,
    which allows it to be used as a callable object. This is particularly useful
    for making the result of build_preprocessor callable, so users can do
    preprocessor(test_input) instead of preprocessor["model"](test_input).

    When called, it will try to invoke the "model" key if it exists, passing all
    arguments and keyword arguments to that function.
    """

    def __call__(self, *args: Any, **kwargs):
        """Call the model function with the given arguments.

        Args:
            *args: Arguments to pass to the model function.
            **kwargs: Keyword arguments to pass to the model function.

        Returns:
            The result of calling the model function.

        Raises:
            KeyError: If the dictionary doesn't have a "model" key.
            TypeError: If the "model" key is not callable.
        """
        if "model" not in self:
            raise KeyError("This dictionary doesn't have a 'model' key")

        if not callable(self["model"]):
            raise TypeError("The 'model' key is not callable")

        # If the input is a dictionary, check if values need to be converted to tensors
        if len(args) > 0 and isinstance(args[0], dict):
            return self["model"](_to_tensor_mapping(args[0]), *args[1:], **kwargs)

        return self["model"](*args, **kwargs)


class OutputModeOptions(str, Enum):
    """Output mode options for the preprocessor model."""

    CONCAT = "concat"
    DICT = "dict"


class TransformerBlockPlacementOptions(str, Enum):
    """Placement options for transformer blocks."""

    CATEGORICAL = "categorical"
    ALL_FEATURES = "all_features"


class TabularAttentionPlacementOptions(str, Enum):
    """Placement options for tabular attention."""

    NONE = "none"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    ALL_FEATURES = "all_features"
    MULTI_RESOLUTION = "multi_resolution"


class FeatureSelectionPlacementOptions(str, Enum):
    """Placement options for feature selection."""

    NONE = "none"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATE = "date"
    ALL_FEATURES = "all_features"


def _validate_option(value, options, param_name: str) -> str:
    """Normalise a placement/mode option and reject anything unrecognised.

    These are compared against the option constants by string equality, so a
    value that merely looks right -- "all" instead of "all_features" -- matched
    nothing and the feature it controls was silently skipped rather than
    raising.

    Args:
        value: The caller-supplied option, in any casing.
        options: The class holding the valid string constants.
        param_name: Name of the parameter, used in the error message.

    Returns:
        str: The canonical option value.

    Raises:
        ValueError: If the value is not one of the valid options.
    """
    valid = {
        v.value if isinstance(v, Enum) else v
        for k, v in vars(options).items()
        if not k.startswith("_") and isinstance(v, str | Enum)
    }
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        for option in valid:
            if value.lower() == option.lower():
                return option
    raise ValueError(
        f"Unsupported {param_name} {value!r}. Expected one of {sorted(valid)}.",
    )


class PreprocessingModel:
    def __init__(
        self,
        features_stats: dict[str, Any] = None,
        path_data: str = None,
        batch_size: int = 50_000,
        feature_crosses: list[tuple[str, str, int]] = None,
        features_stats_path: str = None,
        output_mode: str = OutputModeOptions.CONCAT.value,
        overwrite_stats: bool = False,
        log_to_file: bool = False,
        features_specs: dict[str, FeatureType | str] = None,
        transfo_nr_blocks: int = None,
        transfo_nr_heads: int = 3,
        transfo_ff_units: int = 16,
        transfo_dropout_rate: float = 0.25,
        transfo_placement: str = TransformerBlockPlacementOptions.CATEGORICAL.value,
        tabular_attention: bool = False,
        tabular_attention_heads: int = 4,
        tabular_attention_dim: int = 64,
        tabular_attention_dropout: float = 0.1,
        tabular_attention_placement: str = TabularAttentionPlacementOptions.ALL_FEATURES.value,
        tabular_attention_embedding_dim: int = 32,
        use_caching: bool = True,
        feature_selection_placement: str = FeatureSelectionPlacementOptions.NONE.value,
        use_distribution_aware: bool = False,
        distribution_aware_bins: int = 1000,
        feature_selection_units: int = 32,
        feature_selection_dropout: float = 0.2,
        use_advanced_numerical_embedding: bool = False,
        embedding_dim: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: float = -3.0,
        init_max: float = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_global_numerical_embedding: bool = False,
        global_embedding_dim: int = 8,
        global_mlp_hidden_units: int = 16,
        global_num_bins: int = 10,
        global_init_min: float = -3.0,
        global_init_max: float = 3.0,
        global_dropout_rate: float = 0.1,
        global_use_batch_norm: bool = True,
        global_pooling: str = "average",
        use_feature_moe: bool = False,
        feature_moe_num_experts: int = 4,
        feature_moe_expert_dim: int = 64,
        feature_moe_hidden_dims: list[int] = None,
        feature_moe_routing: str = "learned",
        feature_moe_sparsity: int = 2,
        feature_moe_assignments: dict[str, int] = None,
        feature_moe_dropout: float = 0.1,
        feature_moe_freeze_experts: bool = False,
        feature_moe_use_residual: bool = True,
        include_passthrough_in_output: bool = True,
        name: str = "preprocessor",
    ) -> None:
        """Initialize a preprocessing model.

        Args:
            features_stats (dict[str, Any]): A dictionary containing the statistics of the features.
            path_data (str): The path to the data from which estimate the statistics.
            batch_size (int): The batch size for the data iteration for stats estimation.
            feature_crosses (list[tuple[str, str, int]]):
                A list of tuples containing the names of the features to be crossed,
                and nr_bins to be used for hashing.
            features_stats_path (str): The path where to save/load features statistics.
            output_mode (str): The output mode of the model (concat | dict).
            overwrite_stats (bool): A boolean indicating whether to overwrite the statistics.
            log_to_file (bool): A boolean indicating whether to log to a file.
            features_specs (dict[str, FeatureType | str]): A dictionary containing the features and their types.
            transfo_nr_blocks (int): The number of transformer blocks for the transformer block
                (default=None, transformer block is disabled).
            transfo_nr_heads (int): The number of heads for the transformer block (categorical variables).
            transfo_ff_units (int): The number of feed forward units for the transformer
            transfo_dropout_rate (float): The dropout rate for the transformer block (default=0.25).
            transfo_placement (str): The placement of the transformer block (categorical | all_features).
            tabular_attention (bool): Whether to use tabular attention (default=False).
            tabular_attention_heads (int): Number of attention heads for tabular attention.
            tabular_attention_dim (int): Dimension of the attention model.
            tabular_attention_dropout (float): Dropout rate for tabular attention.
            tabular_attention_placement (str): Where to apply tabular attention (none|numeric|categorical|all_features).
            tabular_attention_embedding_dim (int): Dimension of the embedding for multi-resolution attention.
            use_caching (bool): Whether to cache preprocessed features (default=True).
            feature_selection_placement (str): Where to apply feature selection (none|numeric|categorical|all_features).
            feature_selection_units (int): Number of units for feature selection.
            feature_selection_dropout (float): Dropout rate for feature selection.
            use_distribution_aware (bool): Whether to use distribution-aware encoding for features.
            distribution_aware_bins (int): Number of bins to use for distribution-aware encoding.
            use_advanced_numerical_embedding (bool): Whether to use advanced numerical embedding.
            embedding_dim (int): Dimension of the embedding for advanced numerical embedding.
            mlp_hidden_units (int): Number of units for the MLP in advanced numerical embedding.
            num_bins (int): Number of bins for discretization in advanced numerical embedding.
            init_min (float): Minimum value for the embedding in advanced numerical embedding.
            init_max (float): Maximum value for the embedding in advanced numerical embedding.
            dropout_rate (float): Dropout rate for advanced numerical embedding.
            use_batch_norm (bool): Whether advanced numerical embedding applies batch normalization.
            use_global_numerical_embedding (bool): Whether to embed all numeric features jointly,
                in addition to (or instead of) per-feature embeddings.
            global_embedding_dim (int): Dimension of the global numerical embedding.
            global_mlp_hidden_units (int): Number of units for the MLP in the global numerical embedding.
            global_num_bins (int): Number of bins for discretization in the global numerical embedding.
            global_init_min (float): Minimum value for the global numerical embedding.
            global_init_max (float): Maximum value for the global numerical embedding.
            global_dropout_rate (float): Dropout rate for the global numerical embedding.
            global_use_batch_norm (bool): Whether the global numerical embedding applies batch normalization.
            global_pooling (str): Pooling applied over the global embedding (average | max).
            use_feature_moe (bool): Whether to enable the feature-wise mixture of experts.
            feature_moe_num_experts (int): Number of experts in the mixture.
            feature_moe_expert_dim (int): Output dimension of each expert.
            feature_moe_hidden_dims (list[int]): Hidden layer sizes inside each expert.
            feature_moe_routing (str): How features are routed to experts (learned | predefined).
            feature_moe_sparsity (int): Number of experts each feature is routed to when routing is sparse.
            feature_moe_assignments (dict[str, int]): Explicit feature-to-expert assignments,
                used when routing is "predefined".
            feature_moe_dropout (float): Dropout rate applied inside the mixture of experts.
            feature_moe_freeze_experts (bool): Whether expert weights are frozen during training.
            feature_moe_use_residual (bool): Whether to add a residual connection around the mixture.
            name (str): Name given to the built Keras model. Keras requires
                operation names to be unique within a graph, so give each
                preprocessor its own name when combining several in one model
                (a two-tower recommender, for example).
            include_passthrough_in_output (bool): Whether passthrough features appear in the model output.
        """
        self.path_data = path_data
        self.batch_size = batch_size or 50_000
        self.features_stats = features_stats or {}
        self.features_specs = features_specs or {}
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.feature_crosses = feature_crosses or []
        self.output_mode = output_mode
        self.overwrite_stats = overwrite_stats
        self.use_caching = use_caching

        # transformer blocks control
        self.transfo_nr_blocks = transfo_nr_blocks
        self.transfo_nr_heads = transfo_nr_heads
        self.transfo_ff_units = transfo_ff_units
        self.transfo_dropout_rate = transfo_dropout_rate
        self.transfo_placement = transfo_placement

        # tabular attention control
        self.tabular_attention = tabular_attention
        self.tabular_attention_heads = tabular_attention_heads
        self.tabular_attention_dim = tabular_attention_dim
        self.tabular_attention_dropout = tabular_attention_dropout
        self.tabular_attention_placement = _validate_option(
            tabular_attention_placement,
            TabularAttentionPlacementOptions,
            "tabular_attention_placement",
        )
        self.tabular_attention_embedding_dim = tabular_attention_embedding_dim

        # feature selection control
        self.feature_selection_placement = _validate_option(
            feature_selection_placement,
            FeatureSelectionPlacementOptions,
            "feature_selection_placement",
        )
        self.feature_selection_units = feature_selection_units
        self.use_distribution_aware = use_distribution_aware
        self.distribution_aware_bins = distribution_aware_bins
        self.feature_selection_dropout = feature_selection_dropout

        # advanced numerical embedding control
        self.use_advanced_numerical_embedding = use_advanced_numerical_embedding
        self.embedding_dim = embedding_dim
        self.mlp_hidden_units = mlp_hidden_units
        self.num_bins = num_bins
        self.init_min = init_min
        self.init_max = init_max
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # advanced global numerical embedding control
        self.use_global_numerical_embedding = use_global_numerical_embedding
        self.global_embedding_dim = global_embedding_dim
        self.global_mlp_hidden_units = global_mlp_hidden_units
        self.global_num_bins = global_num_bins
        self.global_init_min = global_init_min
        self.global_init_max = global_init_max
        self.global_dropout_rate = global_dropout_rate
        self.global_use_batch_norm = global_use_batch_norm
        self.global_pooling = global_pooling

        # MoE control
        self.use_feature_moe = use_feature_moe
        self.feature_moe_num_experts = feature_moe_num_experts
        self.feature_moe_expert_dim = feature_moe_expert_dim
        self.feature_moe_hidden_dims = feature_moe_hidden_dims
        self.feature_moe_routing = feature_moe_routing
        self.feature_moe_sparsity = feature_moe_sparsity
        self.feature_moe_assignments = feature_moe_assignments
        self.feature_moe_dropout = feature_moe_dropout
        self.feature_moe_freeze_experts = feature_moe_freeze_experts
        self.feature_moe_use_residual = feature_moe_use_residual

        # Passthrough features control
        self.include_passthrough_in_output = include_passthrough_in_output
        self.model_name = name

        # Initialize feature type lists
        self.numeric_features = []
        self.categorical_features = []
        self.text_features = []
        self.date_features = []
        self.passthrough_features = []
        self.time_series_features = []  # Initialize time_series_features list

        # PLACEHOLDERS
        self.preprocessors = {}
        self.inputs = {}
        self.signature = {}
        self.outputs = {}  # Final outputs for DICT mode
        self.processed_features = {}  # All processed features before final output
        self.passthrough_outputs = {}  # Passthrough features (unprocessed)
        self.concat_all = None  # Final concatenated output for CONCAT mode
        self.model = None  # Set by build_preprocessor
        self._preprocessed_cache = {} if use_caching else None

        # `_monitor_performance` measured the time and GPU memory of every step
        # it wrapped and then wrote them to a debug log, so `get_timing_metrics`
        # and `get_memory_usage` -- both of which the documentation shows -- had
        # nothing to read and did not exist. The measurements are kept here now.
        self._step_seconds: dict[str, float] = {}
        self._step_memory_bytes: dict[str, int] = {}

        if log_to_file:
            logger.info("Logging to file enabled")
            logger.add("PreprocessModel.log")

        # formatting features specs info
        self._init_features_specs(features_specs=features_specs)

        # initializing stats
        self._init_stats()

    def _monitor_performance(func: Callable) -> Callable:
        """Decorator to monitor the performance of a function.

        Args:
            func: Function to monitor

        Returns:
            Wrapped function with performance monitoring
        """

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            """Wrapper function that adds performance monitoring.

            Args:
                self: Instance of the class
                *args: Variable positional arguments
                **kwargs: Variable keyword arguments

            Returns:
                Result of the wrapped function
            """
            try:
                # Record start time
                start_time = time.time()

                # Check for GPU availability using modern TensorFlow API
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    try:
                        # Get initial GPU memory info if available
                        start_memory = tf.config.experimental.get_memory_info("GPU:0")[
                            "current"
                        ]
                    except (ValueError, tf.errors.NotFoundError):
                        # Handle case where GPU memory info is not available
                        start_memory = 0
                else:
                    start_memory = 0

                # Execute the function
                result = func(self, *args, **kwargs)

                # Record end time
                end_time = time.time()

                # Get final GPU memory if available
                if gpus:
                    try:
                        end_memory = tf.config.experimental.get_memory_info("GPU:0")[
                            "current"
                        ]
                    except (ValueError, tf.errors.NotFoundError):
                        end_memory = start_memory
                else:
                    end_memory = start_memory

                # Calculate metrics
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory

                # Keep the measurements as well as logging them; the getters
                # below are the documented way to read them back.
                self._step_seconds[func.__name__] = (
                    self._step_seconds.get(func.__name__, 0.0) + execution_time
                )
                self._step_memory_bytes[func.__name__] = max(
                    self._step_memory_bytes.get(func.__name__, 0),
                    memory_used,
                )

                # Log performance metrics
                logger.debug(
                    f"Function {func.__name__} executed in {execution_time:.2f} seconds. "
                    f"Memory used: {memory_used / (1024 * 1024):.2f} MB",
                )

                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise

        return wrapper

    def _init_features_specs(
        self,
        features_specs: dict[str, FeatureType | str],
    ) -> None:
        """Initialize the features specifications for the model.

        Args:
            features_specs (dict): A dictionary containing the features and their types.
        """
        if features_specs:
            logger.info("Normalizing Feature Space using FeatureSpaceConverter")
            logger.debug(f"Features specs: {features_specs}")
            fsc = FeatureSpaceConverter()
            self.features_specs = fsc._init_features_specs(
                features_specs=features_specs,
            )
            logger.debug(f"Features specs normalized: {self.features_specs}")
            self.numeric_features = fsc.numeric_features
            self.categorical_features = fsc.categorical_features
            self.text_features = fsc.text_features
            self.date_features = fsc.date_features
            self.passthrough_features = fsc.passthrough_features
            self.time_series_features = fsc.time_series_features

    def _init_stats(self) -> None:
        """Initialize the statistics for the model.

        Note:
            Initializing Data Stats object
            we only need numeric and cat features stats for layers
            crosses and numeric do not need layers init
        """
        if not self.features_stats:
            logger.info("No features stats provided, trying to load local file ")
            self.stats_instance = DatasetStatistics(
                path_data=self.path_data,
                features_specs=self.features_specs,
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                text_features=self.text_features,
                date_features=self.date_features,
                time_series_features=self.time_series_features,
                features_stats_path=self.features_stats_path,
                overwrite_stats=self.overwrite_stats,
                batch_size=self.batch_size,
            )
            self.features_stats = self.stats_instance._load_stats()

    def _add_input_column(self, feature_name: str, dtype: tf.dtypes.DType) -> None:
        """Add an input column to the model.

        Args:
            feature_name: The name of the feature.
            dtype: TensorFlow data type for the feature values.

        Note:
            Creates a Keras Input layer with shape (1,) and adds it to self.inputs
        """
        logger.debug(f"Adding {feature_name = }, {dtype =} to the input columns")
        self.inputs[feature_name] = keras.Input(
            shape=(1,),
            name=feature_name,
            dtype=dtype,
        )

    @_monitor_performance
    def _add_input_signature(self, feature_name: str, dtype: tf.dtypes.DType) -> None:
        """Add an input signature to the model.

        Args:
            feature_name: The name of the feature.
            dtype: TensorFlow data type for the feature values.

        Note:
            Creates a TensorSpec with shape (None, 1) and adds it to self.signature
        """
        logger.debug(f"Adding {feature_name = }, {dtype =} to the input signature")
        self.signature[feature_name] = tf.TensorSpec(
            shape=(None, 1),
            dtype=dtype,
            name=feature_name,
        )

    @_monitor_performance
    def _add_custom_steps(
        self,
        preprocessor: FeaturePreprocessor,
        feature: FeatureType,
        feature_name: str,
    ) -> FeaturePreprocessor:
        """Add custom preprocessing steps to the pipeline.

        Args:
            preprocessor: The preprocessor object.
            feature: The feature object.
            feature_name: The name of the feature.

        Returns:
            FeaturePreprocessor: The preprocessor object with the custom steps added.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]
        for position, preprocessor_step in enumerate(feature.preprocessors):
            logger.info(
                f"Adding custom {preprocessor_step} for {feature_name}, {_feature.kwargs}",
            )
            # A step may be given three ways, and the docs lead with the first:
            # a layer name, a layer class, or an already-built layer. Only the
            # class form has `__name__`, so reading it directly crashed on the
            # documented spelling.
            if isinstance(preprocessor_step, str):
                step_name = preprocessor_step
            elif isinstance(preprocessor_step, keras.layers.Layer):
                step_name = preprocessor_step.name
            else:
                step_name = getattr(
                    preprocessor_step,
                    "__name__",
                    type(preprocessor_step).__name__,
                )

            # Layer names have to be unique inside a model, and the same layer
            # type may legitimately appear twice in one pipeline.
            layer_name = f"{step_name}_{feature_name}_{position}"

            if isinstance(preprocessor_step, keras.layers.Layer):
                preprocessor.add_processing_step(
                    layer_creator=lambda _layer=preprocessor_step, **_: _layer,
                    name=layer_name,
                )
            else:
                preprocessor.add_processing_step(
                    layer_class=preprocessor_step,
                    name=layer_name,
                    **_feature.kwargs,
                )
        return preprocessor

    def _process_feature_batch(
        self,
        batch: list[tuple[str, dict]],
        feature_type: str,
    ) -> None:
        """Process a batch of features in parallel.

        Args:
            batch: List of (feature_name, stats) tuples to process
            feature_type: Type of features ('numeric', 'categorical', 'text', 'date', 'passthrough', 'time_series')
        """
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for feature_name, stats in batch:
                if feature_type == "numeric":
                    future = executor.submit(
                        self._add_pipeline_numeric,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        stats=stats,
                    )
                elif feature_type == "categorical":
                    future = executor.submit(
                        self._add_pipeline_categorical,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        stats=stats,
                    )
                elif feature_type == "text":
                    future = executor.submit(
                        self._add_pipeline_text,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        stats=stats,
                    )
                elif feature_type == "date":
                    future = executor.submit(
                        self._add_pipeline_date,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                    )
                elif feature_type == "time_series":
                    future = executor.submit(
                        self._add_pipeline_time_series,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        feature=self.features_specs.get(feature_name),
                    )
                elif feature_type == "passthrough":
                    future = executor.submit(
                        self._add_pipeline_passthrough,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                    )
                futures.append((feature_name, future))

            # Wait for all futures to complete
            for feature_name, future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing feature {feature_name}: {str(e)}")
                    raise

    def _create_feature_preprocessor(
        self,
        feature_name: str,
        feature: Feature,
        preprocessor: FeaturePreprocessor,
    ) -> FeaturePreprocessor:
        """Create feature-specific preprocessor with custom steps if defined.

        This method handles the common pattern of checking for custom preprocessors
        and adding them to the pipeline if they exist.

        Args:
            feature_name: Name of the feature being processed
            feature: Feature object containing specifications
            preprocessor: The preprocessor to augment

        Returns:
            FeaturePreprocessor with custom steps added if they exist
        """
        # Check if feature has specific preprocessing steps defined
        if hasattr(feature, "preprocessors") and feature.preprocessors:
            logger.info(
                f"Custom Preprocessors detected for {feature_name}: {feature.preprocessors}",
            )
            return self._add_custom_steps(
                preprocessor=preprocessor,
                feature=feature,
                feature_name=feature_name,
            )
        return preprocessor

    def _apply_feature_selection(
        self,
        feature_name: str,
        output_pipeline: tf.Tensor,
        feature_type: str,
    ) -> tf.Tensor:
        """Apply feature selection to a processed feature if enabled for its type.

        Args:
            feature_name: Name of the feature
            output_pipeline: The processed feature tensor
            feature_type: Type of the feature ('numeric', 'categorical', 'text', 'date', 'passthrough', 'time_series')

        Returns:
            The processed tensor, possibly with feature selection applied
        """
        # Feature selection is on either for every feature, or only for the
        # placement matching this feature's type. Passthrough and time series
        # features have no dedicated placement, so they are covered by
        # ALL_FEATURES alone.
        placement_for_type = {
            "numeric": FeatureSelectionPlacementOptions.NUMERIC,
            "categorical": FeatureSelectionPlacementOptions.CATEGORICAL,
            "text": FeatureSelectionPlacementOptions.TEXT,
            "date": FeatureSelectionPlacementOptions.DATE,
        }
        apply_selection = self.feature_selection_placement in (
            FeatureSelectionPlacementOptions.ALL_FEATURES,
            placement_for_type.get(feature_type),
        )

        # Apply feature selection if enabled
        if apply_selection:
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            output_pipeline, feature_weights = feature_selector([output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        return output_pipeline

    @_monitor_performance
    def _add_pipeline_numeric(
        self,
        feature_name: str,
        input_layer,
        stats: dict,
    ) -> None:
        """Add a numeric preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the mean and variance of the feature.
        """
        # Get the feature specifications
        _feature = self.features_specs[feature_name]

        # Initialize preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # First, cast to float32 is applied to all numeric features.
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
            name=f"cast_to_float_{feature_name}",
        )

        # Check for custom preprocessors
        preprocessor = self._create_feature_preprocessor(
            feature_name=feature_name,
            feature=_feature,
            preprocessor=preprocessor,
        )

        # If no custom preprocessors, apply standard preprocessing based on feature type
        if not _feature.preprocessors:
            # Check if distribution-aware encoding is enabled
            if self.use_distribution_aware:
                self._add_distribution_aware_encoding(
                    preprocessor,
                    feature_name,
                    _feature,
                )
            else:
                self._add_numeric_type_processing(
                    preprocessor,
                    feature_name,
                    _feature,
                    stats,
                )

        # Check for advanced numerical embedding. `NumericalFeature` carries a
        # `use_embedding` flag with an `embedding_dim` beside it, and nothing
        # read it: a feature asking for its own embedding was handed straight
        # through at its original width. It turns the embedding on for that one
        # feature now, as the model-level switch does for all of them.
        if self.use_advanced_numerical_embedding or getattr(
            _feature,
            "use_embedding",
            False,
        ):
            self._add_advanced_numerical_embedding(
                preprocessor,
                feature_name,
                _feature,
                input_layer,
            )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="numeric",
        )

        self.processed_features[feature_name] = _output_pipeline

    def _add_distribution_aware_encoding(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: NumericalFeature,
    ) -> None:
        """Add distribution-aware encoding to a numeric feature preprocessor.

        Args:
            preprocessor: The preprocessor to add encoding to
            feature_name: Name of the feature
            feature: Feature object with settings
        """
        logger.info(f"Using distribution-aware encoding for {feature_name}")
        # Check if manually specified distribution is provided. `NumericalFeature`
        # exposes it as an attribute; the two kwargs spellings are accepted so
        # that configs written against older releases keep working.
        _prefered_distribution = (
            getattr(feature, "preferred_distribution", None)
            or feature.kwargs.get("preferred_distribution")
            or feature.kwargs.get("prefered_distribution")
        )
        if _prefered_distribution is not None:
            if isinstance(_prefered_distribution, Enum):
                _prefered_distribution = _prefered_distribution.value
            logger.info(
                f"Using manually specified distribution '{_prefered_distribution}' "
                f"for {feature_name}",
            )
        else:
            logger.info(f"Using automatic distribution detection for {feature_name}")

        # Apply distribution-aware encoding
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.distribution_aware_encoder,
            name=f"distribution_aware_layer_{feature_name}",
            num_bins=self.distribution_aware_bins,
            detect_periodicity=True,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            prefered_distribution=_prefered_distribution,
        )
        # Cast to float32 after distribution-aware encoding
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
            name=f"post_dist_cast_to_float_{feature_name}",
        )

    def _add_numeric_type_processing(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: NumericalFeature,
        stats: dict,
    ) -> None:
        """Add type-specific processing to a numeric feature preprocessor.

        Args:
            preprocessor: The preprocessor to add processing to
            feature_name: Name of the feature
            feature: Feature object with type and settings
            stats: Statistics for the feature
        """
        # Default behavior if no specific preprocessing is defined
        if feature.feature_type == FeatureType.FLOAT_NORMALIZED:
            logger.debug("Adding Float Normalized Feature")
            preprocessor.add_processing_step(
                layer_class="Normalization",
                mean=stats["mean"],
                variance=stats["var"],
                name=f"norm_{feature_name}",
            )
        elif feature.feature_type == FeatureType.FLOAT_RESCALED:
            logger.debug("Adding Float Rescaled Feature")
            rescaling_scale = feature.kwargs.get(
                "scale",
                1.0,
            )  # Default scale is 1.0 if not specified
            preprocessor.add_processing_step(
                layer_class="Rescaling",
                scale=rescaling_scale,
                name=f"rescale_{feature_name}",
            )
        elif feature.feature_type == FeatureType.FLOAT_DISCRETIZED:
            logger.debug("Adding Float Discretized Feature")
            boundaries = list(feature.kwargs.get("bin_boundaries") or [])
            if not boundaries:
                # Keras' Discretization needs either explicit boundaries or a
                # call to adapt(); passing only `num_bins` leaves it unbuilt, so
                # the model raised the moment it was called. KDP makes a single
                # statistics pass and never adapts, so derive equal-width
                # boundaries from the statistics that pass already produced.
                boundaries = self._derive_bin_boundaries(feature.num_bins, stats)
            _out_dims = len(boundaries) + 1

            # Create a dictionary of parameters to pass to the Discretization layer
            discretization_params = {
                "name": f"discretize_{feature_name}",
                "bin_boundaries": boundaries,
            }

            # Add any additional kwargs
            for key, value in feature.kwargs.items():
                if key not in ("bin_boundaries", "num_bins"):
                    discretization_params[key] = value

            preprocessor.add_processing_step(
                layer_class="Discretization",
                **discretization_params,
            )

            preprocessor.add_processing_step(
                layer_class="CategoryEncoding",
                num_tokens=_out_dims,
                output_mode="one_hot",
                name=f"one_hot_{feature_name}",
            )
        else:
            logger.debug("Adding Float Normalized Feature -> Default Option")
            preprocessor.add_processing_step(
                layer_class="Normalization",
                mean=stats["mean"],
                variance=stats["var"],
                name=f"norm_{feature_name}",
            )

    @staticmethod
    def _derive_bin_boundaries(num_bins: int, stats: dict) -> list[float]:
        """Derive equal-width discretization boundaries from feature statistics.

        The boundaries span mean +/- 3 standard deviations, which covers
        essentially the whole range of a roughly normal feature while staying
        robust to the extreme values a min/max range would stretch to.

        Args:
            num_bins: Number of buckets wanted.
            stats: Feature statistics holding at least "mean" and "var".

        Returns:
            The interior boundaries, i.e. ``num_bins - 1`` values, which
            Discretization turns into ``num_bins`` buckets. Empty when a single
            bucket is requested.
        """
        if not num_bins or num_bins < 2:
            return []

        mean = float(stats.get("mean", 0.0))
        variance = float(stats.get("var", 1.0))
        std = float(np.sqrt(variance)) if variance > 0 else 0.0
        if std <= 0:
            # A constant feature has no spread to bin across; fall back to a
            # unit window so the boundaries stay strictly increasing.
            std = 1.0

        edges = np.linspace(mean - 3.0 * std, mean + 3.0 * std, num_bins + 1)
        return [float(edge) for edge in edges[1:-1]]

    def _add_advanced_numerical_embedding(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: NumericalFeature,
        input_layer,
    ) -> None:
        """Add advanced numerical embedding to a preprocessor.

        Args:
            preprocessor: The preprocessor to add the embedding to
            feature_name: Name of the feature
            feature: Feature object with settings
            input_layer: Input layer for the feature
        """
        logger.info(f"Using NumericalEmbedding for {feature_name}")
        # These were passed to `add_processing_step` below, where the layer
        # creator ignored every keyword argument and returned an already-built
        # layer, so none of them reached the embedding. They are handed to the
        # feature instead, which uses them for whatever it did not set itself.
        embedding_layer = feature.get_embedding_layer(
            input_shape=input_layer.shape,
            defaults={
                "embedding_dim": self.embedding_dim,
                "mlp_hidden_units": self.mlp_hidden_units,
                "num_bins": self.num_bins,
                "init_min": self.init_min,
                "init_max": self.init_max,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            },
        )
        preprocessor.add_processing_step(
            layer_creator=lambda **kwargs: embedding_layer,
            layer_class="NumericalEmbedding",
            name=f"advanced_embedding_{feature_name}",
        )

        # `NumericalEmbedding` embeds each column it is given, so a feature that
        # arrives one column wide comes back rank 2 while a discretised one --
        # ten one-hot columns -- comes back rank 3. Mixing the two then failed
        # in `Concatenate`, and a model of only discretised features silently
        # produced a three-dimensional output. Flattening makes every numeric
        # feature rank 2; it is a no-op for the ones already there.
        preprocessor.add_processing_step(
            layer_creator=lambda **kwargs: keras.layers.Flatten(
                name=f"flatten_advanced_embedding_{feature_name}",
            ),
            layer_class="Flatten",
            name=f"flatten_advanced_embedding_{feature_name}",
        )

    @_monitor_performance
    def _add_pipeline_categorical(
        self,
        feature_name: str,
        input_layer,
        stats: dict,
    ) -> None:
        """Add a categorical preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the vocabulary of the feature.
        """
        # Get the feature object and its vocabulary
        _feature = self.features_specs[feature_name]

        # For hashing, we don't need a vocabulary
        vocab = []
        if _feature.category_encoding != CategoryEncodingOptions.HASHING:
            vocab = stats.get("vocab", [])

        # Initialize preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check for custom preprocessors
        preprocessor = self._create_feature_preprocessor(
            feature_name=feature_name,
            feature=_feature,
            preprocessor=preprocessor,
        )

        # If no custom preprocessors, apply standard categorical preprocessing
        if not _feature.preprocessors:
            self._add_categorical_lookup(preprocessor, feature_name, _feature, vocab)
            self._add_categorical_encoding(preprocessor, feature_name, _feature, vocab)

        # Flatten the categorical feature
        preprocessor.add_processing_step(
            layer_class="Flatten",
            name=f"flatten_{feature_name}",
        )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="categorical",
        )

        self.processed_features[feature_name] = _output_pipeline

    def _add_categorical_lookup(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: CategoricalFeature,
        vocab: list,
    ) -> None:
        """Add categorical lookup preprocessing.

        Args:
            preprocessor: The preprocessor to add the lookup to
            feature_name: Name of the feature
            feature: Feature object with settings
            vocab: Vocabulary for the feature
        """
        # Skip lookup for hashing - hashing doesn't need a vocabulary
        if feature.category_encoding == CategoryEncodingOptions.HASHING:
            return

        # Handle empty vocabulary by providing a fallback
        if not vocab:
            logger.warning(
                f"Empty vocabulary for categorical feature '{feature_name}'. "
                "Using fallback vocabulary with placeholder values.",
            )
            # Provide a minimal vocabulary with unknown/placeholder values
            vocab = ["<UNK>"]

        # Default behavior if no specific preprocessing is defined
        if feature.feature_type == FeatureType.STRING_CATEGORICAL:
            preprocessor.add_processing_step(
                layer_class="StringLookup",
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )
        elif feature.feature_type == FeatureType.INTEGER_CATEGORICAL:
            preprocessor.add_processing_step(
                layer_class="IntegerLookup",
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )

    def _add_categorical_encoding(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: CategoricalFeature,
        vocab: list,
    ) -> None:
        """Add categorical encoding preprocessing.

        Args:
            preprocessor: The preprocessor to add the encoding to
            feature_name: Name of the feature
            feature: Feature object with settings
            vocab: Vocabulary for the feature
        """
        if feature.category_encoding == CategoryEncodingOptions.EMBEDDING:
            _custom_embedding_size = feature.kwargs.get("embedding_size")
            _vocab_size = len(vocab) + 1
            logger.debug(
                f"Custom embedding size: {_custom_embedding_size}, vocab size: {_vocab_size}",
            )
            emb_size = _custom_embedding_size or feature._embedding_size_rule(
                nr_categories=_vocab_size,
            )
            logger.debug(f"Feature {feature_name} using embedding size: {emb_size}")
            preprocessor.add_processing_step(
                layer_class="Embedding",
                input_dim=len(vocab) + 1,
                output_dim=emb_size,
                name=f"embed_{feature_name}",
            )
        elif feature.category_encoding == CategoryEncodingOptions.ONE_HOT_ENCODING:
            preprocessor.add_processing_step(
                layer_class="CategoryEncoding",
                num_tokens=len(vocab) + 1,
                output_mode="one_hot",
                name=f"one_hot_{feature_name}",
            )
            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )
        elif feature.category_encoding == CategoryEncodingOptions.HASHING:
            # Get hash bucket size from kwargs or calculate it based on vocab size
            hash_bucket_size = feature.kwargs.get(
                "hash_bucket_size",
                min(1024, max(100, len(vocab) * 2)),  # Default sizing strategy
            )

            # Get salt value from kwargs (try hash_salt first, then salt)
            salt_value = feature.kwargs.get(
                "hash_salt",
                feature.kwargs.get("salt", None),
            )

            # Ensure salt_value is in the correct format (integer or tuple of 2 integers)
            if isinstance(salt_value, str):
                # Convert string to integer using hash to ensure different strings get different values
                salt_value = hash(salt_value)

            logger.debug(
                f"Feature {feature_name} using hashing with {hash_bucket_size} buckets and salt={salt_value}",
            )

            # Add hashing layer
            preprocessor.add_processing_step(
                layer_class="Hashing",
                num_bins=hash_bucket_size,
                salt=salt_value,  # Use the validated salt value
                name=f"hash_{feature_name}",
            )

            # Add embedding on top of hashing if specified
            if feature.kwargs.get("hash_with_embedding", False):
                emb_size = feature.kwargs.get("embedding_size", 8)
                preprocessor.add_processing_step(
                    layer_class="Embedding",
                    input_dim=hash_bucket_size,
                    output_dim=emb_size,
                    name=f"hash_embed_{feature_name}",
                )
            else:
                # One-hot encode the hash output if no embedding is used
                preprocessor.add_processing_step(
                    layer_class="CategoryEncoding",
                    num_tokens=hash_bucket_size,
                    output_mode="multi_hot",  # Multi-hot because hashing can cause collisions
                    name=f"hash_encode_{feature_name}",
                )

            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )

    def _adapted_text_vectorizer(
        self,
        feature_name: str,
        **kwargs: Any,
    ) -> keras.layers.TextVectorization:
        """Build a `TextVectorization` adapted on one column of the data.

        Args:
            feature_name: The column to read.
            **kwargs: Passed to `TextVectorization`.

        Returns:
            A layer whose vocabulary and IDF weights come from the data.

        Raises:
            ValueError: If there is no data to adapt on.
        """
        if not self.path_data:
            raise ValueError(
                f"{feature_name}: output_mode='tf_idf' needs the data to compute "
                "IDF weights from, so `path_data` is required.",
            )

        vectorizer = keras.layers.TextVectorization(**kwargs)
        pattern = DatasetStatistics._get_csv_file_pattern(path=self.path_data)
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=pattern,
            num_epochs=1,
            shuffle=False,
            ignore_errors=True,
            batch_size=self.batch_size,
            select_columns=[feature_name],
        )
        vectorizer.adapt(dataset.map(lambda batch: batch[feature_name]))
        logger.info(
            f"Adapted the text vectorizer for {feature_name} on the data; "
            f"vocabulary size {len(vectorizer.get_vocabulary())}",
        )
        return vectorizer

    @_monitor_performance
    def _add_pipeline_text(self, feature_name: str, input_layer, stats: dict) -> None:
        """Add a text preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]

        # getting stats
        _vocab = stats["vocab"]
        logger.debug(f"TEXT: {_vocab = }")

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(f"Custom Preprocessors detected : {_feature.preprocessors}")
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # checking if we have stop words provided
            _stop_words = _feature.kwargs.get("stop_words", [])
            if _stop_words:
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.text_preprocessing_layer,
                    name=f"text_preprocessor_{feature_name}",
                    **_feature.kwargs,
                )
            # `output_sequence_length` only applies to the "int" output mode;
            # TextVectorization rejects it outright for multi_hot, count and
            # tf_idf, so defaulting it unconditionally made those modes
            # unreachable.
            if (
                _feature.kwargs.get("output_mode", TextVectorizerOutputOptions.INT)
                == TextVectorizerOutputOptions.INT
                and "output_sequence_length" not in _feature.kwargs
            ):
                _feature.kwargs["output_sequence_length"] = 35

            # `tf_idf` weights every token by how rare it is across documents,
            # and Keras refuses a vocabulary in that mode without the matching
            # `idf_weights`, which KDP's statistics do not collect -- so the
            # documented, exported `TF_IDF` output mode could not build at all.
            # Letting the layer adapt on the column computes both the vocabulary
            # and the weights, from the same data the statistics came from.
            if _feature.kwargs.get("output_mode") == TextVectorizerOutputOptions.TF_IDF:
                vectorizer = self._adapted_text_vectorizer(
                    feature_name=feature_name,
                    name=f"text_vactorizer_{feature_name}",
                    **_feature.kwargs,
                )
                preprocessor.add_processing_step(
                    layer_creator=lambda _vectorizer=vectorizer, **kwargs: _vectorizer,
                    layer_class="TextVectorization",
                    name=f"text_vactorizer_{feature_name}",
                )
            else:
                # adding text vectorization
                preprocessor.add_processing_step(
                    layer_class="TextVectorization",
                    name=f"text_vactorizer_{feature_name}",
                    vocabulary=_vocab,
                    **_feature.kwargs,
                )
            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )
        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if enabled for categorical features
        if (
            self.feature_selection_placement == FeatureSelectionPlacementOptions.TEXT
            or self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            _output_pipeline, feature_weights = feature_selector([_output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_date(self, feature_name: str, input_layer) -> None:
        """Add a date preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(f"Custom Preprocessors detected : {_feature.preprocessors}")
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # Default behavior if no specific preprocessing is defined
            if _feature.feature_type == FeatureType.DATE:
                logger.debug("Adding Date Parsing layer")
                date_format = _feature.kwargs.get(
                    "format",
                    "YYYY-MM-DD",
                )  # Default format if not specified
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.date_parsing_layer,
                    date_format=date_format,
                    name=f"date_parsing_{feature_name}",
                )

                logger.debug("Adding Date Encoding layer")
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.date_encoding_layer,
                    name=f"date_encoding_{feature_name}",
                )

                # Optionally, add SeasonLayer
                if _feature.kwargs.get("add_season", False):
                    logger.debug("Adding Season layer")
                    preprocessor.add_processing_step(
                        layer_creator=PreprocessorLayerFactory.date_season_layer,
                        name=f"date_season_{feature_name}",
                    )

                # Add cast to float32 for concatenation compatibility
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                    name=f"cast_to_float_{feature_name}",
                )
            else:
                logger.warning(f"No default preprocessing for {feature_name =} defined")

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if enabled for categorical features
        if (
            self.feature_selection_placement == FeatureSelectionPlacementOptions.DATE
            or self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            _output_pipeline, feature_weights = feature_selector([_output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_passthrough(self, feature_name: str, input_layer) -> None:
        """Add a passthrough feature to the pipeline.

        Depending on include_passthrough_in_output setting:
        - If True: Process minimally and include in main output (legacy behavior)
        - If False: Store unprocessed for separate access (recommended for IDs/metadata)

        Args:
            feature_name (str): The name of the feature to be passed through.
            input_layer: The input layer for the feature.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]

        if self.include_passthrough_in_output:
            # Legacy behavior: minimal processing and include in main output
            self._process_passthrough_for_output(feature_name, input_layer, _feature)
        else:
            # New behavior: store unprocessed for separate access
            self._store_passthrough_unprocessed(feature_name, input_layer, _feature)

    def _process_passthrough_for_output(
        self,
        feature_name: str,
        input_layer,
        _feature,
    ) -> None:
        """Process passthrough feature minimally for inclusion in main output."""
        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(
                f"Custom Preprocessors detected for passthrough: {_feature.preprocessors}",
            )
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # For passthrough features, preserve the original dtype or cast to specified dtype
            target_dtype = getattr(_feature, "dtype", None)
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.preserve_dtype_layer,
                name=f"preserve_dtype_{feature_name}",
                target_dtype=target_dtype,
            )

            # Optionally reshape if needed
            if _feature.kwargs.get("reshape", False):
                target_shape = _feature.kwargs.get("target_shape", (-1,))
                preprocessor.add_processing_step(
                    layer_class="Reshape",
                    target_shape=target_shape,
                    name=f"reshape_{feature_name}",
                )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="passthrough",
        )

        self.processed_features[feature_name] = _output_pipeline

    def _store_passthrough_unprocessed(
        self,
        feature_name: str,
        input_layer,
        _feature,
    ) -> None:
        """Store passthrough feature unprocessed for separate access."""
        logger.info(
            f"Storing passthrough feature '{feature_name}' unprocessed for separate access",
        )
        # The values are passed through untouched, but a Keras functional model
        # rejects an Input handed straight back as an output ("`inputs` not
        # connected to `outputs`"). Routing through PreserveDtypeLayer connects
        # the tensor to the graph while leaving the values alone.
        passthrough_layer = PreprocessorLayerFactory.preserve_dtype_layer(
            name=f"passthrough_{feature_name}",
            target_dtype=getattr(_feature, "dtype", None),
        )
        self.passthrough_outputs[feature_name] = passthrough_layer(input_layer)

    @_monitor_performance
    def _add_pipeline_time_series(
        self,
        feature_name: str,
        input_layer,
        feature,
    ) -> None:
        """Add a time series preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            feature: The feature object containing time series configuration.
        """
        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(feature, "preprocessors") and feature.preprocessors:
            logger.info(
                f"Custom Preprocessors detected for time series: {feature.preprocessors}",
            )
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=feature,
                feature_name=feature_name,
            )
        else:
            # A calendar feature arrives as date strings, so the numeric front of
            # this pipeline has to be skipped: casting a date to float raised
            # "Cast string to float is not supported" and made the option
            # unusable. `CalendarFeatureLayer` returns numbers, so the cast
            # happens after it instead.
            reads_dates = getattr(feature, "calendar_feature_config", None) and (
                getattr(feature, "dtype", None) == tf.string
            )

            if not reads_dates:
                # Default time series processing
                # Cast to float32 for concatenation compatibility
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                    name=f"cast_to_float_{feature_name}",
                )

                # Add normalization if specified
                if feature.kwargs.get("normalize", True):
                    preprocessor.add_processing_step(
                        layer_class="Normalization",
                        name=f"norm_{feature_name}",
                    )

            # Add time series transformation layers
            if hasattr(feature, "build_layers"):
                time_series_layers = feature.build_layers()
                for i, layer in enumerate(time_series_layers):
                    # Use the layer's name if available, otherwise create a generic one
                    layer_name = getattr(layer, "name", f"{feature_name}_ts_layer_{i}")
                    # We need to use a lambda to wrap the existing layer
                    preprocessor.add_processing_step(
                        layer_creator=lambda layer=layer, **kwargs: layer,
                        name=layer_name,
                    )
                    logger.info(
                        f"Adding time series layer: {layer_name} to the pipeline",
                    )

            if reads_dates:
                # The calendar layer turns the dates into numbers; everything
                # downstream concatenates floats.
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                    name=f"cast_to_float_{feature_name}",
                )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="time_series",
        )

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_cross(self) -> None:
        """Add a crossing preprocessing step to the pipeline.

        This method processes feature crosses by:
        1. Creating inputs for both features if they don't exist
        2. Applying hashed crossing
        3. Converting output to float32 for compatibility
        4. Adding the result to appropriate output collection based on output mode
        """
        for feature_a, feature_b, nr_bins in self.feature_crosses:
            preprocessor = FeaturePreprocessor(name=f"{feature_a}_x_{feature_b}")

            # checking inputs existance for feature A
            for _feature_name in [feature_a, feature_b]:
                # getting feature object
                _feature = self.features_specs[_feature_name]
                _input = self.inputs.get(_feature_name)
                if _input is None:
                    logger.info(f"Creating: {_feature} inputs and signature")
                    _col_dtype = _feature.dtype
                    self._add_input_column(feature_name=_feature, dtype=_col_dtype)

            feature_name = f"{feature_a}_x_{feature_b}"
            preprocessor.add_processing_step(
                layer_class="HashedCrossing",
                num_bins=nr_bins,
                name=f"cross_{feature_name}",
            )
            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )
            crossed_input = [self.inputs[feature_a], self.inputs[feature_b]]
            _output_pipeline = preprocessor.chain(input_layer=crossed_input)

            # Process the feature
            self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _prepare_outputs(self) -> None:
        """Prepare model outputs based on output mode."""
        logger.info("Building preprocessor Model")
        if self.output_mode == OutputModeOptions.CONCAT:
            self._prepare_concat_mode_outputs()
        else:
            self._prepare_dict_mode_outputs()

    def _prepare_concat_mode_outputs(self) -> None:
        """Prepare outputs for concatenation mode."""
        # Get features to concatenate
        numeric_features, categorical_features = self._group_features_by_type()

        # Concatenate features by type
        concat_num = self._concatenate_numeric_features(numeric_features)
        concat_cat = self._concatenate_categorical_features(categorical_features)

        # Combine all features
        self._combine_all_features(concat_num, concat_cat)

        # Store output dimensions needed for Feature MoE. These used to be read
        # off `norm_<name>` / `flatten_<name>` layers, keyed by looking feature
        # *tensors* up in `self.inputs`, which is a mapping of names -- so the
        # dictionary was always empty and the mixture fell back to cutting
        # `concat_all` into equal parts. With features of unequal width that
        # split lands mid-feature: every "feature" the router sees is a slice
        # spanning several real ones. The widths come from the tensors that
        # were actually concatenated now.
        if self.use_feature_moe and self.concat_all is not None:
            # `GlobalNumericalEmbedding` replaces the whole numeric block with a
            # single learned vector, so the per-feature slices no longer exist
            # inside `concat_all` and there is nothing to route feature-wise.
            if self.use_global_numerical_embedding and self.numeric_features:
                raise ValueError(
                    "use_feature_moe cannot be combined with "
                    "use_global_numerical_embedding when numeric features are "
                    "present: the global embedding merges every numeric feature "
                    "into one vector, leaving no per-feature slices for the "
                    "mixture to route.",
                )

            self.processed_features_dims = {
                name: int(tensor.shape[-1])
                for name, tensor in zip(
                    self._concat_feature_names,
                    self._concat_feature_tensors,
                    strict=True,
                )
            }
            self.output_dims = [
                (None, dim) for dim in self.processed_features_dims.values()
            ]

        # Apply transformations if needed
        if self.use_feature_moe:
            self._apply_feature_moe()

        if self.tabular_attention:
            self._apply_tabular_attention(concat_num, concat_cat)

        if self.transfo_nr_blocks:
            self._apply_transformer_blocks(concat_num, concat_cat)

        logger.info("Concatenating outputs mode enabled")

    def _combine_all_features(
        self,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor | None,
    ) -> None:
        """Combine numeric and categorical features.

        Args:
            concat_num: Concatenated numeric features tensor
            concat_cat: Concatenated categorical features tensor

        Raises:
            ValueError: If no features are available for concatenation
        """
        if concat_num is not None and concat_cat is not None:
            self.concat_all = keras.layers.Concatenate(
                name="ConcatenateAll",
                axis=-1,
            )([concat_num, concat_cat])
        elif concat_num is not None:
            self.concat_all = concat_num
        elif concat_cat is not None:
            self.concat_all = concat_cat
        else:
            # Check if we have passthrough features that are stored separately
            if self.passthrough_outputs and not self.include_passthrough_in_output:
                logger.info(
                    "No processed features to concatenate - only passthrough features exist",
                )
                self.concat_all = None  # Will be handled in model building
            else:
                raise ValueError("No features available for concatenation")

    def _group_features_by_type(self) -> tuple[list, list]:
        """Group processed features by type for concatenation.

        Returns:
            Tuple of (numeric_features, categorical_features) lists
        """
        # Initialize lists for features of different types
        numeric_features = []
        categorical_features = []
        passthrough_features_numeric = []
        passthrough_features_string = []

        # The names behind those tensors, in the same order. The mixture of
        # experts has to cut the concatenated tensor back into features, and it
        # can only do that correctly if it knows which feature each slice is
        # and how wide it is. It previously guessed an equal split.
        numeric_names = []
        categorical_names = []
        passthrough_names_numeric = []
        passthrough_names_string = []

        # Group processed features by type
        for feature_name, feature in self.processed_features.items():
            # Skip feature weights
            if feature_name.endswith("_weights"):
                continue

            # Add to appropriate list based on feature type
            feature_spec = self.features_specs.get(feature_name)
            if feature_spec is None:
                logger.warning(f"No feature spec found for {feature_name}, skipping")
                continue

            if (
                feature_name in self.numeric_features
                or feature_name in self.date_features
                or feature_name
                in self.time_series_features  # Add time series features to numeric features for concatenation
            ):
                logger.debug(f"Adding {feature_name} to numeric features")
                numeric_features.append(feature)
                numeric_names.append(feature_name)
            elif (
                feature_name in self.categorical_features
                or feature_name in self.text_features
            ):
                logger.debug(f"Adding {feature_name} to categorical features")
                categorical_features.append(feature)
                categorical_names.append(feature_name)
            elif feature_name in self.passthrough_features:
                # Only include passthrough features in concatenation if they're meant to be in output
                # When include_passthrough_in_output=False, they should be stored separately
                if self.include_passthrough_in_output:
                    # Separate passthrough features by dtype to avoid concatenation issues
                    feature_dtype = getattr(feature_spec, "dtype", tf.float32)
                    if feature_dtype == tf.string:
                        logger.debug(
                            f"Adding {feature_name} to string passthrough features",
                        )
                        passthrough_features_string.append(feature)
                        passthrough_names_string.append(feature_name)
                    else:
                        logger.debug(
                            f"Adding {feature_name} to numeric passthrough features",
                        )
                        passthrough_features_numeric.append(feature)
                        passthrough_names_numeric.append(feature_name)
                else:
                    logger.debug(
                        f"Skipping {feature_name} from concatenation (stored separately)",
                    )
            else:
                logger.warning(f"Unknown feature type for {feature_name}")

        # Add numeric passthrough features to numeric features (only if include_passthrough_in_output=True)
        if passthrough_features_numeric:
            numeric_features.extend(passthrough_features_numeric)
            numeric_names.extend(passthrough_names_numeric)

        # Add string passthrough features to categorical features (only if include_passthrough_in_output=True)
        # (since categorical features are typically strings and handled separately)
        if passthrough_features_string:
            categorical_features.extend(passthrough_features_string)
            categorical_names.extend(passthrough_names_string)

        # `concat_all` is numeric first, then categorical, so this is the order
        # of the slices inside it.
        self._concat_feature_names = [*numeric_names, *categorical_names]
        self._concat_feature_tensors = [*numeric_features, *categorical_features]

        return numeric_features, categorical_features

    def _concatenate_numeric_features(
        self,
        numeric_features: list,
    ) -> tf.Tensor | None:
        """Concatenate numeric features and apply global embedding if needed.

        Args:
            numeric_features: List of numeric feature tensors

        Returns:
            Concatenated numeric features tensor or None if no numeric features
        """
        if not numeric_features:
            return None

        concat_num = keras.layers.Concatenate(
            name="ConcatenateNumeric",
            axis=-1,
        )(numeric_features)

        if self.use_global_numerical_embedding:
            concat_num = GlobalNumericalEmbedding(
                global_embedding_dim=self.global_embedding_dim,
                global_mlp_hidden_units=self.global_mlp_hidden_units,
                global_num_bins=self.global_num_bins,
                global_init_min=self.global_init_min,
                global_init_max=self.global_init_max,
                global_dropout_rate=self.global_dropout_rate,
                global_use_batch_norm=self.global_use_batch_norm,
                global_pooling=self.global_pooling,
            )(concat_num)

        return concat_num

    def _concatenate_categorical_features(
        self,
        categorical_features: list,
    ) -> tf.Tensor | None:
        """Concatenate categorical features.

        Args:
            categorical_features: List of categorical feature tensors

        Returns:
            Concatenated categorical features tensor or None if no categorical features
        """
        if not categorical_features:
            return None

        return keras.layers.Concatenate(
            name="ConcatenateCategorical",
            axis=-1,
        )(categorical_features)

    def _apply_multi_resolution_attention(
        self,
        concat_num: tf.Tensor,
        concat_cat: tf.Tensor,
    ) -> None:
        """Apply multi-resolution tabular attention to features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info("Adding multi-resolution tabular attention")

        # Reshape numeric features to 3D tensor
        num_features_3d = keras.layers.Reshape(
            target_shape=(1, -1),
            name="reshape_numeric_3d",
        )(concat_num)

        # Reshape categorical features to 3D tensor
        cat_features_3d = keras.layers.Reshape(
            target_shape=(1, -1),
            name="reshape_categorical_3d",
        )(concat_cat)

        (
            num_output,
            cat_output,
        ) = PreprocessorLayerFactory.multi_resolution_attention_layer(
            num_heads=self.tabular_attention_heads,
            d_model=self.tabular_attention_dim,
            embedding_dim=self.tabular_attention_embedding_dim,
            dropout_rate=self.tabular_attention_dropout,
            name="multi_resolution_attention",
        )(num_features_3d, cat_features_3d)

        # Squeeze back to 2D
        num_output = keras.layers.Reshape(
            target_shape=(-1,),
            name="reshape_num_output_2d",
        )(num_output)

        cat_output = keras.layers.Reshape(
            target_shape=(-1,),
            name="reshape_cat_output_2d",
        )(cat_output)

        self.concat_all = keras.layers.Concatenate(
            name="ConcatenateMultiResolutionAttention",
            axis=-1,
        )([num_output, cat_output])

    def _apply_standard_attention(
        self,
        placement: str,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor | None,
    ) -> None:
        """Apply standard tabular attention based on placement.

        Args:
            placement: Where to apply attention (all_features, numeric, categorical)
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        if placement == TabularAttentionPlacementOptions.ALL_FEATURES:
            logger.info("Adding tabular attention to all features")
            # Reshape to 3D tensor (batch_size, 1, features)
            features_3d = keras.layers.Reshape(
                target_shape=(1, -1),
                name="reshape_features_3d",
            )(self.concat_all)

            attention_output = PreprocessorLayerFactory.tabular_attention_layer(
                num_heads=self.tabular_attention_heads,
                d_model=self.tabular_attention_dim,
                dropout_rate=self.tabular_attention_dropout,
                name="tabular_attention",
            )(features_3d)

            # Reshape back to 2D
            self.concat_all = keras.layers.Reshape(
                target_shape=(-1,),
                name="reshape_attention_2d",
            )(attention_output)

        elif placement == TabularAttentionPlacementOptions.NUMERIC:
            self._apply_numeric_attention(concat_num, concat_cat)

        elif placement == TabularAttentionPlacementOptions.CATEGORICAL:
            self._apply_categorical_attention(concat_num, concat_cat)

    def _apply_numeric_attention(
        self,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor | None,
    ) -> None:
        """Apply attention to numeric features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info("Adding tabular attention to numeric features")
        if concat_num is not None:
            # Reshape numeric features to 3D
            num_features_3d = keras.layers.Reshape(
                target_shape=(1, -1),
                name="reshape_numeric_3d",
            )(concat_num)

            attention_output = PreprocessorLayerFactory.tabular_attention_layer(
                num_heads=self.tabular_attention_heads,
                d_model=self.tabular_attention_dim,
                dropout_rate=self.tabular_attention_dropout,
                name="tabular_attention_numeric",
            )(num_features_3d)

            # Reshape back to 2D
            concat_num = keras.layers.Reshape(
                target_shape=(-1,),
                name="reshape_numeric_attention_2d",
            )(attention_output)

        if concat_cat is not None:
            self.concat_all = keras.layers.Concatenate(
                name="ConcatenateTabularAttention",
                axis=-1,
            )([concat_num, concat_cat])
        else:
            self.concat_all = concat_num

    def _apply_categorical_attention(
        self,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor | None,
    ) -> None:
        """Apply attention to categorical features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info("Adding tabular attention to categorical features")
        if concat_cat is not None:
            # Reshape categorical features to 3D
            cat_features_3d = keras.layers.Reshape(
                target_shape=(1, -1),
                name="reshape_categorical_3d",
            )(concat_cat)

            attention_output = PreprocessorLayerFactory.tabular_attention_layer(
                num_heads=self.tabular_attention_heads,
                d_model=self.tabular_attention_dim,
                dropout_rate=self.tabular_attention_dropout,
                name="tabular_attention_categorical",
            )(cat_features_3d)

            # Reshape back to 2D
            concat_cat = keras.layers.Reshape(
                target_shape=(-1,),
                name="reshape_categorical_attention_2d",
            )(attention_output)

        if concat_num is not None:
            self.concat_all = keras.layers.Concatenate(
                name="ConcatenateTabularAttention",
                axis=-1,
            )([concat_num, concat_cat])
        else:
            self.concat_all = concat_cat

    def _apply_tabular_attention(
        self,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor | None,
    ) -> None:
        """Apply tabular attention based on configuration.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        if (
            self.tabular_attention_placement
            == TabularAttentionPlacementOptions.MULTI_RESOLUTION
        ):
            if concat_num is not None and concat_cat is not None:
                self._apply_multi_resolution_attention(concat_num, concat_cat)
            else:
                logger.warning(
                    "Multi-resolution attention requires both numerical and categorical features",
                )
                if concat_num is not None:
                    self.concat_all = concat_num
                elif concat_cat is not None:
                    self.concat_all = concat_cat
        else:
            # Original tabular attention logic with 3D tensor support
            self._apply_standard_attention(
                self.tabular_attention_placement,
                concat_num,
                concat_cat,
            )

    def _apply_transformer_blocks(
        self,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor | None,
    ) -> None:
        """Apply transformer blocks based on configuration.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        if (
            self.transfo_placement == TransformerBlockPlacementOptions.CATEGORICAL
            and concat_cat is not None
        ):
            self._apply_categorical_transformer(concat_num, concat_cat)
        elif self.transfo_placement == TransformerBlockPlacementOptions.ALL_FEATURES:
            self._apply_all_features_transformer()

    def _apply_categorical_transformer(
        self,
        concat_num: tf.Tensor | None,
        concat_cat: tf.Tensor,
    ) -> None:
        """Apply transformer blocks to categorical features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info(
            f"Adding transformer blocks to categorical features: #{self.transfo_nr_blocks}",
        )
        transformed = concat_cat
        for block_idx in range(self.transfo_nr_blocks):
            transformed = PreprocessorLayerFactory.transformer_block_layer(
                dim_model=transformed.shape[-1],
                num_heads=self.transfo_nr_heads,
                ff_units=self.transfo_ff_units,
                dropout_rate=self.transfo_dropout_rate,
                name=f"transformer_block_{block_idx}_{self.transfo_nr_heads}heads",
            )(transformed)
        # Reshape transformer output to remove the extra dimension
        transformed = keras.layers.Reshape(
            target_shape=(-1,),  # Flatten to match numeric shape
            name="reshape_transformer_output",
        )(transformed)

        # Recombine with numeric features if they exist
        if concat_num is not None:
            self.concat_all = keras.layers.Concatenate(
                name="ConcatenateTransformed",
                axis=-1,
            )([concat_num, transformed])
        else:
            self.concat_all = transformed

    def _apply_all_features_transformer(self) -> None:
        """Apply transformer blocks to all features."""
        logger.info(
            f"Adding transformer blocks to all features: #{self.transfo_nr_blocks}",
        )
        for block_idx in range(self.transfo_nr_blocks):
            self.concat_all = PreprocessorLayerFactory.transformer_block_layer(
                dim_model=self.concat_all.shape[-1],
                num_heads=self.transfo_nr_heads,
                ff_units=self.transfo_ff_units,
                dropout_rate=self.transfo_dropout_rate,
                name=f"transformer_block_{block_idx}_{self.transfo_nr_heads}heads",
            )(self.concat_all)

    def _prepare_dict_mode_outputs(self) -> None:
        """Prepare outputs for dictionary mode."""
        # Dictionary mode
        if self.use_feature_moe:
            self._apply_feature_moe_dict_mode()

        outputs = OrderedDict(
            [(k, None) for k in self.inputs if k in self.processed_features],
        )
        outputs.update(OrderedDict(self.processed_features))
        self.outputs = outputs
        logger.info("OrderedDict outputs mode enabled")

    def _apply_feature_moe_dict_mode(self) -> None:
        """Apply Feature-wise Mixture of Experts in dictionary output mode.

        This method enhances individual feature representations using MoE
        and updates the processed_features dictionary with the enhanced versions.
        """
        logger.info(
            f"Applying Feature-wise Mixture of Experts (dict mode) with {self.feature_moe_num_experts} experts",
        )

        # Get feature names and corresponding processed features
        feature_names = []
        individual_features = []

        # Sorted, and that is load-bearing. `.keras` stores a layer's weights
        # under a name derived from its class and the order it was created --
        # `dense`, `dense_1`, `dense_2` -- not under the name we gave it. On
        # load the layers are rebuilt in the config's order, and Keras lists a
        # dict input's layers alphabetically, so building these in
        # `self.inputs` order made the two sequences disagree: two features
        # came back holding each other's projection weights. Building in the
        # same order Keras rebuilds in keeps them lined up.
        for feature_name in sorted(self.inputs):
            if feature_name in self.processed_features:
                feature_names.append(feature_name)
                individual_features.append(self.processed_features[feature_name])

        if not individual_features:
            logger.warning("No individual features found for Feature MoE in dict mode.")
            return

        # Same padding as the concat branch: stacking needs one width, and
        # dict mode hands over the processed features untouched, so a
        # normalised float (one column) and a discretised one (ten) were
        # stacked together. The build succeeded -- `StackFeaturesLayer` used to
        # report the first feature's width whatever the rest were -- and the
        # model raised on its first real batch.
        widths = [int(feature.shape[-1]) for feature in individual_features]
        if len(set(widths)) > 1:
            # Every feature gets a padding layer, including the widest, where it
            # is a no-op. Padding only the narrow ones leaves the features at
            # different graph depths, and Keras restores weights in traversal
            # order: on reload two projection layers swapped kernels and their
            # features came back with each other's values.
            target_width = max(widths)
            individual_features = [
                PadFeatureLayer(target_width, name=f"moe_pad_dict_{name}")(feature)
                for name, feature in zip(
                    feature_names,
                    individual_features,
                    strict=True,
                )
            ]

        # Stack the features along a new axis
        stacked_features = StackFeaturesLayer(name="stacked_features_for_moe_dict")(
            individual_features,
        )

        # Create the Feature MoE layer
        moe = FeatureMoE(
            num_experts=self.feature_moe_num_experts,
            expert_dim=self.feature_moe_expert_dim,
            expert_hidden_dims=self.feature_moe_hidden_dims,
            routing=self.feature_moe_routing,
            sparsity=self.feature_moe_sparsity,
            feature_names=feature_names,
            predefined_assignments=self.feature_moe_assignments,
            freeze_experts=self.feature_moe_freeze_experts,
            dropout_rate=self.feature_moe_dropout,
            name="feature_moe_dict",
        )

        # Apply the MoE layer
        moe_outputs = moe(stacked_features)

        # One projection per feature, in a single layer. This used to be one
        # `Dense` per feature, and `.keras` stores a layer's weights under a
        # name derived from its class and build order rather than the name it
        # was given -- so when Keras reordered those sibling layers rebuilding
        # the graph, two features came back holding each other's kernels.
        projected = PerFeatureDense(
            self.feature_moe_expert_dim,
            activation="relu",
            name="moe_projection_dict",
        )(moe_outputs)

        # Unstack the outputs back to individual features
        unstacked_outputs = UnstackLayer(name="unstack_moe_features_dict")(projected)

        for i, feature_name in enumerate(feature_names):
            projection = unstacked_outputs[i]

            # Same residual as the concat branch, where the widths allow it.
            if self.feature_moe_use_residual:
                original = individual_features[i]
                if original.shape[-1] == projection.shape[-1]:
                    projection = keras.layers.Add(
                        name=f"{feature_name}_moe_residual_dict",
                    )([original, projection])

            # Update the processed features with the MoE-enhanced version
            self.processed_features[feature_name] = projection

        logger.info("Feature MoE applied successfully in dict mode")

    def _apply_feature_moe(self) -> None:
        """Enhances the combined feature representation using Feature-wise Mixture of Experts (MoE)
        in concatenated output mode.

        This method creates a Feature MoE layer that routes features to different experts
        based on their content, improving the overall representational power.
        """
        logger.info(
            f"Applying Feature-wise Mixture of Experts (concat mode) with {self.feature_moe_num_experts} experts",
        )

        # Check if we have concatenated features to work with
        if not hasattr(self, "concat_all") or self.concat_all is None:
            logger.warning("No concatenated features found to apply Feature MoE")
            return

        # The concatenation order and the real width of every slice, recorded
        # when the features were grouped. Reading them off layer names and
        # falling back to an equal split cut features mid-value.
        moe_feature_names = list(getattr(self, "_concat_feature_names", []) or [])
        feature_dims = [
            int(dim)
            for dim in (getattr(self, "processed_features_dims", {}) or {}).values()
        ]
        if not moe_feature_names or not feature_dims:
            logger.warning("No features found to apply Feature MoE")
            return

        logger.info(
            f"Splitting concat_all into {len(feature_dims)} features "
            f"for Feature MoE: {dict(zip(moe_feature_names, feature_dims, strict=False))}",
        )
        feature_outputs = SplitLayer(feature_dims, name="split_layer")(self.concat_all)

        # Stacking needs one width for every feature, and a real feature set has
        # several -- one column for a normalised float, ten for a discretised
        # one. Padding keeps each feature whole; projecting would cost
        # parameters and a feature already at the target width is untouched, so
        # a uniform-width model is unaffected.
        target_width = max(feature_dims)
        if len(set(feature_dims)) > 1:
            # Padded uniformly, widest included, so every feature sits at the
            # same graph depth; see the dict-mode branch for why that matters.
            feature_outputs = [
                PadFeatureLayer(target_width, name=f"moe_pad_{name}")(output)
                for name, output in zip(
                    moe_feature_names,
                    feature_outputs,
                    strict=True,
                )
            ]

        # Stack the features for the MoE layer
        stacked_features = StackFeaturesLayer(name="stacked_features_for_moe")(
            feature_outputs,
        )

        # Only `num_experts`, `expert_dim` and `routing` used to be forwarded
        # here, so `feature_moe_hidden_dims`, `feature_moe_sparsity`,
        # `feature_moe_freeze_experts` and `feature_moe_dropout` did nothing in
        # concat mode, and `routing="predefined"` raised because the assignments
        # never arrived. The dict-mode branch already passed them all.
        feature_moe = FeatureMoE(
            num_experts=self.feature_moe_num_experts,
            expert_dim=self.feature_moe_expert_dim,
            expert_hidden_dims=self.feature_moe_hidden_dims,
            routing=self.feature_moe_routing,
            sparsity=self.feature_moe_sparsity,
            feature_names=moe_feature_names,
            predefined_assignments=self.feature_moe_assignments,
            freeze_experts=self.feature_moe_freeze_experts,
            dropout_rate=self.feature_moe_dropout,
            name="feature_moe_concat",
        )(stacked_features)

        # Unstack the features after MoE processing using a custom layer
        unstacked_features = UnstackLayer(name="unstack_moe_features")(feature_moe)

        # `feature_moe_use_residual` was stored on the model and read nowhere, so
        # the residual connection it names never existed. It is added here on
        # the features whose width the mixture preserved; where the expert
        # changed the width there is nothing to add the original to.
        if self.feature_moe_use_residual:
            with_residual = []
            for index, expert_output in enumerate(unstacked_features):
                original = feature_outputs[index]
                if original.shape[-1] == expert_output.shape[-1]:
                    expert_output = keras.layers.Add(
                        name=f"moe_residual_{index}",
                    )([original, expert_output])
                with_residual.append(expert_output)
            unstacked_features = with_residual

        # Concatenate the processed features back together
        self.concat_all = keras.layers.Concatenate(axis=-1, name="concat_moe_features")(
            unstacked_features,
        )

    @_monitor_performance
    def _cleanup_intermediate_tensors(self) -> None:
        """Clean up intermediate tensors to free memory."""
        if self._preprocessed_cache:
            self._preprocessed_cache.clear()

        # Clear intermediate tensors that are no longer needed
        if hasattr(self, "features_to_concat"):
            del self.features_to_concat
        if hasattr(self, "features_cat_to_concat"):
            del self.features_cat_to_concat

        # Force garbage collection
        gc.collect()

        # Clear backend session to free GPU memory if using GPU
        keras.backend.clear_session()

    @_monitor_performance
    def _requires_dataset_statistics(self) -> bool:
        """Report whether any configured feature needs a pass over the data.

        Numeric features need moments and text features need a vocabulary.
        Categorical features need a vocabulary unless they are hashed with an
        explicit bucket count -- hashing exists precisely so the data need not
        be scanned. Date, passthrough and time series features derive
        everything from their inputs.

        Returns:
            True when the statistics pass is required.
        """
        if self.numeric_features or self.text_features:
            return True

        for feature_name in self.categorical_features:
            feature = self.features_specs.get(feature_name)
            if feature is None:
                return True
            if (
                getattr(feature, "category_encoding", None)
                != CategoryEncodingOptions.HASHING
            ):
                return True
            # Without an explicit size, the bucket count is derived from the
            # vocabulary, which still requires the statistics pass.
            if "hash_bucket_size" not in getattr(feature, "kwargs", {}):
                return True

        return False

    def build_preprocessor(self) -> dict:
        """Building preprocessing model.

        Returns:
            dict: Dictionary containing:
                - model: The preprocessing model
                - inputs: Model inputs
                - signature: Model signature
                - output_dims: Output dimensions
                - feature_stats: Feature statistics

        Raises:
            ValueError: If no features are specified or if required stats are missing
        """
        try:
            # Validate inputs
            if not self.features_specs:
                raise ValueError(
                    "No features specified. Please provide features_specs.",
                )

            # preparing statistics if they do not exist
            if not self.features_stats or self.overwrite_stats:
                if not self._requires_dataset_statistics():
                    # Hashing sizes its own buckets and dates parse themselves;
                    # a model built only from such features has nothing to learn
                    # from the data, so it should not demand a dataset.
                    logger.info(
                        "No feature requires dataset statistics; skipping the "
                        "statistics pass.",
                    )
                    self.features_stats = self.features_stats or {}
                else:
                    logger.info("No input features_stats detected !")
                    if self.path_data is None:
                        raise ValueError(
                            "Statistics are needed to build this preprocessor "
                            "(numeric, text, or vocabulary-based categorical "
                            "features are present) but neither `features_stats` "
                            "nor `path_data` was provided. Pass `path_data` "
                            "pointing at your CSV data, or supply precomputed "
                            "`features_stats`.",
                        )
                    if not hasattr(self, "stats_instance"):
                        raise ValueError(
                            "stats_instance not initialized. Cannot calculate features stats.",
                        )
                    self.features_stats = self.stats_instance.main()
                    logger.debug(
                        f"Features Stats were calculated: {self.features_stats}",
                    )

            # Set up inputs for all feature types BEFORE processing them
            for feature_name in (
                self.numeric_features
                + self.categorical_features
                + self.text_features
                + self.date_features
                + self.passthrough_features
                + self.time_series_features  # Add time series features
            ):
                if feature_name not in self.inputs:
                    # Get feature and its data type
                    feature = self.features_specs.get(feature_name)
                    if feature:
                        # Try to get dtype from feature first, then from stats, then default
                        dtype = getattr(feature, "dtype", None)
                        if dtype is None:
                            # Try to get dtype from stats if available
                            feature_stats = None
                            if feature_name in self.numeric_features:
                                feature_stats = self.features_stats.get(
                                    "numeric_stats",
                                    {},
                                ).get(feature_name, {})
                            elif feature_name in self.categorical_features:
                                feature_stats = self.features_stats.get(
                                    "categorical_stats",
                                    {},
                                ).get(feature_name, {})
                            elif feature_name in self.text_features:
                                feature_stats = self.features_stats.get("text", {}).get(
                                    feature_name,
                                    {},
                                )
                            elif feature_name in self.date_features:
                                feature_stats = self.features_stats.get("date", {}).get(
                                    feature_name,
                                    {},
                                )
                            elif feature_name in self.time_series_features:
                                feature_stats = self.features_stats.get(
                                    "time_series",
                                    {},
                                ).get(feature_name, {})

                            if feature_stats:
                                dtype = feature_stats.get("dtype", tf.float32)
                            else:
                                # Final fallback based on feature type
                                if isinstance(feature, PassthroughFeature):
                                    dtype = getattr(feature, "dtype", tf.float32)
                                elif feature.feature_type in (
                                    FeatureType.STRING_CATEGORICAL,
                                    FeatureType.TEXT,
                                    FeatureType.DATE,
                                ):
                                    dtype = tf.string
                                else:
                                    dtype = tf.float32

                        self._add_input_column(feature_name=feature_name, dtype=dtype)
                        self._add_input_signature(
                            feature_name=feature_name,
                            dtype=dtype,
                        )

            # Process features in batches by type
            numeric_batch = []
            categorical_batch = []
            text_batch = []
            date_batch = []
            passthrough_batch = []
            time_series_batch = []  # Add time series batch

            # Get the numeric stats from the correct location in features_stats
            numeric_stats = self.features_stats.get("numeric_stats", {})
            categorical_stats = self.features_stats.get("categorical_stats", {})
            text_stats = self.features_stats.get("text", {})
            time_series_stats = self.features_stats.get(
                "time_series",
                {},
            )  # Add time series stats

            for f_name in self.numeric_features:
                numeric_batch.append((f_name, numeric_stats.get(f_name, {})))
            for f_name in self.categorical_features:
                categorical_batch.append((f_name, categorical_stats.get(f_name, {})))
            for f_name in self.text_features:
                text_batch.append((f_name, text_stats.get(f_name, {})))
            for f_name in self.date_features:
                date_batch.append((f_name, {}))
            for f_name in self.time_series_features:  # Process time series features
                time_series_batch.append((f_name, time_series_stats.get(f_name, {})))
            for f_name in self.passthrough_features:
                passthrough_batch.append((f_name, {}))

            # Process features in parallel by type
            if numeric_batch:
                self._process_feature_batch(numeric_batch, "numeric")
            if categorical_batch:
                self._process_feature_batch(categorical_batch, "categorical")
            if text_batch:
                self._process_feature_batch(text_batch, "text")
            if date_batch:
                self._process_feature_batch(date_batch, "date")
            if time_series_batch:  # Process time series batch
                self._process_feature_batch(time_series_batch, "time_series")
            if passthrough_batch:
                self._process_feature_batch(passthrough_batch, "passthrough")

            # CROSSING FEATURES (based on defined inputs)
            if self.feature_crosses:
                logger.info("Processing feature type: cross feature")
                self._add_pipeline_cross()

            # Prepare outputs based on mode
            logger.info("Preparing outputs for the model")
            self._prepare_outputs()

            # Build the model based on output mode
            logger.info("Building preprocessor Model")
            if self.output_mode == OutputModeOptions.CONCAT.value:
                # Handle case where only passthrough features exist
                if (
                    self.concat_all is None
                    and self.passthrough_outputs
                    and not self.include_passthrough_in_output
                ):
                    logger.info(
                        "Only passthrough features detected - creating passthrough-only model",
                    )
                    self.model = keras.Model(
                        inputs=self.inputs,
                        outputs=self.passthrough_outputs,
                        name=self.model_name,
                    )
                    _output_dims = "passthrough_only"
                elif self.concat_all is None:
                    raise ValueError(
                        "No features were concatenated. Check if features were properly processed.",
                    )
                else:
                    # Determine outputs based on passthrough settings
                    if (
                        self.passthrough_outputs
                        and not self.include_passthrough_in_output
                    ):
                        # Include both processed (concat) and passthrough outputs
                        model_outputs = {
                            "processed": self.concat_all,
                            "passthrough": self.passthrough_outputs,
                        }
                        logger.info(
                            "Creating model with separate passthrough outputs: "
                            f"{list(self.passthrough_outputs.keys())}",
                        )
                    else:
                        # Standard concat output
                        model_outputs = self.concat_all

                    self.model = keras.Model(
                        inputs=self.inputs,
                        outputs=model_outputs,
                        name=self.model_name,
                    )
                    _output_dims = (
                        self.model.output_shape[1]
                        if isinstance(model_outputs, tf.Tensor)
                        else "mixed"
                    )

            else:  # DICT mode
                if not self.outputs and not self.passthrough_outputs:
                    raise ValueError(
                        "No outputs were created. Check if features were properly processed.",
                    )

                # Include passthrough outputs in dict mode if they exist
                final_outputs = self.outputs.copy() if self.outputs else {}
                if self.passthrough_outputs and not self.include_passthrough_in_output:
                    final_outputs.update(self.passthrough_outputs)
                    logger.info(
                        f"Adding passthrough outputs to dict mode: {list(self.passthrough_outputs.keys())}",
                    )
                elif not final_outputs:
                    # Only passthrough features exist
                    final_outputs = self.passthrough_outputs
                    logger.info(
                        "Only passthrough features detected - creating passthrough-only dict model",
                    )

                self.model = keras.Model(
                    inputs=self.inputs,
                    outputs=final_outputs,
                    name=self.model_name,
                )
                _output_dims = self.model.output_shape

            # Log model information
            logger.info("Preprocessor Model built successfully")
            logger.info(f"Model Summary: {self.model.summary()}")
            logger.info(f"Inputs: {list(self.inputs.keys())}")
            logger.info(f"Output Mode: {self.output_mode}")
            logger.info(f"Output Dimensions: {_output_dims}")

            # Get feature statistics for return
            feature_stats = {
                "numeric": self.features_stats.get("numeric", {}),
                "categorical": self.features_stats.get("categorical", {}),
                "text": self.features_stats.get("text", {}),
                "time_series": self.features_stats.get(
                    "time_series",
                    {},
                ),  # Add time series stats
            }

            # Clean up intermediate tensors
            self._cleanup_intermediate_tensors()

            return CallableDict(
                {
                    "model": self.model,
                    "inputs": self.inputs,
                    "signature": self.signature,
                    "output_dims": _output_dims,
                    "feature_stats": feature_stats,
                },
            )

        except Exception as e:
            logger.error(f"Error building preprocessor model: {str(e)}")
            raise

    @_monitor_performance
    def save_model(self, save_path: str) -> None:
        """Save the preprocessing model and its metadata.

        This method saves both the TensorFlow model and additional metadata
        needed to fully reconstruct the preprocessing pipeline.

        Args:
            save_path: Directory path where to save the model and metadata

        Raises:
            ValueError: If the model hasn't been built yet
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model must be built before saving. Call build_preprocessor() first.",
            )

        # Create the directory if it doesn't exist
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        # Save the TensorFlow model with proper extension
        model_path = save_path / "model.keras"
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Prepare metadata
        metadata = {
            "output_mode": self.output_mode,
            "use_feature_moe": self.use_feature_moe,
            "features_specs": {
                name: str(feature) for name, feature in self.features_specs.items()
            },
            "features_stats": self.features_stats,
        }

        # Add MoE configuration if enabled
        if self.use_feature_moe:
            metadata["feature_moe_config"] = {
                "num_experts": self.feature_moe_num_experts,
                "expert_dim": self.feature_moe_expert_dim,
                "routing": self.feature_moe_routing,
                "sparsity": self.feature_moe_sparsity,
                "dropout": self.feature_moe_dropout,
            }

        # Save metadata as JSON
        metadata_path = save_path / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved to {metadata_path}")

    @staticmethod
    def load_model(load_path: str) -> tuple:
        """Load a saved preprocessing model and its metadata.

        Args:
            load_path: Directory path where the model and metadata are saved

        Returns:
            tuple: (loaded_model, metadata)

        Raises:
            ValueError: If the model directory doesn't exist or is missing required files
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise ValueError(f"Model path {load_path} does not exist")

        # Check if both model and metadata exist
        model_path = load_path / "model.keras"
        metadata_path = load_path / "metadata.json"

        if not model_path.exists():
            raise ValueError(f"Model file {model_path} does not exist")
        if not metadata_path.exists():
            raise ValueError(f"Metadata file {metadata_path} does not exist")

        # Load the model
        loaded_model = keras.saving.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")

        # Load metadata
        with metadata_path.open() as f:
            metadata = json.load(f)
        logger.info(f"Model metadata loaded from {metadata_path}")

        return loaded_model, metadata

    def batch_predict(self, dataset: tf.data.Dataset) -> Generator:
        """Process batches of data through the model.

        Args:
            dataset: TensorFlow dataset containing batches of input data

        Yields:
            Preprocessed batches

        Raises:
            ValueError: If the model hasn't been built yet
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model must be built before prediction. Call build_preprocessor() first.",
            )

        # Process each batch of data
        for batch in dataset:
            # Apply preprocessing
            yield self.model(batch)

    def get_timing_metrics(self) -> dict:
        """Report how long the monitored build steps took.

        Returns:
            dict: `total_seconds` for the whole build plus a `steps` mapping of
            each monitored method to the seconds it accounted for. Empty of
            steps until `build_preprocessor` has run.
        """
        return {
            "total_seconds": sum(self._step_seconds.values()),
            "steps": dict(self._step_seconds),
        }

    def get_memory_usage(self) -> dict:
        """Report the GPU memory the monitored build steps needed.

        The figures come from `tf.config.experimental.get_memory_info`, which
        only reports GPU memory. On a CPU-only machine, and for steps that
        allocated nothing, every figure is `0.0`.

        Returns:
            dict: `peak_mb` for the largest single step and a `steps` mapping of
            each monitored method to the megabytes it peaked at.
        """
        steps = {
            name: value / (1024 * 1024)
            for name, value in self._step_memory_bytes.items()
        }
        return {
            "peak_mb": max(steps.values(), default=0.0),
            "steps": steps,
        }

    def plot_model(self, to_file: str = "model_architecture.png", **kwargs: Any) -> Any:
        """Write a diagram of the preprocessing model to an image file.

        Args:
            to_file: Where to write the image.
            **kwargs: Passed through to `keras.utils.plot_model`; the shape,
                dtype and layer-name flags default to on because they are what
                makes a preprocessing graph readable.

        Returns:
            Whatever `keras.utils.plot_model` returns, so the diagram renders
            inline in a notebook.

        Raises:
            ValueError: If the preprocessor has not been built yet.
            ImportError: If `pydot` and Graphviz are not installed.
        """
        if self.model is None:
            raise ValueError(
                "There is no model to plot yet. Call build_preprocessor() first.",
            )

        options = {
            "show_shapes": True,
            "show_dtype": True,
            "show_layer_names": True,
            **kwargs,
        }
        result = keras.utils.plot_model(self.model, to_file=to_file, **options)

        # `keras.utils.plot_model` prints its complaint and returns None when
        # pydot or Graphviz are missing, so a caller who only looked at the
        # return value would think the diagram had been written.
        if not Path(to_file).exists():
            raise ImportError(
                "Plotting the model needs `pydot` and Graphviz, and no image "
                f"was written to {to_file!r}. Install them with "
                "`pip install pydot` and your system's `graphviz` package.",
            )
        return result

    def get_feature_importances(self, data: dict | None = None) -> dict:
        """Get feature importance weights if feature selection was enabled.

        The selection layer computes a softmax over features for every row, so
        the importances depend on the data rather than being fixed weights.
        Pass a batch to get numbers; without one there is nothing to score and
        only a description of each weight tensor can be returned.

        Args:
            data: Optional mapping of feature name to a batch of values. When
                given, the model is run and the mean importance per feature is
                returned as a float.

        Returns:
            dict: Feature name to mean importance (a float) when `data` is
                supplied, otherwise feature name to a description of its weight
                tensor.

        Raises:
            ValueError: If feature selection was not enabled or model hasn't been built
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model must be built before getting feature importances")

        if self.feature_selection_placement == FeatureSelectionPlacementOptions.NONE:
            return {}

        # Collect feature importance descriptions instead of the tensors themselves
        feature_importances = {}

        for key in self.processed_features:
            if key.endswith("_weights"):
                feature_name = key.replace("_weights", "")
                tensor = self.processed_features[key]

                # Without data there is nothing to score, so describe the
                # tensor the weights will come from.
                feature_importances[feature_name] = {
                    "shape": str(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "layer_name": tensor.name if hasattr(tensor, "name") else "unknown",
                }

        if data is None:
            return feature_importances

        weight_tensors = {
            key.replace("_weights", ""): self.processed_features[key]
            for key in self.processed_features
            if key.endswith("_weights")
        }
        if not weight_tensors:
            return {}

        weights_model = keras.Model(inputs=self.inputs, outputs=weight_tensors)
        scored = weights_model(_to_tensor_mapping(data))
        return {
            name: float(tf.reduce_mean(tf.cast(value, tf.float32)))
            for name, value in scored.items()
        }

    @staticmethod
    def _is_value_sequence(value) -> bool:
        """Report whether a value carries several time steps rather than one.

        Tensors count: `TimeSeriesInferenceFormatter` hands back tensors when
        asked to, and rejecting them meant KDP refused the output of its own
        companion formatter.

        Args:
            value: The value supplied for a time series feature.

        Returns:
            True when the value holds a sequence of time steps.
        """
        if isinstance(value, list | tuple | np.ndarray):
            return True
        if tf.is_tensor(value):
            rank = value.shape.rank
            return rank is None or rank >= 1
        # pandas Series and anything else sized, but not a bare string.
        return hasattr(value, "__len__") and not isinstance(value, str | bytes)

    def _validate_time_series_inference_data(self, data) -> bool:
        """Validate that the provided data meets minimum requirements for time series inference.

        Args:
            data: The data to validate, can be pandas DataFrame, dict, or TensorFlow dataset.

        Returns:
            bool: True if validation passes, False otherwise.

        Raises:
            ValueError: If data is insufficient for time series inference.
        """
        # Only validate if we have time series features
        time_series_features = [
            name
            for name, feature in self.features_specs.items()
            if (
                hasattr(feature, "feature_type")
                and feature.feature_type == FeatureType.TIME_SERIES
            )
        ]

        if not time_series_features:
            return True

        # Each time series feature needs a sequence of past values, not a scalar.
        if isinstance(data, dict):
            for key, value in data.items():
                if key in time_series_features and not self._is_value_sequence(value):
                    raise ValueError(
                        f"Time series feature '{key}' requires historical context. "
                        f"Please provide a list or array of values, not a single value.",
                    )

        # For each time series feature, check that we have enough data
        for feature_name in time_series_features:
            feature = self.features_specs[feature_name]

            # Check grouping column exists if needed
            if (
                getattr(feature, "group_by", None)
                and isinstance(data, dict)
                and feature.group_by not in data
            ):
                raise ValueError(
                    f"Time series feature '{feature_name}' requires grouping by "
                    f"'{feature.group_by}', but this column is not in the data.",
                )

            # Check sorting column exists if needed
            if (
                getattr(feature, "sort_by", None)
                and isinstance(data, dict)
                and feature.sort_by not in data
            ):
                raise ValueError(
                    f"Time series feature '{feature_name}' requires sorting by "
                    f"'{feature.sort_by}', but this column is not in the data.",
                )

            # Calculate minimum required history
            min_history = 1  # Default minimum

            # Check lag features
            if hasattr(feature, "lag_config") and feature.lag_config:
                lags = feature.lag_config.get("lags", [])
                if lags:
                    min_history = max(min_history, max(lags))

            # Check rolling statistics
            if (
                hasattr(feature, "rolling_stats_config")
                and feature.rolling_stats_config
            ):
                window_size = feature.rolling_stats_config.get("window_size", 1)
                min_history = max(min_history, window_size)

            # Check differencing
            if hasattr(feature, "differencing_config") and feature.differencing_config:
                order = feature.differencing_config.get("order", 1)
                min_history = max(min_history, order)

            # Check moving averages
            if (
                hasattr(feature, "moving_average_config")
                and feature.moving_average_config
            ):
                periods = feature.moving_average_config.get("periods", [])
                if periods:
                    min_history = max(min_history, max(periods))

            # Check wavelet transform
            if (
                hasattr(feature, "wavelet_transform_config")
                and feature.wavelet_transform_config
            ):
                levels = feature.wavelet_transform_config.get("levels", 3)
                min_history = max(min_history, 2**levels)

            # Check data size if it's a dict with lists/arrays
            if isinstance(data, dict) and feature_name in data:
                feature_data = data[feature_name]
                if isinstance(feature_data, list | np.ndarray):
                    data_length = len(feature_data)
                    if data_length < min_history:
                        raise ValueError(
                            f"Time series feature '{feature_name}' requires at least {min_history} "
                            f"historical data points, but only {data_length} were provided.",
                        )

        return True

    def predict(self, data, **kwargs) -> Any:
        """Predict using the preprocessor model.

        Args:
            data: The data to predict on, can be pandas DataFrame, dict, or TensorFlow dataset.
            **kwargs: Additional keyword arguments to pass to the model's predict method.

        Returns:
            The prediction output.
        """
        # Validate time series inference data
        self._validate_time_series_inference_data(data)

        # Keras cannot consume NumPy string arrays, which every categorical,
        # text or date feature produces when the caller passes a plain dict.
        if isinstance(data, dict):
            data = _to_tensor_mapping(data)

        # Call the model's predict method
        return self.model.predict(data, **kwargs)


# Define serializable custom layers
@keras.saving.register_keras_serializable(package="kdp.processor")
class SplitLayer(keras.layers.Layer):
    """Custom layer to split a tensor into individual features based on dimensions."""

    def __init__(self, feature_dims, **kwargs):
        """Initialize the SplitLayer.

        See the class docstring for the accepted arguments and what
        each one controls.
        """
        super().__init__(**kwargs)
        self.feature_dims = feature_dims

    def call(self, inputs) -> list:
        # Handle case where feature_dims is None or empty
        """Apply the layer to its inputs.

        Args:
            inputs: Input tensor to process.

        Returns:
            The transformed tensor.
        """
        if not self.feature_dims:
            # Return the input as a single feature if no dimensions are provided
            return [inputs]

        # Handle case where feature_dims is a list of integers
        if isinstance(self.feature_dims[0], int):
            # Create running index
            start_indices = [0]
            for dim in self.feature_dims[:-1]:
                start_indices.append(start_indices[-1] + dim)

            # Create [(start_idx, dim), ...] format
            split_indices = list(zip(start_indices, self.feature_dims, strict=False))
            return [inputs[:, i : i + dim] for i, dim in split_indices]

        # Handle case where feature_dims is already a list of tuples (i, dim)
        if (
            isinstance(self.feature_dims[0], list | tuple)
            and len(self.feature_dims[0]) == 2
        ):
            return [inputs[:, i : i + dim] for i, dim in self.feature_dims]

        # If we get here, feature_dims is in an invalid format
        raise ValueError(
            f"Invalid feature_dims format: {self.feature_dims}. "
            "Expected a list of integers or a list of (index, dimension) tuples.",
        )

    def get_config(self) -> dict:
        """Return the configuration needed to re-create this layer.

        Returns:
            The layer configuration.
        """
        config = super().get_config()
        config.update({"feature_dims": self.feature_dims})
        return config

    def compute_output_shape(self, input_shape) -> tuple:
        # Return a list of shapes for each split
        """Compute the output shape for a given input shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            The corresponding output shape.
        """
        if not self.feature_dims:
            return [input_shape]
        elif isinstance(self.feature_dims[0], int):
            return [(input_shape[0], dim) for dim in self.feature_dims]
        elif (
            isinstance(self.feature_dims[0], list | tuple)
            and len(self.feature_dims[0]) == 2
        ):
            return [(input_shape[0], dim) for _, dim in self.feature_dims]
        else:
            raise ValueError(
                f"Invalid feature_dims format: {self.feature_dims}. "
                "Expected a list of integers or a list of (index, dimension) tuples.",
            )
