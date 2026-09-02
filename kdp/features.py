from enum import Enum, auto
from typing import Any

import keras
import tensorflow as tf
from loguru import logger

from kdp.layers.distribution_aware_encoder_layer import (
    DistributionType as _EncoderDistributionType,
)
from kdp.layers_factory import PreprocessorLayerFactory


class TextVectorizerOutputOptions(str, Enum):
    """Output modes accepted by `TextFeature(output_mode=...)`.

    The members are the exact strings `keras.layers.TextVectorization` and
    KDP's own checks compare against. They used to be `auto()` integers here
    while `kdp.processor` defined a second, string-valued class of the same
    name, so whichever one a caller imported decided whether their option
    worked or was quietly discarded.
    """

    TF_IDF = "tf_idf"
    INT = "int"
    MULTI_HOT = "multi_hot"


class CategoryEncodingOptions:
    ONE_HOT_ENCODING = "ONE_HOT_ENCODING"
    EMBEDDING = "EMBEDDING"
    HASHING = "HASHING"


class CrossFeatureOutputOptions(str, Enum):
    """Output mode of a crossed feature.

    `feature_crosses` are hashed into integer bins, so `"int"` is the only
    mode there is. It is spelled as the string the layer takes, for the same
    reason as `TextVectorizerOutputOptions`.
    """

    INT = "int"


class FeatureType(Enum):
    FLOAT = auto()
    FLOAT_NORMALIZED = auto()
    FLOAT_RESCALED = auto()
    FLOAT_DISCRETIZED = auto()
    INTEGER_CATEGORICAL = auto()
    STRING_CATEGORICAL = auto()
    TEXT = auto()
    # Crosses are configured with `PreprocessingModel(feature_crosses=[...])`,
    # not by giving a column this type: a feature declared as CROSSES is
    # rejected by the feature-space converter.
    CROSSES = auto()
    DATE = auto()
    TIME_SERIES = auto()
    PASSTHROUGH = auto()

    @staticmethod
    def from_string(type_str: str) -> "FeatureType":
        """Converts a string to a FeatureType.

        Args:
            type_str (str): The string representation of the feature type.

        Returns:
            FeatureType: The corresponding enum value

        Raises:
            ValueError: If the string doesn't match any FeatureType
        """
        try:
            return FeatureType[type_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown feature type: {type_str}")


# `kdp.features` defined a second `DistributionType` whose members differed from
# the encoder's own: it carried a `WEIBULL` the encoder does not know, so
# `preferred_distribution=DistributionType.WEIBULL` was warned about and
# silently replaced with "normal". The encoder's class is the one that decides
# what is valid, so it is the one re-exported here.
DistributionType = _EncoderDistributionType


class Feature:
    """Base class for features with support for dynamic kwargs."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType | str,
        preprocessors: list[PreprocessorLayerFactory | Any] = None,
        **kwargs,
    ) -> None:
        """Initializes a Feature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType | str): The type of the feature.
            preprocessors (List[Union[PreprocessorLayerFactory, Any]]): The preprocessors to apply to the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        self.name = name
        self.feature_type = (
            FeatureType.from_string(feature_type)
            if isinstance(feature_type, str)
            else feature_type
        )
        self.preprocessors = preprocessors or []
        self.kwargs = kwargs

    def add_preprocessor(self, preprocessor: PreprocessorLayerFactory | Any) -> None:
        """Adds a preprocessor to the feature.

        Args:
            preprocessor (Union[PreprocessorLayerFactory, Any]): The preprocessor to add.
        """
        logger.info(f"Adding preprocessor {preprocessor} to feature {self.name}")
        if isinstance(preprocessor, PreprocessorLayerFactory):
            self.preprocessors.append(preprocessor.create_layer(**self.kwargs))
        else:
            self.preprocessors.append(preprocessor)

    def update_kwargs(self, **kwargs) -> None:
        """Updates the kwargs with new or modified parameters.

        Args:
            **kwargs: The new or modified parameters.
        """
        self.kwargs.update(kwargs)

    @staticmethod
    def from_string(type_str: str) -> "FeatureType":
        """Converts a string to a FeatureType.

        Args:
            type_str (str): The string representation of the feature type.
        """
        return FeatureType.from_string(type_str)


class _Unset:
    """Sentinel marking a parameter the caller never supplied.

    A model-level setting can then fill it in without overriding a value the
    caller chose. A bare `object()` would render as `<object object at 0x...>`
    in the generated API reference, so the repr is spelled out here to keep
    those pages readable and stable across runs.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Render as `<unset>` wherever a default is printed."""
        return "<unset>"

    def __bool__(self) -> bool:
        """A missing value is falsy, like the `None` it stands in for."""
        return False


_UNSET = _Unset()


class NumericalFeature(Feature):
    """NumericalFeature with dynamic kwargs passing and embedding support."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.FLOAT_NORMALIZED,
        preferred_distribution: DistributionType | None = None,
        use_embedding: bool = False,
        embedding_dim: int | _Unset = _UNSET,
        num_bins: int | _Unset = _UNSET,
        **kwargs,
    ) -> None:
        """Initializes a NumericalFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            preferred_distribution (DistributionType | None): The preferred distribution type.
            use_embedding (bool): Whether to use advanced numerical embedding.
            embedding_dim (int): Dimension of the embedding space.
            num_bins (int): Number of bins for discretization.
            **kwargs: Additional keyword arguments for the feature.
        """
        # `kdp.layers` and the model advisor both spell this option with one
        # "r", so callers reach for that spelling too -- and it landed in
        # `**kwargs`, which this class swallows, leaving the feature on
        # automatic detection while looking configured.
        legacy = kwargs.pop("prefered_distribution", None)
        if legacy is not None:
            if preferred_distribution is None:
                preferred_distribution = legacy
            logger.warning(
                f"{name}: `prefered_distribution` is a misspelling of "
                "`preferred_distribution` and is accepted for compatibility. "
                "Use `preferred_distribution`.",
            )

        super().__init__(name, feature_type, **kwargs)
        self.dtype = tf.float32
        self.preferred_distribution = preferred_distribution
        self.use_embedding = use_embedding
        # Remember what the caller actually asked for. `PreprocessingModel`
        # passes its own embedding settings, and those must not overwrite a
        # value set on the feature itself.
        self._explicit_embedding_options = {
            name
            for name, value in (
                ("embedding_dim", embedding_dim),
                ("num_bins", num_bins),
            )
            if value is not _UNSET
        }
        self.embedding_dim = 8 if embedding_dim is _UNSET else embedding_dim
        self.num_bins = 10 if num_bins is _UNSET else num_bins

    def get_embedding_layer(
        self,
        input_shape: tuple | None = None,  # noqa: ARG002 - kept for API compatibility
        defaults: dict | None = None,
    ) -> keras.layers.Layer:
        """Creates and returns a NumericalEmbedding layer configured for this feature.

        Args:
            input_shape: Unused. `NumericalEmbedding` derives the feature count
                in its own `build`, so nothing here depends on the shape. The
                parameter is kept, and optional, so existing callers that pass
                it keep working.
            defaults: Model-level embedding settings, used for every option
                this feature did not set itself. `PreprocessingModel` passes
                its `embedding_dim`, `mlp_hidden_units`, `num_bins`,
                `init_min`, `init_max`, `dropout_rate` and `use_batch_norm`
                here; without them those arguments had no effect at all.

        Returns:
            A `NumericalEmbedding` layer built from this feature's settings.
        """
        from kdp.layers.numerical_embedding_layer import NumericalEmbedding

        defaults = defaults or {}

        def resolve(option: str, fallback: object) -> object:
            """Feature setting wins, then the model-level one, then the default.

            Args:
                option: Name of the embedding option to resolve.
                fallback: Value to use when nothing else supplies one.

            Returns:
                The resolved value for the option.
            """
            if option in self._explicit_embedding_options:
                return getattr(self, option)
            if option in self.kwargs:
                return self.kwargs[option]
            if option in defaults and defaults[option] is not None:
                return defaults[option]
            return fallback

        embedding_dim = resolve("embedding_dim", self.embedding_dim)
        return NumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=resolve("mlp_hidden_units", max(16, embedding_dim * 2)),
            num_bins=resolve("num_bins", self.num_bins),
            init_min=resolve("init_min", -3.0),
            init_max=resolve("init_max", 3.0),
            dropout_rate=resolve("dropout_rate", 0.1),
            use_batch_norm=resolve("use_batch_norm", True),
            name=f"{self.name}_embedding",
        )


class CategoricalFeature(Feature):
    """CategoricalFeature with dynamic kwargs passing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.INTEGER_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.EMBEDDING,
        **kwargs,
    ) -> None:
        """Initializes a CategoricalFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            category_encoding (str): The category encoding type.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.category_encoding = self._normalise_encoding(category_encoding)
        self.dtype = (
            tf.int32 if feature_type == FeatureType.INTEGER_CATEGORICAL else tf.string
        )
        self.kwargs = kwargs

    @staticmethod
    def _normalise_encoding(category_encoding) -> str:
        """Accept any casing, reject anything that is not a real option.

        The encoding is compared against `CategoryEncodingOptions` by string
        equality further down the pipeline, so a value such as "hashing"
        matched nothing and the feature silently degraded to a bare lookup
        index -- no hashing, no embedding, no one-hot, and no error.

        Args:
            category_encoding: The requested encoding, in any casing.

        Returns:
            str: The canonical encoding name.

        Raises:
            ValueError: If the encoding is not one of the supported options.
        """
        valid = {
            CategoryEncodingOptions.ONE_HOT_ENCODING,
            CategoryEncodingOptions.EMBEDDING,
            CategoryEncodingOptions.HASHING,
        }
        if isinstance(category_encoding, str):
            canonical = category_encoding.upper()
            if canonical in valid:
                return canonical
        raise ValueError(
            f"Unsupported category_encoding {category_encoding!r}. "
            f"Expected one of {sorted(valid)}.",
        )

    def _embedding_size_rule(self, nr_categories: int) -> int:
        """Returns the embedding size for a given number of categories using the Embedding Size Rule of Thumb.

        Args:
            nr_categories (int): The number of categories.

        Returns:
            int: The embedding size.
        """
        return min(500, round(1.6 * nr_categories**0.56))


class TextFeature(Feature):
    """TextFeature with dynamic kwargs passing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.TEXT,
        **kwargs,
    ) -> None:
        """Initializes a TextFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = tf.string
        self.kwargs = kwargs


class DateFeature(Feature):
    """TextFeature with dynamic kwargs passing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.DATE,
        **kwargs,
    ) -> None:
        """Initializes a DateFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = tf.string
        self.kwargs = kwargs


class PassthroughFeature(Feature):
    """PassthroughFeature for including features in output without processing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.PASSTHROUGH,
        dtype: tf.DType = tf.float32,
        **kwargs,
    ) -> None:
        """Initializes a PassthroughFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            dtype (tf.DType): The data type of the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = dtype
        self.kwargs = kwargs


class TimeSeriesFeature(Feature):
    """TimeSeriesFeature with support for lag features and temporal processing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.TIME_SERIES,
        lag_config: dict = None,
        rolling_stats_config: dict = None,
        differencing_config: dict = None,
        moving_average_config: dict = None,
        wavelet_transform_config: dict = None,
        tsfresh_feature_config: dict = None,
        calendar_feature_config: dict = None,
        sequence_length: int = None,
        sort_by: str = None,
        sort_ascending: bool = True,
        group_by: str = None,
        dtype: tf.DType = tf.float32,
        is_target: bool = False,
        exclude_from_input: bool = False,
        input_type: str = "continuous",
        **kwargs,
    ) -> None:
        """Initializes a TimeSeriesFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            lag_config (dict): Configuration for lag features. If None, no lag features will be created.
                Example: {'lags': [1, 7, 14], 'drop_na': True}
            rolling_stats_config (dict): Configuration for rolling statistics.
                Example: {'window_size': 7, 'statistics': ['mean', 'std']}
            differencing_config (dict): Configuration for differencing.
                Example: {'order': 1}
            moving_average_config (dict): Configuration for moving averages.
                Example: {'periods': [7, 14, 30]}
            wavelet_transform_config (dict): Configuration for wavelet transform.
                Example: {'levels': 3, 'window_sizes': [4, 8, 16], 'flatten_output': True}
            tsfresh_feature_config (dict): Configuration for statistical feature extraction.
                Example: {'features': ['mean', 'std', 'min', 'max'], 'normalize': True}
            calendar_feature_config (dict): Configuration for calendar features.
                Example: {'features': ['month', 'day', 'day_of_week'], 'cyclic_encoding': True}
            sequence_length (int): Length of the sequence. If None, no sequence will be created.
            sort_by (str): Column name to sort the time series data by (typically a timestamp column).
                Required for proper time series ordering.
            sort_ascending (bool): Whether to sort in ascending order (True) or descending order (False).
                Default is True for chronological ordering.
            group_by (str): Optional column name to group time series data by. Useful for multiple series
                (e.g., data for different stores, customers, products, etc.)
            dtype (tf.DType): The data type of the feature.
            is_target (bool): Whether this feature is a target for prediction.
            exclude_from_input (bool): Whether to exclude this feature from the input.
            input_type (str): The input type of the feature (e.g., "continuous").
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = dtype
        self.is_target = is_target
        self.exclude_from_input = exclude_from_input
        self.input_type = input_type

        # Time series specific configurations
        self.lag_config = lag_config
        self.rolling_stats_config = rolling_stats_config
        self.differencing_config = differencing_config
        self.moving_average_config = moving_average_config
        self.wavelet_transform_config = wavelet_transform_config
        self.tsfresh_feature_config = tsfresh_feature_config
        self.calendar_feature_config = calendar_feature_config

        # Calendar features are read from date *strings*, so a feature that asks
        # for them takes a string column. It used to be declared float like
        # every other time series feature, and the cast in front of the pipeline
        # failed with "Cast string to float is not supported" -- the option
        # could not be used at all. The numeric configs read a number from the
        # same column, so they cannot be combined with it.
        if calendar_feature_config:
            numeric_configs = {
                "lag_config": lag_config,
                "rolling_stats_config": rolling_stats_config,
                "differencing_config": differencing_config,
                "moving_average_config": moving_average_config,
                "wavelet_transform_config": wavelet_transform_config,
                "tsfresh_feature_config": tsfresh_feature_config,
            }
            conflicting = sorted(
                name for name, value in numeric_configs.items() if value
            )
            if conflicting:
                raise ValueError(
                    f"{name}: calendar_feature_config reads dates from a string "
                    f"column, so it cannot be combined with {conflicting}, which "
                    "read numbers from the same column. Declare them as separate "
                    "features.",
                )
            self.dtype = tf.string

        self.sequence_length = sequence_length
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.group_by = group_by

        # Set default values for backward compatibility - use when needed, don't modify the original attributes
        if (
            hasattr(self, "lag_config")
            and self.lag_config is not None
            and "lags" not in self.lag_config
            and self.lag_config
        ):
            self.lag_config["lags"] = [1]
        if (
            hasattr(self, "lag_config")
            and self.lag_config is not None
            and "drop_na" not in self.lag_config
            and self.lag_config
        ):
            self.lag_config["drop_na"] = True

        # Validate configurations
        if self.rolling_stats_config and "window_size" not in self.rolling_stats_config:
            raise ValueError("window_size is required in rolling_stats_config")

        self.kwargs.update(
            {
                "lag_config": self.lag_config,
                "rolling_stats_config": self.rolling_stats_config,
                "differencing_config": self.differencing_config,
                "moving_average_config": self.moving_average_config,
                "wavelet_transform_config": self.wavelet_transform_config,
                "tsfresh_feature_config": self.tsfresh_feature_config,
                "calendar_feature_config": self.calendar_feature_config,
                "sequence_length": self.sequence_length,
                "sort_by": self.sort_by,
                "sort_ascending": self.sort_ascending,
                "group_by": self.group_by,
                "is_target": self.is_target,
                "exclude_from_input": self.exclude_from_input,
                "input_type": self.input_type,
            },
        )

    def _resolve_drop_na(
        self,
        config: dict,
        config_name: str,
        row_preserving: bool,
    ) -> bool:
        """Decide whether a transform may drop its warm-up rows.

        Args:
            config: The transform's configuration dictionary.
            config_name: Name of the configuration, used in the warning.
            row_preserving: Whether the caller needs the row count preserved.

        Returns:
            The drop_na value the layer should be built with.
        """
        drop_na = config.get("drop_na", True)
        if row_preserving and drop_na:
            if "drop_na" in config:
                logger.warning(
                    f"{config_name} for feature '{self.name}' requested "
                    "drop_na=True, but a preprocessing model must keep one output "
                    "row per input row or the features cannot be concatenated. "
                    "Building with drop_na=False; the warm-up rows are padded "
                    "instead, and can be discarded downstream.",
                )
            return False
        return drop_na

    def build_layers(self, row_preserving: bool = True) -> list:
        """Build the appropriate layers for this time series feature based on configuration.

        Args:
            row_preserving: When True (the default) the layers keep every input
                row, padding the warm-up positions instead of dropping them.
                A preprocessing model lays features out side by side, so a layer
                that removes its feature's leading rows leaves that column
                shorter than every other one and the concatenation fails. Pass
                False only when driving the returned layers directly.

        Returns:
            list: List of TensorFlow layers for time series preprocessing
        """
        from kdp.layers.time_series.lag_feature_layer import LagFeatureLayer
        from kdp.layers.time_series.rolling_stats_layer import RollingStatsLayer
        from kdp.layers.time_series.differencing_layer import DifferencingLayer
        from kdp.layers.time_series.moving_average_layer import MovingAverageLayer
        from kdp.layers.time_series.wavelet_transform_layer import WaveletTransformLayer
        from kdp.layers.time_series.tsfresh_feature_layer import TSFreshFeatureLayer
        from kdp.layers.time_series.calendar_feature_layer import CalendarFeatureLayer

        layers = []

        # Add lag layer if configured
        if self.lag_config and "lags" in self.lag_config:
            lags = self.lag_config.get("lags", [1])
            drop_na = self._resolve_drop_na(
                self.lag_config,
                "lag_config",
                row_preserving,
            )
            keep_original = self.lag_config.get("keep_original", True)
            fill_value = self.lag_config.get("fill_value", 0.0)

            layers.append(
                LagFeatureLayer(
                    lag_indices=lags,
                    drop_na=drop_na,
                    keep_original=keep_original,
                    fill_value=fill_value,
                    name=f"{self.name}_lag",
                ),
            )

        # Add rolling stats layer if configured
        if self.rolling_stats_config and "statistics" in self.rolling_stats_config:
            window_size = self.rolling_stats_config.get("window_size")
            statistics = self.rolling_stats_config.get("statistics")
            window_stride = self.rolling_stats_config.get("window_stride", 1)
            drop_na = self._resolve_drop_na(
                self.rolling_stats_config,
                "rolling_stats_config",
                row_preserving,
            )
            keep_original = self.rolling_stats_config.get("keep_original", True)
            pad_value = self.rolling_stats_config.get("pad_value", 0.0)

            layers.append(
                RollingStatsLayer(
                    window_size=window_size,
                    statistics=statistics,
                    window_stride=window_stride,
                    drop_na=drop_na,
                    keep_original=keep_original,
                    pad_value=pad_value,
                    name=f"{self.name}_rolling_stats",
                ),
            )

        # Add differencing layer if configured
        if self.differencing_config and "order" in self.differencing_config:
            order = self.differencing_config.get("order", 1)
            drop_na = self._resolve_drop_na(
                self.differencing_config,
                "differencing_config",
                row_preserving,
            )
            keep_original = self.differencing_config.get("keep_original", True)
            fill_value = self.differencing_config.get("fill_value", 0.0)

            layers.append(
                DifferencingLayer(
                    order=order,
                    drop_na=drop_na,
                    keep_original=keep_original,
                    fill_value=fill_value,
                    name=f"{self.name}_differencing",
                ),
            )

        # Add moving average layer if configured
        if self.moving_average_config and "periods" in self.moving_average_config:
            periods = self.moving_average_config.get("periods", [7])
            drop_na = self._resolve_drop_na(
                self.moving_average_config,
                "moving_average_config",
                row_preserving,
            )
            keep_original = self.moving_average_config.get("keep_original", True)
            pad_value = self.moving_average_config.get("pad_value", 0.0)

            layers.append(
                MovingAverageLayer(
                    periods=periods,
                    drop_na=drop_na,
                    keep_original=keep_original,
                    pad_value=pad_value,
                    name=f"{self.name}_moving_average",
                ),
            )

        # Add wavelet transform layer if configured
        if self.wavelet_transform_config:
            levels = self.wavelet_transform_config.get("levels", 3)
            window_sizes = self.wavelet_transform_config.get("window_sizes", None)
            keep_levels = self.wavelet_transform_config.get("keep_levels", "all")
            flatten_output = self.wavelet_transform_config.get("flatten_output", True)
            drop_na = self._resolve_drop_na(
                self.wavelet_transform_config,
                "wavelet_transform_config",
                row_preserving,
            )

            layers.append(
                WaveletTransformLayer(
                    levels=levels,
                    window_sizes=window_sizes,
                    keep_levels=keep_levels,
                    flatten_output=flatten_output,
                    drop_na=drop_na,
                    name=f"{self.name}_wavelet",
                ),
            )

        # Add TSFresh feature layer if configured
        if self.tsfresh_feature_config:
            features = self.tsfresh_feature_config.get(
                "features",
                ["mean", "std", "min", "max", "median"],
            )
            window_size = self.tsfresh_feature_config.get("window_size", None)
            stride = self.tsfresh_feature_config.get("stride", 1)
            drop_na = self._resolve_drop_na(
                self.tsfresh_feature_config,
                "tsfresh_feature_config",
                row_preserving,
            )
            normalize = self.tsfresh_feature_config.get("normalize", False)

            layers.append(
                TSFreshFeatureLayer(
                    features=features,
                    window_size=window_size,
                    stride=stride,
                    drop_na=drop_na,
                    normalize=normalize,
                    name=f"{self.name}_tsfresh",
                ),
            )

        # Add calendar feature layer if configured
        if self.calendar_feature_config:
            features = self.calendar_feature_config.get(
                "features",
                ["month", "day", "day_of_week", "is_weekend"],
            )
            cyclic_encoding = self.calendar_feature_config.get("cyclic_encoding", True)
            input_format = self.calendar_feature_config.get("input_format", "%Y-%m-%d")
            normalize = self.calendar_feature_config.get("normalize", True)

            layers.append(
                CalendarFeatureLayer(
                    features=features,
                    cyclic_encoding=cyclic_encoding,
                    input_format=input_format,
                    normalize=normalize,
                    name=f"{self.name}_calendar",
                ),
            )

        return layers

    def get_output_dim(self) -> int:
        """Calculate the output dimension of this feature after all transformations.

        The layers built by :meth:`build_layers` are applied in sequence, and each
        one that keeps its originals passes them through alongside its new
        columns. The widths therefore compose multiplicatively, not additively:
        two lags on a single column give 3 columns, and differencing that while
        keeping the originals gives 6, not 4.

        Returns:
            int: The number of columns the layer stack produces.
        """
        dim = 1

        if self.lag_config and "lags" in self.lag_config:
            lags = self.lag_config.get("lags", [1])
            keep_original = self.lag_config.get("keep_original", True)
            dim *= len(lags) + 1 if keep_original else len(lags)

        if self.rolling_stats_config and "statistics" in self.rolling_stats_config:
            statistics = self.rolling_stats_config.get("statistics", [])
            keep_original = self.rolling_stats_config.get("keep_original", True)
            dim *= len(statistics) + 1 if keep_original else len(statistics)

        if self.differencing_config and "order" in self.differencing_config:
            keep_original = self.differencing_config.get("keep_original", True)
            # Successive orders collapse into a single differenced block, so
            # keeping the originals doubles the width whatever the order is.
            if keep_original:
                dim *= 2

        if self.moving_average_config and "periods" in self.moving_average_config:
            periods = self.moving_average_config.get("periods", [7])
            keep_original = self.moving_average_config.get("keep_original", True)
            dim *= len(periods) + 1 if keep_original else len(periods)

        # The remaining transforms append their columns to whatever came before.
        if self.wavelet_transform_config:
            levels = self.wavelet_transform_config.get("levels", 3)
            keep_levels = self.wavelet_transform_config.get("keep_levels", "all")
            flatten_output = self.wavelet_transform_config.get("flatten_output", True)

            if not flatten_output:
                wavelet_dims = 1
            elif keep_levels == "all":
                wavelet_dims = levels
            elif isinstance(keep_levels, list):
                wavelet_dims = len(keep_levels)
            else:
                wavelet_dims = 1
            dim += wavelet_dims

        if self.tsfresh_feature_config:
            features = self.tsfresh_feature_config.get(
                "features",
                ["mean", "std", "min", "max", "median"],
            )
            dim += len(features)

        if self.calendar_feature_config:
            features = self.calendar_feature_config.get(
                "features",
                ["month", "day", "day_of_week", "is_weekend"],
            )
            cyclic_encoding = self.calendar_feature_config.get("cyclic_encoding", True)
            # Cyclic components are encoded as a sin/cos pair.
            cyclic_features = {
                "month",
                "day",
                "day_of_week",
                "quarter",
                "hour",
                "minute",
            }
            if cyclic_encoding:
                for feature in features:
                    dim += 2 if feature in cyclic_features else 1
            else:
                dim += len(features)

        return dim

    def to_dict(self) -> dict:
        """Convert the feature configuration to a dictionary.

        Returns:
            dict: Dictionary representation of the feature
        """
        return {
            "name": self.name,
            "feature_type": "time_series",
            "lag_config": self.lag_config,
            "rolling_stats_config": self.rolling_stats_config,
            "differencing_config": self.differencing_config,
            "moving_average_config": self.moving_average_config,
            "wavelet_transform_config": self.wavelet_transform_config,
            "tsfresh_feature_config": self.tsfresh_feature_config,
            "calendar_feature_config": self.calendar_feature_config,
            "sort_by": self.sort_by,
            "sort_ascending": self.sort_ascending,
            "group_by": self.group_by,
            "is_target": self.is_target,
            "exclude_from_input": self.exclude_from_input,
            "input_type": self.input_type,
        }

    @classmethod
    def from_dict(cls, feature_dict) -> "TimeSeriesFeature":
        """Create a TimeSeriesFeature from a dictionary representation.

        Args:
            feature_dict (dict): Dictionary representation of the feature

        Returns:
            TimeSeriesFeature: The created feature
        """
        # Extract only the keys that are used in the constructor
        allowed_keys = {
            "name",
            "feature_type",
            "lag_config",
            "rolling_stats_config",
            "differencing_config",
            "moving_average_config",
            "wavelet_transform_config",
            "tsfresh_feature_config",
            "calendar_feature_config",
            "sort_by",
            "sort_ascending",
            "group_by",
            "is_target",
            "exclude_from_input",
            "input_type",
        }

        constructor_args = {k: v for k, v in feature_dict.items() if k in allowed_keys}

        # Create and return the feature
        return cls(**constructor_args)


class FeatureSpaceConverter:
    def __init__(self) -> None:
        """Initialize a feature space converter."""
        self.features_space = {}
        self.numeric_features = []
        self.categorical_features = []
        self.text_features = []
        self.date_features = []
        self.passthrough_features = []
        self.time_series_features = []  # Add time series features list

    def _init_features_specs(
        self,
        features_specs: dict[str, FeatureType | str],
    ) -> dict[str, Feature]:
        """Format the features space into a dictionary.

        Args:
            features_specs: A dictionary with the features and their types,
                            where types can be specified as either FeatureType enums,
                            class instances (NumericalFeature, CategoricalFeature, TextFeature, DateFeature),
                            or strings.

        Returns:
            A dictionary with feature names as keys and Feature objects as values.
        """
        for name, spec in features_specs.items():
            # Direct instance check for standard pipelines
            if isinstance(
                spec,
                NumericalFeature
                | CategoricalFeature
                | TextFeature
                | DateFeature
                | PassthroughFeature
                | TimeSeriesFeature,  # Add TimeSeriesFeature to direct instance check
            ):
                feature_instance = spec
            else:
                # handling custom features pipelines
                if isinstance(spec, Feature):
                    feature_type = spec.feature_type
                else:
                    # Convert string to FeatureType if necessary
                    feature_type = (
                        FeatureType[spec.upper()] if isinstance(spec, str) else spec
                    )

                # Creating feature objects based on type
                if feature_type in {
                    FeatureType.FLOAT,
                    FeatureType.FLOAT_NORMALIZED,
                    FeatureType.FLOAT_RESCALED,
                    FeatureType.FLOAT_DISCRETIZED,
                }:
                    # Get preferred_distribution from kwargs if provided
                    preferred_distribution = (
                        spec.kwargs.get("preferred_distribution")
                        if isinstance(spec, Feature)
                        else None
                    )
                    feature_instance = NumericalFeature(
                        name=name,
                        feature_type=feature_type,
                        preferred_distribution=preferred_distribution,
                    )
                elif feature_type in {
                    FeatureType.INTEGER_CATEGORICAL,
                    FeatureType.STRING_CATEGORICAL,
                }:
                    feature_instance = CategoricalFeature(
                        name=name,
                        feature_type=feature_type,
                    )
                elif feature_type == FeatureType.TEXT:
                    feature_instance = TextFeature(name=name, feature_type=feature_type)
                elif feature_type == FeatureType.DATE:
                    feature_instance = DateFeature(name=name, feature_type=feature_type)
                elif feature_type == FeatureType.TIME_SERIES:
                    # Create TimeSeriesFeature instance
                    feature_instance = TimeSeriesFeature(
                        name=name,
                        feature_type=feature_type,
                    )
                elif feature_type == FeatureType.PASSTHROUGH:
                    # Get dtype from kwargs if provided
                    dtype = (
                        spec.kwargs.get("dtype", tf.float32)
                        if isinstance(spec, Feature)
                        else tf.float32
                    )
                    feature_instance = PassthroughFeature(
                        name=name,
                        feature_type=feature_type,
                        dtype=dtype,
                    )
                else:
                    raise ValueError(
                        f"Unsupported feature type for feature '{name}': {spec}",
                    )

            # Adding custom pipelines
            if isinstance(spec, Feature):
                logger.info(
                    f"Adding custom preprocessors to the object: {spec.preprocessors}",
                )
                feature_instance.preprocessors = spec.preprocessors
                feature_instance.kwargs = spec.kwargs

            # Categorize feature based on its class
            if isinstance(feature_instance, NumericalFeature):
                self.numeric_features.append(name)
            elif isinstance(feature_instance, CategoricalFeature):
                self.categorical_features.append(name)
            elif isinstance(feature_instance, TextFeature):
                self.text_features.append(name)
            elif isinstance(feature_instance, DateFeature):
                self.date_features.append(name)
            elif isinstance(feature_instance, TimeSeriesFeature):
                # Add to time series features
                self.time_series_features.append(name)
            elif isinstance(feature_instance, PassthroughFeature):
                # Add to passthrough features
                self.passthrough_features.append(name)

            # Adding formatted spec to the features_space dictionary
            self.features_space[name] = feature_instance

        return self.features_space
