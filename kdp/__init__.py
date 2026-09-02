# project plugin
from importlib.metadata import PackageNotFoundError, version as _package_version

from kdp.features import (
    CategoricalFeature,
    DateFeature,
    DistributionType,
    Feature,
    FeatureSpaceConverter,
    FeatureType,
    NumericalFeature,
    PassthroughFeature,
    TextFeature,
    TimeSeriesFeature,
)
from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor, Pipeline, ProcessingStep
from kdp.processor import (
    CategoryEncodingOptions,
    OutputModeOptions,
    PreprocessingModel,
    TabularAttentionPlacementOptions,
    TransformerBlockPlacementOptions,
)
from kdp.stats import DatasetStatistics
from kdp.auto_config import auto_configure
from kdp.inference.base import InferenceFormatter
from kdp.time_series.inference import TimeSeriesInferenceFormatter

try:
    __version__ = _package_version("kdp")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "ProcessingStep",
    "Pipeline",
    "FeaturePreprocessor",
    "Feature",
    "FeatureType",
    "FeatureSpaceConverter",
    "DistributionType",
    "NumericalFeature",
    "CategoricalFeature",
    "TextFeature",
    "DateFeature",
    "TimeSeriesFeature",
    "PassthroughFeature",
    "DatasetStatistics",
    "PreprocessorLayerFactory",
    "PreprocessingModel",
    "CategoryEncodingOptions",
    "TransformerBlockPlacementOptions",
    "OutputModeOptions",
    "TabularAttentionPlacementOptions",
    "auto_configure",
    "InferenceFormatter",
    "TimeSeriesInferenceFormatter",
]
