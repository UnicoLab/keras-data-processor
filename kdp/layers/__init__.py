"""Custom Keras layers used by KDP preprocessing pipelines.

Importing this package registers every KDP layer with the Keras
serialization registry, which is what makes ``keras.models.load_model``
able to rebuild a saved preprocessing model without an explicit
``custom_objects`` mapping.
"""

from kdp.layers.cast_to_float import CastToFloat32Layer
from kdp.layers.date_encoding_layer import DateEncodingLayer
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.layers.distribution_aware_encoder_layer import (
    DistributionAwareEncoder,
    DistributionType,
)
from kdp.layers.distribution_transform_layer import DistributionTransformLayer
from kdp.layers.gated_linear_unit_layer import GatedLinearUnit
from kdp.layers.gated_residual_network_layer import GatedResidualNetwork
from kdp.layers.global_numerical_embedding_layer import GlobalNumericalEmbedding
from kdp.layers.multi_resolution_tabular_attention_layer import (
    MultiResolutionTabularAttention,
)
from kdp.layers.numerical_embedding_layer import NumericalEmbedding
from kdp.layers.preserve_dtype import PreserveDtypeLayer
from kdp.layers.season_layer import SeasonLayer
from kdp.layers.tabular_attention_layer import TabularAttention
from kdp.layers.text_preprocessing_layer import TextPreprocessingLayer
from kdp.layers.time_series import (
    AutoLagSelectionLayer,
    CalendarFeatureLayer,
    DifferencingLayer,
    FFTFeatureLayer,
    LagFeatureLayer,
    MissingValueHandlerLayer,
    MovingAverageLayer,
    RollingStatsLayer,
    SeasonalDecompositionLayer,
    TSFreshFeatureLayer,
    WaveletTransformLayer,
)
from kdp.layers.transformer_block_layer import TransformerBlock
from kdp.layers.variable_selection_layer import VariableSelection

__all__ = [
    "AutoLagSelectionLayer",
    "CalendarFeatureLayer",
    "CastToFloat32Layer",
    "DateEncodingLayer",
    "DateParsingLayer",
    "DifferencingLayer",
    "DistributionAwareEncoder",
    "DistributionTransformLayer",
    "DistributionType",
    "FFTFeatureLayer",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "GlobalNumericalEmbedding",
    "LagFeatureLayer",
    "MissingValueHandlerLayer",
    "MovingAverageLayer",
    "MultiResolutionTabularAttention",
    "NumericalEmbedding",
    "PreserveDtypeLayer",
    "RollingStatsLayer",
    "SeasonLayer",
    "SeasonalDecompositionLayer",
    "TSFreshFeatureLayer",
    "TabularAttention",
    "TextPreprocessingLayer",
    "TransformerBlock",
    "VariableSelection",
    "WaveletTransformLayer",
]
