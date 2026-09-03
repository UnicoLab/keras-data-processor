# API Reference

This section provides detailed API documentation extracted directly from the codebase.

## kdp.processor

- [CallableDict](api/processor_CallableDict.md)
- [FeatureSelectionPlacementOptions](api/processor_FeatureSelectionPlacementOptions.md)
- [OutputModeOptions](api/processor_OutputModeOptions.md)
- [PreprocessingModel](api/processor_PreprocessingModel.md)
- [SplitLayer](api/processor_SplitLayer.md)
- [TabularAttentionPlacementOptions](api/processor_TabularAttentionPlacementOptions.md)
- [TransformerBlockPlacementOptions](api/processor_TransformerBlockPlacementOptions.md)

## kdp.dynamic_pipeline

- [DynamicPreprocessingPipeline](api/dynamic_pipeline_DynamicPreprocessingPipeline.md)

## kdp.features

- [CategoricalFeature](api/features_CategoricalFeature.md)
- [CategoryEncodingOptions](api/features_CategoryEncodingOptions.md)
- [CrossFeatureOutputOptions](api/features_CrossFeatureOutputOptions.md)
- [DateFeature](api/features_DateFeature.md)
- [Feature](api/features_Feature.md)
- [FeatureSpaceConverter](api/features_FeatureSpaceConverter.md)
- [FeatureType](api/features_FeatureType.md)
- [NumericalFeature](api/features_NumericalFeature.md)
- [PassthroughFeature](api/features_PassthroughFeature.md)
- [TextFeature](api/features_TextFeature.md)
- [TextVectorizerOutputOptions](api/features_TextVectorizerOutputOptions.md)
- [TimeSeriesFeature](api/features_TimeSeriesFeature.md)

## kdp.stats

- [CategoricalAccumulator](api/stats_CategoricalAccumulator.md)
- [DatasetStatistics](api/stats_DatasetStatistics.md)
- [DateAccumulator](api/stats_DateAccumulator.md)
- [TextAccumulator](api/stats_TextAccumulator.md)
- [WelfordAccumulator](api/stats_WelfordAccumulator.md)

## kdp.auto_config


## kdp.model_advisor

- [ModelAdvisor](api/model_advisor_ModelAdvisor.md)

## kdp.moe

- [ExpertBlock](api/moe_ExpertBlock.md)
- [FeatureMoE](api/moe_FeatureMoE.md)
- [PadFeatureLayer](api/moe_PadFeatureLayer.md)
- [PerFeatureDense](api/moe_PerFeatureDense.md)
- [StackFeaturesLayer](api/moe_StackFeaturesLayer.md)
- [UnstackLayer](api/moe_UnstackLayer.md)

## kdp.pipeline

- [FeaturePreprocessor](api/pipeline_FeaturePreprocessor.md)
- [Pipeline](api/pipeline_Pipeline.md)
- [ProcessingStep](api/pipeline_ProcessingStep.md)

## kdp.layers_factory

- [PreprocessorLayerFactory](api/layers_factory_PreprocessorLayerFactory.md)

## kdp.inference.base

- [InferenceFormatter](api/base_InferenceFormatter.md)

## kdp.time_series.inference

- [TimeSeriesInferenceFormatter](api/inference_TimeSeriesInferenceFormatter.md)

## kdp.layers.cast_to_float

- [CastToFloat32Layer](api/cast_to_float_CastToFloat32Layer.md)

## kdp.layers.date_encoding_layer

- [DateEncodingLayer](api/date_encoding_layer_DateEncodingLayer.md)

## kdp.layers.date_parsing_layer

- [DateParsingLayer](api/date_parsing_layer_DateParsingLayer.md)

## kdp.layers.distribution_aware_encoder_layer

- [DistributionAwareEncoder](api/distribution_aware_encoder_layer_DistributionAwareEncoder.md)
- [DistributionType](api/distribution_aware_encoder_layer_DistributionType.md)

## kdp.layers.distribution_transform_layer

- [DistributionTransformLayer](api/distribution_transform_layer_DistributionTransformLayer.md)

## kdp.layers.gated_linear_unit_layer

- [GatedLinearUnit](api/gated_linear_unit_layer_GatedLinearUnit.md)

## kdp.layers.gated_residual_network_layer

- [GatedResidualNetwork](api/gated_residual_network_layer_GatedResidualNetwork.md)

## kdp.layers.global_numerical_embedding_layer

- [GlobalNumericalEmbedding](api/global_numerical_embedding_layer_GlobalNumericalEmbedding.md)

## kdp.layers.multi_resolution_tabular_attention_layer

- [MultiResolutionTabularAttention](api/multi_resolution_tabular_attention_layer_MultiResolutionTabularAttention.md)

## kdp.layers.numerical_embedding_layer

- [NumericalEmbedding](api/numerical_embedding_layer_NumericalEmbedding.md)

## kdp.layers.preserve_dtype

- [PreserveDtypeLayer](api/preserve_dtype_PreserveDtypeLayer.md)

## kdp.layers.season_layer

- [SeasonLayer](api/season_layer_SeasonLayer.md)

## kdp.layers.tabular_attention_layer

- [TabularAttention](api/tabular_attention_layer_TabularAttention.md)

## kdp.layers.text_preprocessing_layer

- [TextPreprocessingLayer](api/text_preprocessing_layer_TextPreprocessingLayer.md)

## kdp.layers.time_series.auto_lag_selection_layer

- [AutoLagSelectionLayer](api/auto_lag_selection_layer_AutoLagSelectionLayer.md)

## kdp.layers.time_series.calendar_feature_layer

- [CalendarFeatureLayer](api/calendar_feature_layer_CalendarFeatureLayer.md)

## kdp.layers.time_series.differencing_layer

- [DifferencingLayer](api/differencing_layer_DifferencingLayer.md)

## kdp.layers.time_series.fft_feature_layer

- [FFTFeatureLayer](api/fft_feature_layer_FFTFeatureLayer.md)

## kdp.layers.time_series.lag_feature_layer

- [LagFeatureLayer](api/lag_feature_layer_LagFeatureLayer.md)

## kdp.layers.time_series.missing_value_handler_layer

- [MissingValueHandlerLayer](api/missing_value_handler_layer_MissingValueHandlerLayer.md)

## kdp.layers.time_series.moving_average_layer

- [MovingAverageLayer](api/moving_average_layer_MovingAverageLayer.md)

## kdp.layers.time_series.rolling_stats_layer

- [RollingStatsLayer](api/rolling_stats_layer_RollingStatsLayer.md)

## kdp.layers.time_series.seasonal_decomposition_layer

- [SeasonalDecompositionLayer](api/seasonal_decomposition_layer_SeasonalDecompositionLayer.md)

## kdp.layers.time_series.tsfresh_feature_layer

- [TSFreshFeatureLayer](api/tsfresh_feature_layer_TSFreshFeatureLayer.md)

## kdp.layers.time_series.wavelet_transform_layer

- [WaveletTransformLayer](api/wavelet_transform_layer_WaveletTransformLayer.md)

## kdp.layers.transformer_block_layer

- [TransformerBlock](api/transformer_block_layer_TransformerBlock.md)

## kdp.layers.variable_selection_layer

- [VariableSelection](api/variable_selection_layer_VariableSelection.md)
