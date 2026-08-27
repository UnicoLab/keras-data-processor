import keras
import unittest
import numpy as np
import tensorflow as tf
import pytest

from kdp.layers_factory import PreprocessorLayerFactory
from kdp.layers.distribution_aware_encoder_layer import (
    DistributionAwareEncoder,
    DistributionType,
)
from kdp.layers.distribution_transform_layer import DistributionTransformLayer
from kdp.layers.text_preprocessing_layer import TextPreprocessingLayer
from kdp.layers.cast_to_float import CastToFloat32Layer
from kdp.layers.preserve_dtype import PreserveDtypeLayer
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.layers.date_encoding_layer import DateEncodingLayer
from kdp.layers.season_layer import SeasonLayer
from kdp.layers.transformer_block_layer import TransformerBlock
from kdp.layers.tabular_attention_layer import TabularAttention
from kdp.layers.multi_resolution_tabular_attention_layer import (
    MultiResolutionTabularAttention,
)
from kdp.layers.variable_selection_layer import VariableSelection
from kdp.layers.numerical_embedding_layer import NumericalEmbedding
from kdp.layers.global_numerical_embedding_layer import GlobalNumericalEmbedding
from kdp.layers.gated_linear_unit_layer import GatedLinearUnit
from kdp.layers.gated_residual_network_layer import GatedResidualNetwork


@pytest.mark.layers
@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.micro
class TestPreprocessorLayerFactory(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Create sample data with different distributions
        self.n_samples = 100
        # Log-normal distribution (right-skewed)
        self.log_normal_data = np.random.lognormal(
            mean=0, sigma=1, size=(self.n_samples, 1)
        ).astype(np.float32)
        # Standard normal distribution
        self.normal_data = np.random.normal(0, 1, size=(self.n_samples, 1)).astype(
            np.float32
        )
        # Multi-feature data
        self.multi_feature_data = np.random.normal(
            0, 1, size=(self.n_samples, 3)
        ).astype(np.float32)
        # Text data
        self.text_data = np.array(
            ["This is a test", "Another example", "Sample text data"]
        )
        # Date strings
        self.date_strings = np.array(["2023-01-01", "2023-02-15", "2023-12-31"])

    def test_create_layer(self):
        # Test with string class name
        dense_layer = PreprocessorLayerFactory.create_layer(
            layer_class="Dense", units=10, activation="relu"
        )
        self.assertIsInstance(dense_layer, keras.layers.Dense)
        self.assertEqual(dense_layer.units, 10)
        self.assertEqual(dense_layer.activation.__name__, "relu")

        # Test with class object
        dropout_layer = PreprocessorLayerFactory.create_layer(
            layer_class=keras.layers.Dropout, rate=0.5
        )
        self.assertIsInstance(dropout_layer, keras.layers.Dropout)
        self.assertEqual(dropout_layer.rate, 0.5)

    def test_distribution_aware_encoder(self):
        # We'll test specifically with embedding_dim to get a more predictable output shape
        encoder = PreprocessorLayerFactory.distribution_aware_encoder(
            name="test_encoder",
            num_bins=100,
            embedding_dim=32,  # Add embedding_dim for predictable output
            epsilon=1e-8,
            detect_periodicity=False,  # Disable periodicity to avoid shape changes
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=2,
            prefered_distribution=DistributionType.NORMAL,
        )
        self.assertIsInstance(encoder, DistributionAwareEncoder)
        self.assertEqual(encoder.name, "test_encoder")
        self.assertEqual(encoder.num_bins, 100)
        self.assertEqual(encoder.epsilon, 1e-8)
        self.assertEqual(encoder.embedding_dim, 32)
        self.assertEqual(encoder.detect_periodicity, False)
        self.assertEqual(encoder.handle_sparsity, True)
        self.assertEqual(encoder.prefered_distribution, DistributionType.NORMAL)

        # Test computation
        output = encoder(self.normal_data)
        # With embedding_dim=32, output shape should be (n_samples, 32)
        self.assertEqual(output.shape, (self.n_samples, 32))

    def test_distribution_transform_layer(self):
        # Test with log transform
        log_transform = PreprocessorLayerFactory.distribution_transform_layer(
            name="log_transform", transform_type="log", epsilon=1e-5
        )
        self.assertIsInstance(log_transform, DistributionTransformLayer)
        self.assertEqual(log_transform.name, "log_transform")
        self.assertEqual(log_transform.transform_type, "log")
        self.assertEqual(log_transform.epsilon, 1e-5)

        # Test computation
        output = log_transform(self.log_normal_data)
        self.assertEqual(output.shape, self.log_normal_data.shape)

        # Test with box-cox transform
        box_cox = PreprocessorLayerFactory.distribution_transform_layer(
            name="box_cox", transform_type="box-cox", lambda_param=0.5
        )
        self.assertEqual(box_cox.transform_type, "box-cox")
        self.assertEqual(box_cox.lambda_param, 0.5)

        # Test with auto transform
        auto_transform = PreprocessorLayerFactory.distribution_transform_layer(
            name="auto_transform",
            transform_type="auto",
            auto_candidates=["log", "sqrt"],
        )
        self.assertEqual(auto_transform.transform_type, "auto")
        self.assertEqual(auto_transform.auto_candidates, ["log", "sqrt"])

    def test_text_preprocessing_layer(self):
        text_layer = PreprocessorLayerFactory.text_preprocessing_layer(
            name="text_preproc", stop_words=["and", "the", "a"]
        )
        self.assertIsInstance(text_layer, TextPreprocessingLayer)
        self.assertEqual(text_layer.name, "text_preproc")
        self.assertEqual(text_layer.stop_words, ["and", "the", "a"])

    def test_cast_to_float32_layer(self):
        cast_layer = PreprocessorLayerFactory.cast_to_float32_layer(name="cast_layer")
        self.assertIsInstance(cast_layer, CastToFloat32Layer)

        # Test with integer data
        int_data = np.array([[1], [2], [3]], dtype=np.int32)
        output = cast_layer(int_data)
        self.assertEqual(output.dtype, tf.float32)

    def test_preserve_dtype_layer(self):
        # Test preserving original dtype
        preserve_layer = PreprocessorLayerFactory.preserve_dtype_layer(
            name="preserve_layer"
        )
        self.assertIsInstance(preserve_layer, PreserveDtypeLayer)

        # Test with integer data - should preserve int32
        int_data = np.array([[1], [2], [3]], dtype=np.int32)
        output = preserve_layer(int_data)
        self.assertEqual(output.dtype, tf.int32)

        # Test with target dtype
        cast_layer = PreprocessorLayerFactory.preserve_dtype_layer(
            name="cast_layer", target_dtype=tf.float32
        )
        self.assertIsInstance(cast_layer, PreserveDtypeLayer)

        # Test casting to float32
        output = cast_layer(int_data)
        self.assertEqual(output.dtype, tf.float32)

    def test_date_parsing_layer(self):
        date_parser = PreprocessorLayerFactory.date_parsing_layer(
            name="date_parser", date_format="YYYY-MM-DD"
        )
        self.assertIsInstance(date_parser, DateParsingLayer)
        self.assertEqual(date_parser.name, "date_parser")
        self.assertEqual(date_parser.date_format, "YYYY-MM-DD")

    def test_date_encoding_layer(self):
        date_encoder = PreprocessorLayerFactory.date_encoding_layer(
            name="date_encoder", output_parts=["year", "month", "day"]
        )
        self.assertIsInstance(date_encoder, DateEncodingLayer)
        self.assertEqual(date_encoder.name, "date_encoder")

    def test_date_season_layer(self):
        season_layer = PreprocessorLayerFactory.date_season_layer(name="season_layer")
        self.assertIsInstance(season_layer, SeasonLayer)
        self.assertEqual(season_layer.name, "season_layer")

    def test_transformer_block_layer(self):
        transformer = PreprocessorLayerFactory.transformer_block_layer(
            name="transformer", dim_model=32, num_heads=4, ff_units=64, dropout_rate=0.1
        )
        self.assertIsInstance(transformer, TransformerBlock)
        self.assertEqual(transformer.name, "transformer")
        self.assertEqual(transformer.d_model, 32)  # Different attribute name
        self.assertEqual(transformer.num_heads, 4)
        self.assertEqual(transformer.ff_units, 64)
        self.assertEqual(transformer.dropout_rate, 0.1)

    def test_tabular_attention_layer(self):
        tab_attention = PreprocessorLayerFactory.tabular_attention_layer(
            name="tab_attention", num_heads=4, d_model=32, dropout_rate=0.1
        )
        self.assertIsInstance(tab_attention, TabularAttention)
        self.assertEqual(tab_attention.name, "tab_attention")
        self.assertEqual(tab_attention.num_heads, 4)
        self.assertEqual(tab_attention.d_model, 32)
        self.assertEqual(tab_attention.dropout_rate, 0.1)

    def test_multi_resolution_attention_layer(self):
        multi_res_attention = PreprocessorLayerFactory.multi_resolution_attention_layer(
            name="multi_res_attention",
            num_heads=4,
            d_model=32,
            embedding_dim=16,
            dropout_rate=0.1,
        )
        self.assertIsInstance(multi_res_attention, MultiResolutionTabularAttention)
        self.assertEqual(multi_res_attention.name, "multi_res_attention")
        self.assertEqual(multi_res_attention.num_heads, 4)
        self.assertEqual(multi_res_attention.d_model, 32)
        self.assertEqual(multi_res_attention.embedding_dim, 16)
        self.assertEqual(multi_res_attention.dropout_rate, 0.1)

    def test_variable_selection_layer(self):
        var_selection = PreprocessorLayerFactory.variable_selection_layer(
            name="var_selection",
            nr_features=5,  # Use correct parameter name
            units=16,
        )
        self.assertIsInstance(var_selection, VariableSelection)
        self.assertEqual(var_selection.name, "var_selection")
        self.assertEqual(var_selection.nr_features, 5)
        self.assertEqual(var_selection.units, 16)

    def test_numerical_embedding_layer(self):
        num_embedding = PreprocessorLayerFactory.numerical_embedding_layer(
            name="num_embedding",
            embedding_dim=16,
            mlp_hidden_units=32,
            num_bins=20,
            init_min=-5.0,
            init_max=5.0,
            dropout_rate=0.2,
            use_batch_norm=True,
        )
        self.assertIsInstance(num_embedding, NumericalEmbedding)
        self.assertEqual(num_embedding.name, "num_embedding")
        self.assertEqual(num_embedding.embedding_dim, 16)
        self.assertEqual(num_embedding.mlp_hidden_units, 32)
        self.assertEqual(num_embedding.num_bins, 20)
        self.assertEqual(num_embedding.dropout_rate, 0.2)
        self.assertEqual(num_embedding.use_batch_norm, True)

        # Test computation
        output = num_embedding(self.normal_data)
        self.assertEqual(output.shape[-1], 16)  # Check output embedding dimension

    def test_global_numerical_embedding_layer(self):
        global_num_embedding = (
            PreprocessorLayerFactory.global_numerical_embedding_layer(
                name="global_num_embedding",
                global_embedding_dim=32,
                global_mlp_hidden_units=64,
                global_num_bins=15,
                global_init_min=-5.0,
                global_init_max=5.0,
                global_dropout_rate=0.1,
                global_use_batch_norm=True,
                global_pooling="average",
            )
        )
        self.assertIsInstance(global_num_embedding, GlobalNumericalEmbedding)
        self.assertEqual(global_num_embedding.name, "global_num_embedding")
        self.assertEqual(global_num_embedding.global_embedding_dim, 32)
        self.assertEqual(global_num_embedding.global_mlp_hidden_units, 64)
        self.assertEqual(global_num_embedding.global_num_bins, 15)
        self.assertEqual(global_num_embedding.global_dropout_rate, 0.1)
        self.assertEqual(global_num_embedding.global_use_batch_norm, True)
        self.assertEqual(global_num_embedding.global_pooling, "average")

        # Test computation
        output = global_num_embedding(self.multi_feature_data)
        self.assertEqual(output.shape[-1], 32)  # Check output embedding dimension

        # Test with max pooling
        max_pool_embedding = PreprocessorLayerFactory.global_numerical_embedding_layer(
            global_pooling="max"
        )
        self.assertEqual(max_pool_embedding.global_pooling, "max")

    def test_gated_linear_unit_layer(self):
        glu = PreprocessorLayerFactory.gated_linear_unit_layer(name="glu", units=16)
        self.assertIsInstance(glu, GatedLinearUnit)
        self.assertEqual(glu.name, "glu")
        self.assertEqual(glu.units, 16)

        # Test computation
        output = glu(self.normal_data)
        self.assertEqual(output.shape, (self.n_samples, 16))

    def test_gated_residual_network_layer(self):
        grn = PreprocessorLayerFactory.gated_residual_network_layer(
            name="grn", units=16, dropout_rate=0.2
        )
        self.assertIsInstance(grn, GatedResidualNetwork)
        self.assertEqual(grn.name, "grn")
        self.assertEqual(grn.units, 16)
        self.assertEqual(grn.dropout_rate, 0.2)

        # Test computation
        output = grn(self.normal_data)
        self.assertEqual(output.shape, (self.n_samples, 16))


if __name__ == "__main__":
    unittest.main()
