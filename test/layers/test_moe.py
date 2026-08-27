"""
Test suite for Feature-wise Mixture of Experts (MoE) implementation.

This file tests the functionality of the FeatureMoE, StackFeaturesLayer,
UnstackLayer, and ExpertBlock classes.
"""

import unittest
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
import tempfile

from kdp.moe import FeatureMoE, StackFeaturesLayer, UnstackLayer, ExpertBlock
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, FeatureType
from test.test_processor import generate_fake_data


class TestStackUnstackLayers(unittest.TestCase):
    """Test the StackFeaturesLayer and UnstackLayer functionality."""

    def test_stack_features_layer(self):
        """Test that StackFeaturesLayer correctly stacks features."""
        # Create test inputs
        batch_size = 8
        feature_dim = 4
        num_features = 3

        inputs = [
            tf.random.normal(shape=(batch_size, feature_dim))
            for _ in range(num_features)
        ]

        # Create and apply the layer
        stack_layer = StackFeaturesLayer()
        output = stack_layer(inputs)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_features, feature_dim))

        # Check values - should match original inputs when indexed
        for i in range(num_features):
            np.testing.assert_array_equal(output[:, i, :].numpy(), inputs[i].numpy())

    def test_unstack_layer(self):
        """Test that UnstackLayer correctly unstacks features."""
        # Create a test input tensor to unstack
        batch_size = 8
        num_features = 3
        feature_dim = 4

        stacked_input = tf.random.normal(shape=(batch_size, num_features, feature_dim))

        # Create and apply the layer
        unstack_layer = UnstackLayer(axis=1)
        outputs = unstack_layer(stacked_input)

        # Check output format and shapes
        self.assertEqual(len(outputs), num_features)
        for output in outputs:
            self.assertEqual(output.shape, (batch_size, feature_dim))

        # Check round-trip consistency
        stack_layer = StackFeaturesLayer()
        restacked = stack_layer(outputs)
        np.testing.assert_array_equal(restacked.numpy(), stacked_input.numpy())

    def test_stack_unstack_config(self):
        """Test that stack and unstack layers have proper config."""
        # Test StackFeaturesLayer config
        stack_layer = StackFeaturesLayer(name="test_stack")
        config = stack_layer.get_config()
        self.assertEqual(config["name"], "test_stack")

        # Recreate from config
        recreated_stack = StackFeaturesLayer.from_config(config)
        self.assertEqual(recreated_stack.name, "test_stack")

        # Test UnstackLayer config
        unstack_layer = UnstackLayer(axis=2, name="test_unstack")
        config = unstack_layer.get_config()
        self.assertEqual(config["name"], "test_unstack")
        self.assertEqual(config["axis"], 2)

        # Recreate from config
        recreated_unstack = UnstackLayer.from_config(config)
        self.assertEqual(recreated_unstack.name, "test_unstack")
        self.assertEqual(recreated_unstack.axis, 2)


class TestExpertBlock(unittest.TestCase):
    """Test the ExpertBlock functionality."""

    def test_expert_initialization(self):
        """Test that an expert block can be initialized with various configurations."""
        # Basic initialization
        expert = ExpertBlock(expert_dim=32)
        self.assertEqual(expert.expert_dim, 32)
        self.assertEqual(
            len(expert.hidden_layers), 6
        )  # 2 layers with activation, BN, no dropout

        # With custom hidden dims
        expert = ExpertBlock(expert_dim=32, hidden_dims=[64, 48, 32])
        self.assertEqual(
            len(expert.hidden_layers), 9
        )  # 3 layers with activation, BN, no dropout

        # With dropout
        expert = ExpertBlock(expert_dim=32, dropout_rate=0.2)
        self.assertEqual(
            len(expert.hidden_layers), 8
        )  # 2 layers with activation, BN, dropout

        # Without batch normalization
        expert = ExpertBlock(expert_dim=32, use_batch_norm=False)
        self.assertEqual(
            len(expert.hidden_layers), 4
        )  # 2 layers with activation, no BN, no dropout

    def test_expert_forward_pass(self):
        """Test the forward pass through an expert block."""
        batch_size = 16
        input_dim = 8
        expert_dim = 32

        # Create an expert
        expert = ExpertBlock(expert_dim=expert_dim)

        # Test input
        inputs = tf.random.normal(shape=(batch_size, input_dim))

        # Forward pass
        outputs = expert(inputs, training=True)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, expert_dim))

        # Test in inference mode
        outputs_inference = expert(inputs, training=False)
        self.assertEqual(outputs_inference.shape, (batch_size, expert_dim))

    def test_expert_training(self):
        """Test that an expert block can be trained."""
        # Create a simple model with an expert
        inputs = keras.layers.Input(shape=(10,))
        expert = ExpertBlock(expert_dim=1)
        outputs = expert(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)

        # Train for a few steps
        initial_weights = model.get_weights()
        model.fit(x, y, epochs=1, verbose=0)
        final_weights = model.get_weights()

        # Verify weights changed (model trained)
        self.assertFalse(
            all(
                np.array_equal(w1, w2) for w1, w2 in zip(initial_weights, final_weights)
            )
        )

    def test_expert_config(self):
        """Test expert block configuration and serialization."""
        expert = ExpertBlock(
            expert_dim=48,
            hidden_dims=[64, 56],
            activation="gelu",
            dropout_rate=0.3,
            use_batch_norm=True,
            name="test_expert",
        )

        config = expert.get_config()
        self.assertEqual(config["expert_dim"], 48)
        self.assertEqual(config["hidden_dims"], [64, 56])
        self.assertEqual(config["activation"], "gelu")
        self.assertEqual(config["dropout_rate"], 0.3)
        self.assertTrue(config["use_batch_norm"])
        self.assertEqual(config["name"], "test_expert")

        # Recreate from config
        recreated_expert = ExpertBlock.from_config(config)
        self.assertEqual(recreated_expert.expert_dim, 48)
        self.assertEqual(recreated_expert.hidden_dims, [64, 56])
        self.assertEqual(recreated_expert.activation, "gelu")
        self.assertEqual(recreated_expert.dropout_rate, 0.3)
        self.assertTrue(recreated_expert.use_batch_norm)
        self.assertEqual(recreated_expert.name, "test_expert")


class TestFeatureMoE(unittest.TestCase):
    """Test the FeatureMoE layer functionality."""

    def test_moe_initialization(self):
        """Test initialization of FeatureMoE with different configurations."""
        # Basic learned routing
        moe = FeatureMoE(num_experts=4, expert_dim=32)
        self.assertEqual(moe.num_experts, 4)
        self.assertEqual(moe.expert_dim, 32)
        self.assertEqual(moe.routing, "learned")
        self.assertEqual(len(moe.experts), 4)

        # Predefined routing
        feature_names = ["feat1", "feat2", "feat3"]
        assignments = {"feat1": 0, "feat2": 1, "feat3": 2}
        moe = FeatureMoE(
            num_experts=3,
            expert_dim=32,
            routing="predefined",
            feature_names=feature_names,
            predefined_assignments=assignments,
        )
        self.assertEqual(moe.routing, "predefined")
        self.assertEqual(moe.feature_names, feature_names)
        self.assertEqual(moe.predefined_assignments, assignments)

        # Check that assignment matrix was created
        self.assertIsNotNone(getattr(moe, "assignment_matrix", None))

        # Check for value error when predefined routing lacks feature names
        with self.assertRaises(ValueError):
            moe = FeatureMoE(num_experts=3, routing="predefined")

    def test_learned_routing(self):
        """Test learned routing mechanism in FeatureMoE."""
        batch_size = 16
        num_features = 5
        feature_dim = 8
        num_experts = 4
        expert_dim = 32

        # Create inputs
        inputs = tf.random.normal(shape=(batch_size, num_features, feature_dim))

        # Create MoE with learned routing
        moe = FeatureMoE(
            num_experts=num_experts, expert_dim=expert_dim, routing="learned"
        )

        # Forward pass
        outputs = moe(inputs, training=True)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, num_features, expert_dim))

        # Get router weights - should distribute across all experts
        router_weights = moe.router.kernel  # [feature_dim, num_experts]
        self.assertEqual(router_weights.shape, (feature_dim, num_experts))

    def test_predefined_routing(self):
        """Test predefined routing in FeatureMoE."""
        batch_size = 16
        num_features = 3
        feature_dim = 8
        num_experts = 3
        expert_dim = 32

        # Create inputs
        inputs = tf.random.normal(shape=(batch_size, num_features, feature_dim))

        # Define feature names and assignments
        feature_names = ["feat1", "feat2", "feat3"]
        assignments = {"feat1": 0, "feat2": 1, "feat3": 2}

        # Create MoE with predefined routing
        moe = FeatureMoE(
            num_experts=num_experts,
            expert_dim=expert_dim,
            routing="predefined",
            feature_names=feature_names,
            predefined_assignments=assignments,
        )

        # Forward pass
        outputs = moe(inputs, training=True)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, num_features, expert_dim))

        # Check assignment matrix
        assignment_matrix = moe.assignment_matrix.numpy()
        self.assertEqual(assignment_matrix.shape, (num_features, num_experts))

        # Each feature should be assigned to exactly one expert in our test case
        for i, feature in enumerate(feature_names):
            expert_idx = assignments[feature]
            # Expert assignment should be 1.0 for assigned expert, 0 elsewhere
            self.assertEqual(assignment_matrix[i, expert_idx], 1.0)
            # Sum of assignments for each feature should be 1.0
            self.assertEqual(np.sum(assignment_matrix[i]), 1.0)

    def test_sparse_routing(self):
        """Test sparse routing in FeatureMoE."""
        batch_size = 16
        num_features = 5
        feature_dim = 8
        num_experts = 6
        expert_dim = 32
        sparsity = 2  # Only use top 2 experts per feature

        # Create inputs
        inputs = tf.random.normal(shape=(batch_size, num_features, feature_dim))

        # Create MoE with sparse routing
        moe = FeatureMoE(
            num_experts=num_experts,
            expert_dim=expert_dim,
            routing="learned",
            sparsity=sparsity,
            routing_activation="sparse",  # This should trigger sparse routing
        )

        # Forward pass
        outputs = moe(inputs, training=True)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, num_features, expert_dim))

    def test_expert_freeze(self):
        """Test freezing experts in FeatureMoE."""
        # This is harder to test directly, but we can verify the behavior:
        batch_size = 16
        num_features = 5
        feature_dim = 8
        num_experts = 4
        expert_dim = 32

        # Create inputs
        inputs = tf.random.normal(shape=(batch_size, num_features, feature_dim))

        # Create MoE with frozen experts
        moe = FeatureMoE(
            num_experts=num_experts, expert_dim=expert_dim, freeze_experts=True
        )

        # Forward pass in training mode
        outputs = moe(inputs, training=True)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, num_features, expert_dim))

    def test_get_expert_assignments(self):
        """Test getting expert assignments from FeatureMoE."""
        # For predefined routing
        feature_names = ["feat1", "feat2", "feat3"]
        assignments = {"feat1": 0, "feat2": 1, "feat3": 2}

        moe = FeatureMoE(
            num_experts=3,
            expert_dim=32,
            routing="predefined",
            feature_names=feature_names,
            predefined_assignments=assignments,
        )

        retrieved_assignments = moe.get_expert_assignments()
        self.assertEqual(retrieved_assignments, assignments)

    def test_moe_config(self):
        """Test FeatureMoE configuration and serialization."""
        # Create a complex MoE with predefined routing
        feature_names = ["feat1", "feat2", "feat3"]
        assignments = {"feat1": 0, "feat2": 1, "feat3": 2}

        moe = FeatureMoE(
            num_experts=4,
            expert_dim=48,
            expert_hidden_dims=[64, 56],
            routing="predefined",
            sparsity=2,
            routing_activation="softmax",
            feature_names=feature_names,
            predefined_assignments=assignments,
            freeze_experts=True,
            dropout_rate=0.2,
            use_batch_norm=True,
            name="test_moe",
        )

        config = moe.get_config()
        self.assertEqual(config["num_experts"], 4)
        self.assertEqual(config["expert_dim"], 48)
        self.assertEqual(config["expert_hidden_dims"], [64, 56])
        self.assertEqual(config["routing"], "predefined")
        self.assertEqual(config["sparsity"], 2)
        self.assertEqual(config["feature_names"], feature_names)
        self.assertEqual(config["predefined_assignments"], assignments)
        self.assertEqual(config["name"], "test_moe")

        # Test a learned routing config
        moe_learned = FeatureMoE(
            num_experts=4, expert_dim=48, routing="learned", name="test_learned_moe"
        )

        config_learned = moe_learned.get_config()
        self.assertEqual(config_learned["routing"], "learned")
        self.assertNotIn("feature_names", config_learned)
        self.assertNotIn("predefined_assignments", config_learned)


class TestMoEIntegration(unittest.TestCase):
    """Test integration of Feature-wise MoE in models."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.data_path = self.base_path / "test_data.csv"
        self.features_stats_path = self.base_path / "features_stats.json"

        # Define features for testing
        self.features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
        }

        # Generate fake data
        df = generate_fake_data(self.features, num_rows=100)
        df.to_csv(self.data_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_moe_in_keras_model(self):
        """Test FeatureMoE layer in a pure Keras model."""
        # Create a simple keras model with MoE
        inputs = [keras.layers.Input(shape=(1,), name=f"input_{i}") for i in range(5)]

        # Add some processing layers to get feature representations
        processed = [
            keras.layers.Dense(8, activation="relu")(input_tensor)
            for input_tensor in inputs
        ]

        # Stack features for MoE
        stacked = StackFeaturesLayer()(processed)

        # Apply MoE
        moe = FeatureMoE(num_experts=3, expert_dim=16)
        moe_output = moe(stacked)

        # Unstack features
        unstacked = UnstackLayer()(moe_output)

        # Create model
        model = keras.Model(inputs=inputs, outputs=unstacked)

        # Compile and verify
        model.compile(optimizer="adam", loss="mse")

        # Check model structure - accounting for expert layers within MoE
        self.assertEqual(
            len(model.layers), 13
        )  # 5 inputs, 5 dense, stack, MoE (with experts), unstack
        self.assertEqual(len(model.outputs), 5)

        # Test with random data
        test_inputs = [np.random.randn(10, 1) for _ in range(5)]
        outputs = model.predict(test_inputs)

        # Check outputs
        self.assertEqual(len(outputs), 5)
        for output in outputs:
            self.assertEqual(output.shape, (10, 16))

    def test_moe_with_preprocessing_model(self):
        """Test FeatureMoE in a PreprocessingModel."""
        # Create preprocessor with MoE enabled
        ppr = PreprocessingModel(
            path_data=str(self.data_path),
            features_specs=self.features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_feature_moe=True,
            feature_moe_num_experts=3,
            feature_moe_expert_dim=16,
            feature_moe_routing="learned",
            output_mode="concat",
        )

        # Build preprocessor
        result = ppr.build_preprocessor()
        model = result["model"]

        # Verify model contains MoE layer
        moe_layers = [layer for layer in model.layers if isinstance(layer, FeatureMoE)]
        self.assertEqual(len(moe_layers), 1)

        # Test model on sample data
        test_data = generate_fake_data(self.features, num_rows=10)
        input_dict = {
            name: test_data[name].values.reshape(-1, 1) for name in self.features.keys()
        }

        # Try prediction
        output = model.predict(input_dict)
        self.assertIsNotNone(output)

        # Verify output shape
        expected_dim = 16 * len(self.features)  # expert_dim * num_features
        self.assertEqual(output.shape[1], expected_dim)

    def test_moe_save_load(self):
        """Test saving and loading models with FeatureMoE."""
        # Create preprocessor with MoE enabled
        ppr = PreprocessingModel(
            path_data=str(self.data_path),
            features_specs=self.features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_feature_moe=True,
            feature_moe_num_experts=3,
            feature_moe_expert_dim=16,
            feature_moe_routing="learned",
            feature_moe_sparsity=2,
            output_mode="concat",
        )

        # Build preprocessor
        result = ppr.build_preprocessor()

        # Verify the model was built successfully
        self.assertIn("model", result)

        # Save model
        save_path = self.base_path / "test_moe_model"
        ppr.save_model(save_path)

        # Load model
        loaded_model, metadata = PreprocessingModel.load_model(save_path)

        # Verify loaded model has MoE metadata
        self.assertTrue(metadata.get("use_feature_moe", False))
        self.assertIn("feature_moe_config", metadata)

        # Check specific MoE configuration was preserved
        moe_config = metadata.get("feature_moe_config", {})
        self.assertEqual(moe_config.get("num_experts"), 3)
        self.assertEqual(moe_config.get("expert_dim"), 16)
        self.assertEqual(moe_config.get("routing"), "learned")
        self.assertEqual(moe_config.get("sparsity"), 2)

        # Test loaded model on sample data
        test_data = generate_fake_data(self.features, num_rows=10)
        input_dict = {
            name: test_data[name].values.reshape(-1, 1) for name in self.features.keys()
        }

        # Try prediction with loaded model
        output = loaded_model.predict(input_dict)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
