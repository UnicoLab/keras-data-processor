"""Tests for DynamicPreprocessingPipeline's dispatch rules.

Layers are addressed by name: a layer reads the entry that shares its name when
the data provides one, and otherwise consumes the previous layer's output.
"""

import keras
import unittest

import numpy as np
import tensorflow as tf

from kdp.dynamic_pipeline import DynamicPreprocessingPipeline
from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor


class ScalingLayer(keras.layers.Layer):
    """Multiplies its input by a constant factor."""

    def __init__(self, scaling_factor: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor

    def call(self, inputs):
        """Scale the inputs."""
        return inputs * self.scaling_factor

    def get_config(self):
        """Return the layer configuration."""
        config = super().get_config()
        config.update({"scaling_factor": self.scaling_factor})
        return config


def _flat(tensor) -> list:
    """Return a tensor as a flat list of Python floats."""
    return np.asarray(tensor).flatten().tolist()


class TestDynamicPipelineDispatch(unittest.TestCase):
    """The rule that decides what each layer consumes."""

    def test_layers_chain_when_only_the_first_key_is_supplied(self):
        """A layer with no entry of its own consumes the previous layer's output."""
        pipeline = DynamicPreprocessingPipeline(
            [ScalingLayer(2.0, name="scaling"), ScalingLayer(10.0, name="amplified")]
        )
        result = pipeline.transform({"scaling": tf.constant([[1.0], [2.0]])})

        self.assertEqual(_flat(result["scaling"]), [2.0, 4.0])
        # 1.0 -> *2 -> *10, i.e. chained rather than re-reading the raw input.
        self.assertEqual(_flat(result["amplified"]), [20.0, 40.0])

    def test_layers_run_independently_when_each_key_is_supplied(self):
        """A layer reads its own entry whenever the data provides one."""
        pipeline = DynamicPreprocessingPipeline(
            [ScalingLayer(2.0, name="scaling"), ScalingLayer(10.0, name="amplified")]
        )
        result = pipeline.transform(
            {"scaling": tf.constant([[1.0]]), "amplified": tf.constant([[3.0]])}
        )

        self.assertEqual(_flat(result["scaling"]), [2.0])
        # 3.0 * 10 -- taken from its own entry, not from the scaling output.
        self.assertEqual(_flat(result["amplified"]), [30.0])

    def test_dispatch_is_deterministic(self):
        """Repeated runs on the same inputs always route the same way."""
        pipeline = DynamicPreprocessingPipeline(
            [ScalingLayer(2.0, name="scaling"), ScalingLayer(10.0, name="amplified")]
        )
        observed = {
            tuple(
                _flat(
                    pipeline.transform(
                        {
                            "scaling": tf.constant([[1.0]]),
                            "amplified": tf.constant([[3.0]]),
                        }
                    )["scaling"]
                )
            )
            for _ in range(25)
        }
        self.assertEqual(observed, {(2.0,)})

    def test_input_dictionary_is_not_mutated(self):
        """Processing returns a new dictionary and leaves the caller's alone."""
        pipeline = DynamicPreprocessingPipeline([ScalingLayer(2.0, name="scaling")])
        features = {"scaling": tf.constant([[1.0]])}
        pipeline.transform(features)
        self.assertEqual(list(features), ["scaling"])

    def test_missing_input_raises_keyerror(self):
        """A layer with nothing to read fails loudly instead of being skipped."""
        pipeline = DynamicPreprocessingPipeline([ScalingLayer(name="scaling")])
        with self.assertRaises(KeyError) as ctx:
            pipeline.transform({"unrelated": tf.constant([[1.0]])})
        self.assertIn("scaling", str(ctx.exception))

    def test_duplicate_layer_names_are_rejected(self):
        """Names double as data keys, so they have to be unique."""
        with self.assertRaises(ValueError):
            DynamicPreprocessingPipeline(
                [ScalingLayer(name="dup"), ScalingLayer(name="dup")]
            )

    def test_initialize_and_transform_matches_transform(self):
        """The legacy alias behaves exactly like transform()."""
        pipeline = DynamicPreprocessingPipeline([ScalingLayer(3.0, name="scaling")])
        features = {"scaling": tf.constant([[2.0]])}
        self.assertEqual(
            _flat(pipeline.initialize_and_transform(features)["scaling"]),
            _flat(pipeline.transform(features)["scaling"]),
        )


class TestDynamicPipelineOverDatasets(unittest.TestCase):
    """The tf.data entry point."""

    def test_process_applies_the_same_chaining_rules(self):
        """Streaming through tf.data produces the same values as transform()."""
        pipeline = DynamicPreprocessingPipeline(
            [ScalingLayer(2.0, name="scaling"), ScalingLayer(10.0, name="amplified")]
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            {"scaling": np.array([[1.0], [2.0]], dtype=np.float32)}
        ).batch(2)

        batch = next(iter(pipeline.process(dataset).take(1)))
        self.assertEqual(_flat(batch["scaling"]), [2.0, 4.0])
        self.assertEqual(_flat(batch["amplified"]), [20.0, 40.0])


class TestFeaturePreprocessorDynamicMode(unittest.TestCase):
    """FeaturePreprocessor(use_dynamic=True) drives the dynamic pipeline."""

    def _build(self) -> FeaturePreprocessor:
        preprocessor = FeaturePreprocessor(name="feature", use_dynamic=True)
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.cast_to_float32_layer, name="cast"
        )
        preprocessor.add_processing_step(
            layer_class="Rescaling", scale=3.0, name="rescale"
        )
        return preprocessor

    def test_transform_returns_the_final_layer_output(self):
        """Dynamic transform chains every step instead of raising."""
        result = self._build().transform(tf.constant([[2]], dtype=tf.int32))
        self.assertEqual(_flat(result), [6.0])

    def test_chain_builds_a_usable_keras_graph(self):
        """Dynamic chain() returns a symbolic tensor usable in a Keras model."""
        preprocessor = self._build()
        inputs = keras.Input(shape=(1,), dtype=tf.int32)
        model = keras.Model(inputs, preprocessor.chain(inputs))
        self.assertEqual(_flat(model(tf.constant([[2]], dtype=tf.int32))), [6.0])

    def test_empty_dynamic_pipeline_is_a_passthrough(self):
        """With no steps added, the input flows through untouched."""
        preprocessor = FeaturePreprocessor(name="feature", use_dynamic=True)
        data = tf.constant([[5.0]])
        self.assertEqual(_flat(preprocessor.transform(data)), [5.0])


if __name__ == "__main__":
    unittest.main()
