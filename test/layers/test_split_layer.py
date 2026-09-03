"""Tests for SplitLayer.

`SplitLayer` cuts the single concatenated tensor back into per-feature slices.
Every path that needs to see features individually goes through it -- the
feature mixture of experts, feature selection, the residual connections -- and
it was named in no test, so a wrong slice would have surfaced as a wrong model
rather than as a failure.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from kdp.processor import SplitLayer


def _inputs():
    """Two rows of six columns, so every slice is identifiable."""
    return tf.reshape(tf.range(12, dtype=tf.float32), (2, 6))


def test_widths_split_left_to_right():
    """A list of widths consumes the tensor in order, with no gaps."""
    parts = SplitLayer([2, 3, 1])(_inputs())
    assert [tuple(part.shape) for part in parts] == [(2, 2), (2, 3), (2, 1)]
    np.testing.assert_array_equal(parts[0].numpy()[0], [0.0, 1.0])
    np.testing.assert_array_equal(parts[1].numpy()[0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(parts[2].numpy()[0], [5.0])


def test_start_and_width_pairs_are_honoured():
    """The (start, width) form lets callers overlap or skip columns."""
    parts = SplitLayer([(0, 2), (2, 4)])(_inputs())
    assert [tuple(part.shape) for part in parts] == [(2, 2), (2, 4)]
    np.testing.assert_array_equal(parts[1].numpy()[0], [2.0, 3.0, 4.0, 5.0])


def test_no_dimensions_returns_the_input_untouched():
    """With nothing to split on, the tensor passes through as one piece."""
    parts = SplitLayer([])(_inputs())
    assert len(parts) == 1
    np.testing.assert_array_equal(parts[0].numpy(), _inputs().numpy())


def test_an_unusable_specification_is_rejected():
    """Silently returning the wrong slices would corrupt every feature."""
    with pytest.raises(ValueError, match="Invalid feature_dims"):
        SplitLayer(["two"])(_inputs())


def test_output_shape_matches_what_call_produces():
    """Keras builds the graph from this, so it has to agree with `call`."""
    layer = SplitLayer([2, 3, 1])
    declared = layer.compute_output_shape((None, 6))
    produced = [tuple(part.shape)[1:] for part in layer(_inputs())]
    assert [shape[1:] for shape in declared] == produced


def test_output_shape_for_pairs():
    """The (start, width) form has its own shape computation."""
    assert SplitLayer([(0, 2), (2, 4)]).compute_output_shape((None, 6)) == [
        (None, 2),
        (None, 4),
    ]


def test_output_shape_rejects_the_same_bad_specification():
    """`call` and `compute_output_shape` must fail on the same input."""
    with pytest.raises(ValueError, match="Invalid feature_dims"):
        SplitLayer(["two"]).compute_output_shape((None, 6))


def test_config_round_trip():
    """The split has to survive saving, or a reloaded model slices wrongly."""
    layer = SplitLayer([2, 3, 1], name="split")
    restored = SplitLayer.from_config(layer.get_config())
    assert restored.feature_dims == [2, 3, 1]
    assert restored.name == "split"


def test_it_works_inside_a_functional_model():
    """The real use is symbolic, not eager."""
    keras.backend.clear_session()
    inputs = keras.Input(shape=(6,))
    outputs = SplitLayer([2, 4])(inputs)
    model = keras.Model(inputs, outputs)
    parts = model.predict(np.arange(12, dtype="float32").reshape(2, 6), verbose=0)
    assert [part.shape for part in parts] == [(2, 2), (2, 4)]
