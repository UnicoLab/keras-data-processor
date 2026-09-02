"""Tests for the transformer block.

The file existed but was empty, so the block shipped untested: it had no
`get_config`, no `build`, and Keras 3 warned on every model that used it that
the layer "does not have a `build()` method implemented and it looks like it
has unbuilt state ... which may cause failures down the line".
"""

import tempfile
import warnings
from pathlib import Path

import keras
import numpy as np
import pytest

from kdp.layers.transformer_block_layer import TransformerBlock


def _model(**kwargs):
    """A one-block model over a (4, 8) sequence."""
    keras.backend.clear_session()
    inputs = keras.Input(shape=(4, 8))
    return keras.Model(inputs, TransformerBlock(**kwargs)(inputs))


def test_defaults_are_recorded():
    """The constructor arguments are what `get_config` has to return."""
    block = TransformerBlock()
    assert block.d_model == 32
    assert block.num_heads == 3
    assert block.ff_units == 16
    assert block.dropout_rate == 0.2


def test_output_keeps_the_input_shape():
    """The residual connections require the width to survive the block."""
    model = _model(dim_model=8, num_heads=2, ff_units=16)
    assert model.output_shape == (None, 4, 8)


def test_two_dimensional_input_becomes_one_step():
    """`call` expands a 2D input, which is how the processor feeds it."""
    keras.backend.clear_session()
    inputs = keras.Input(shape=(8,))
    model = keras.Model(inputs, TransformerBlock(dim_model=8, num_heads=2)(inputs))
    assert model.output_shape == (None, 1, 8)


def test_building_emits_no_unbuilt_state_warning():
    """Keras warns about sub-layers created outside `build`; it must not here."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model(dim_model=8, num_heads=2, ff_units=16)
    unbuilt = [w for w in caught if "does not have a `build()`" in str(w.message)]
    assert unbuilt == [], [str(w.message) for w in unbuilt]


def test_a_width_mismatch_says_why():
    """The failure used to surface deep inside `Add` as a shape error."""
    keras.backend.clear_session()
    inputs = keras.Input(shape=(4, 16))
    with pytest.raises(ValueError, match="dim_model=8"):
        TransformerBlock(dim_model=8)(inputs)


def test_config_round_trip():
    """Every constructor argument has to survive `get_config`/`from_config`."""
    original = TransformerBlock(
        dim_model=16,
        num_heads=4,
        ff_units=64,
        dropout_rate=0.35,
        name="block",
    )
    restored = TransformerBlock.from_config(original.get_config())
    assert restored.d_model == 16
    assert restored.num_heads == 4
    assert restored.ff_units == 64
    assert restored.dropout_rate == 0.35
    assert restored.name == "block"


def test_saved_and_reloaded_model_is_identical():
    """A reloaded block must keep its size and reproduce its outputs."""
    model = _model(dim_model=8, num_heads=8, ff_units=64, dropout_rate=0.5)
    sample = np.random.default_rng(0).normal(size=(3, 4, 8)).astype("float32")
    before = model.predict(sample, verbose=0)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "block.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

    block = next(
        layer for layer in reloaded.layers if isinstance(layer, TransformerBlock)
    )
    assert (block.d_model, block.num_heads, block.ff_units, block.dropout_rate) == (
        8,
        8,
        64,
        0.5,
    )
    assert reloaded.count_params() == model.count_params()
    np.testing.assert_allclose(reloaded.predict(sample, verbose=0), before, atol=1e-5)


def test_ff_units_reach_the_feed_forward_layer():
    """A wider feed-forward layer has to show up in the parameter count."""
    narrow = _model(dim_model=8, num_heads=2, ff_units=16).count_params()
    wide = _model(dim_model=8, num_heads=2, ff_units=128).count_params()
    assert wide > narrow


def test_num_heads_reaches_the_attention():
    """More heads means more attention parameters at a fixed key dimension."""
    few = _model(dim_model=8, num_heads=2, ff_units=16).count_params()
    many = _model(dim_model=8, num_heads=8, ff_units=16).count_params()
    assert many > few


def test_dropout_only_applies_while_training():
    """Inference must be deterministic even at a punishing dropout rate."""
    model = _model(dim_model=8, num_heads=2, ff_units=16, dropout_rate=0.9)
    sample = np.random.default_rng(1).normal(size=(5, 4, 8)).astype("float32")
    np.testing.assert_allclose(
        model.predict(sample, verbose=0),
        model.predict(sample, verbose=0),
        atol=1e-6,
    )


def test_the_block_trains():
    """The parameters have to receive gradients, not merely exist."""
    model = _model(dim_model=8, num_heads=2, ff_units=16, dropout_rate=0.0)
    rng = np.random.default_rng(2)
    sample = rng.normal(size=(32, 4, 8)).astype("float32")
    target = rng.normal(size=(32, 4, 8)).astype("float32")

    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    before = model.evaluate(sample, target, verbose=0)
    model.fit(sample, target, epochs=3, batch_size=8, verbose=0)
    assert model.evaluate(sample, target, verbose=0) < before
