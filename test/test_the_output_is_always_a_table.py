"""Whatever the options, a preprocessor returns one row of numbers per input row.

A transformer block adds a sequence axis to a 2-D input and hands it back.
The categorical placement flattened that away; `transfo_placement="all_features"`
did not, so that one configuration returned (rows, 1, width). A Dense head
bolted onto the preprocessor sees a different rank depending on a flag that has
nothing to do with the output shape, and nothing said so.

Rank is the contract every downstream model depends on, so it is checked for
every model-level option rather than for the ones that happened to be tried.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel

ROWS = 80

SPECS = {
    "num1": FeatureType.FLOAT_NORMALIZED,
    "num2": FeatureType.FLOAT_RESCALED,
    "cat1": FeatureType.STRING_CATEGORICAL,
    "cat2": FeatureType.INTEGER_CATEGORICAL,
}

# Every model-level switch that reshapes the concatenated block, and the
# combinations that stack two of them.
CONFIGURATIONS = {
    "plain": {},
    "attention_all": {
        "tabular_attention": True,
        "tabular_attention_heads": 2,
        "tabular_attention_dim": 16,
        "tabular_attention_placement": "all_features",
    },
    "attention_numeric": {
        "tabular_attention": True,
        "tabular_attention_heads": 2,
        "tabular_attention_dim": 16,
        "tabular_attention_placement": "numeric",
    },
    "attention_categorical": {
        "tabular_attention": True,
        "tabular_attention_heads": 2,
        "tabular_attention_dim": 16,
        "tabular_attention_placement": "categorical",
    },
    "attention_multi_resolution": {
        "tabular_attention": True,
        "tabular_attention_heads": 2,
        "tabular_attention_dim": 16,
        "tabular_attention_placement": "multi_resolution",
    },
    "transformer_categorical": {
        "transfo_nr_blocks": 2,
        "transfo_nr_heads": 2,
        "transfo_ff_units": 8,
        "transfo_placement": "categorical",
    },
    "transformer_all_features": {
        "transfo_nr_blocks": 2,
        "transfo_nr_heads": 2,
        "transfo_ff_units": 8,
        "transfo_placement": "all_features",
    },
    "transformer_and_attention": {
        "tabular_attention": True,
        "tabular_attention_heads": 1,
        "tabular_attention_dim": 12,
        "tabular_attention_placement": "all_features",
        "transfo_nr_blocks": 1,
        "transfo_nr_heads": 2,
        "transfo_ff_units": 8,
        "transfo_placement": "all_features",
    },
    "feature_selection": {
        "feature_selection_placement": "all_features",
        "feature_selection_units": 8,
    },
    "feature_moe": {"use_feature_moe": True, "feature_moe_num_experts": 3},
    "global_numerical_embedding": {
        "use_global_numerical_embedding": True,
        "global_embedding_dim": 6,
    },
    "feature_crosses": {"feature_crosses": [("cat1", "cat2", 8)]},
}


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "num1": generator.normal(0.0, 1.0, ROWS),
            "num2": generator.uniform(0.0, 10.0, ROWS),
            "cat1": generator.choice(["a", "b", "c"], ROWS),
            "cat2": generator.integers(0, 4, ROWS),
        },
    )


@pytest.mark.unit
class TestTheOutputIsAlwaysATable(unittest.TestCase):
    """Rank 2, one row out per row in, for every option."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        cls.frame = _frame()
        cls.data = cls.directory / "data.csv"
        cls.frame.to_csv(cls.data, index=False)

    def _build_and_call(self, name: str, options: dict) -> np.ndarray:
        preprocessor = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / f"stats_{name}.json"),
            features_specs=SPECS,
            overwrite_stats=True,
            **options,
        )
        preprocessor.build_preprocessor()
        return np.asarray(
            preprocessor.model(
                {
                    column: tf.constant(self.frame[column].values.reshape(-1, 1))
                    for column in self.frame
                },
            ),
        )

    def test_every_configuration_returns_rows_by_width(self) -> None:
        wrong = {}
        for name, options in CONFIGURATIONS.items():
            result = self._build_and_call(name, options)
            if result.ndim != 2 or result.shape[0] != ROWS:
                wrong[name] = result.shape
        self.assertEqual(wrong, {}, f"not (rows, width): {wrong}")

    def test_every_configuration_returns_finite_numbers(self) -> None:
        wrong = [
            name
            for name, options in CONFIGURATIONS.items()
            if not np.isfinite(self._build_and_call(name, options)).all()
        ]
        self.assertEqual(wrong, [], f"non-finite output: {wrong}")


if __name__ == "__main__":
    unittest.main()
