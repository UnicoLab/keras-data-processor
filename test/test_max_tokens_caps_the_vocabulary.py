"""`max_tokens` has to cap the vocabulary, which is the only reason to set it.

The statistics pass collects every distinct word in the column and the layer was
handed all of them alongside the cap. Keras refuses that outright -- "Attempted
to set a vocabulary larger than the maximum vocab size" -- so asking for a
vocabulary smaller than the data holds could not build at all, while asking for
a larger one worked and did nothing. The option was usable only where it had no
effect.

The statistics carry no word counts to choose the survivors with, so when the
cap bites the vectorizer adapts on the column instead, which is how `max_tokens`
picks its words everywhere else in Keras: by frequency.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.features import TextFeature
from kdp.processor import OutputModeOptions

ROWS = 150
SENTENCES = [
    "red car fast",
    "blue sky wide open",
    "green grass",
    "red red red",
]


@pytest.mark.unit
class TestMaxTokensCapsTheVocabulary(unittest.TestCase):
    """The output width has to follow the cap, above and below the data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        generator = np.random.default_rng(13)
        cls.frame = pd.DataFrame({"txt": generator.choice(SENTENCES, ROWS)})
        cls.data = cls.directory / "data.csv"
        cls.frame.to_csv(cls.data, index=False)
        # "red car fast blue sky wide open green grass" -- nine distinct words.
        cls.distinct_words = len(
            {word for sentence in SENTENCES for word in sentence.split()},
        )

    def _width(self, **kwargs) -> int:
        feature = TextFeature(
            name="txt",
            feature_type=FeatureType.TEXT,
            output_mode="multi_hot",
            **kwargs,
        )
        preprocessor = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / f"stats_{kwargs}.json"),
            features_specs={"txt": feature},
            overwrite_stats=True,
            output_mode=OutputModeOptions.DICT,
        )
        preprocessor.build_preprocessor()
        result = preprocessor.model(
            {"txt": tf.constant(self.frame["txt"].values[:5].reshape(-1, 1))},
        )
        return int(np.asarray(result["txt"]).shape[1])

    def test_a_cap_below_the_data_is_honoured(self) -> None:
        for cap in (3, 5, 7):
            with self.subTest(max_tokens=cap):
                self.assertLessEqual(self._width(max_tokens=cap), cap)

    def test_a_cap_below_the_data_still_produces_a_usable_vocabulary(self) -> None:
        self.assertGreater(self._width(max_tokens=4), 1)

    def test_a_cap_above_the_data_keeps_every_word(self) -> None:
        width = self._width(max_tokens=500)
        self.assertGreaterEqual(width, self.distinct_words)

    def test_the_int_mode_reserves_its_two_slots(self) -> None:
        """`int` keeps a padding slot as well as the out-of-vocabulary one."""
        feature = TextFeature(
            name="txt",
            feature_type=FeatureType.TEXT,
            output_mode="int",
            output_sequence_length=6,
            max_tokens=4,
        )
        preprocessor = PreprocessingModel(
            path_data=str(self.data),
            features_stats_path=str(self.directory / "stats_int.json"),
            features_specs={"txt": feature},
            overwrite_stats=True,
            output_mode=OutputModeOptions.DICT,
        )
        preprocessor.build_preprocessor()
        result = np.asarray(
            preprocessor.model(
                {"txt": tf.constant(self.frame["txt"].values[:5].reshape(-1, 1))},
            )["txt"],
        )
        self.assertEqual(result.shape[1], 6)
        self.assertLess(result.max(), 4, "a token index landed outside max_tokens")


if __name__ == "__main__":
    unittest.main()
