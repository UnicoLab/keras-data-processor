"""The vocabulary collected from the data has to be the one the layer looks up.

`TextVectorization` standardizes before it splits: by default it lowercases and
strips punctuation. The statistics pass lowercased and split on whitespace but
kept the punctuation, so a column of ordinary prose produced a vocabulary of
"product," and "it!" while the layer, at inference, looked up "product" and
"it". Every punctuated word therefore landed in the single out-of-vocabulary
slot. The output width was right, the counts summed correctly, and roughly half
the words in a normal English sentence were being thrown away.

The same mismatch reached further: `ngrams` asks for word pairs the statistics
never recorded, and a custom `standardize` or `split` spells tokens differently
again. Those cases read the column and let the layer build its own vocabulary.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kdp import FeatureType, PreprocessingModel
from kdp.features import TextFeature
from kdp.processor import OutputModeOptions
from kdp.stats import DatasetStatistics

SENTENCES = [
    "Great product, loved it!",
    "Terrible. Would not buy.",
    "It's okay -- nothing special",
    "Great value!",
]


def _frame(repeats: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"txt": SENTENCES * repeats})


@pytest.mark.unit
class TestTextVocabularyMatchesTheLayer(unittest.TestCase):
    """What the statistics collect, and what the layer expects, are one list."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.frame = _frame()
        self.data = self.directory / "data.csv"
        self.frame.to_csv(self.data, index=False)

    def test_the_statistics_vocabulary_is_what_keras_would_build(self) -> None:
        statistics = DatasetStatistics(
            path_data=str(self.data),
            features_stats_path=str(self.directory / "stats.json"),
            features_specs={"txt": FeatureType.TEXT},
            overwrite_stats=True,
        ).main()
        collected = sorted(statistics["text"]["txt"]["vocab"])

        vectorizer = keras.layers.TextVectorization(output_mode="count")
        vectorizer.adapt(tf.constant(self.frame["txt"].values))
        adapted = sorted(
            str(word)
            for word in vectorizer.get_vocabulary()
            if word not in ("", "[UNK]")
        )

        self.assertEqual(collected, adapted)

    def _counts(self, **kwargs) -> np.ndarray:
        feature = TextFeature(
            name="txt",
            feature_type=FeatureType.TEXT,
            output_mode="count",
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
        return np.asarray(
            preprocessor.model(
                {"txt": tf.constant(np.array(SENTENCES).reshape(-1, 1))},
            )["txt"],
        )

    def test_no_word_of_the_training_text_falls_out_of_vocabulary(self) -> None:
        """Column 0 of a `count` output is the out-of-vocabulary bucket."""
        counts = self._counts()
        self.assertEqual(
            counts[:, 0].sum(),
            0.0,
            f"words went missing: {counts[:, 0].tolist()}",
        )

    def test_every_word_is_accounted_for(self) -> None:
        counts = self._counts()
        for sentence, row in zip(SENTENCES, counts, strict=True):
            # "--" is punctuation, and disappears under standardization.
            expected = len([w for w in sentence.split() if w.strip("-")])
            self.assertEqual(
                row.sum(),
                expected,
                f"{sentence!r} matched {row.sum()} of {expected} tokens",
            )


@pytest.mark.unit
class TestOptionsThatChangeWhatATokenIs(unittest.TestCase):
    """`ngrams` has to reach the vocabulary, not just the tokenizer."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        generator = np.random.default_rng(13)
        phrases = ["red car fast", "blue sky wide open", "green grass", "red red red"]
        self.frame = pd.DataFrame({"txt": generator.choice(phrases, 150)})
        self.data = self.directory / "data.csv"
        self.frame.to_csv(self.data, index=False)
        self.phrases = phrases

    def _width_and_oov(self, **kwargs) -> tuple[int, float]:
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
        result = np.asarray(
            preprocessor.model(
                {"txt": tf.constant(np.array(self.phrases).reshape(-1, 1))},
            )["txt"],
        )
        return int(result.shape[1]), float(result[:, 0].sum())

    def test_ngrams_widen_the_vocabulary_instead_of_hitting_the_oov_slot(self) -> None:
        unigram_width, _ = self._width_and_oov()
        bigram_width, bigram_oov = self._width_and_oov(ngrams=2)
        self.assertGreater(
            bigram_width,
            unigram_width,
            "asking for bigrams produced the unigram vocabulary",
        )
        self.assertEqual(bigram_oov, 0.0, "every bigram fell out of vocabulary")

    def test_the_width_matches_what_keras_builds_on_the_same_data(self) -> None:
        for ngrams in (None, 2, 3):
            with self.subTest(ngrams=ngrams):
                width, _ = self._width_and_oov(ngrams=ngrams)
                vectorizer = keras.layers.TextVectorization(
                    output_mode="multi_hot",
                    ngrams=ngrams,
                )
                vectorizer.adapt(tf.constant(self.frame["txt"].values))
                self.assertEqual(width, len(vectorizer.get_vocabulary()))


if __name__ == "__main__":
    unittest.main()
