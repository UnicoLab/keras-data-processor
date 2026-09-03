"""Tests for the text output modes TextVectorization supports.

`output_sequence_length` is only meaningful for the "int" output mode. KDP used
to default it unconditionally, which made every other mode unreachable: Keras
rejects the combination outright.
"""

import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import kdp.features
import kdp.processor
from kdp import PreprocessingModel
from kdp.features import FeatureType, TextFeature


def _dataset(directory, rows: int = 150):
    """Write a small text column with a stable vocabulary."""
    rng = np.random.default_rng(3)
    csv_path = directory / "text.csv"
    pd.DataFrame(
        {
            "bio": rng.choice(
                ["the hello world foo", "data science rocks the world"], rows
            )
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _build(tmp_path, **kwargs):
    keras.backend.clear_session()
    preprocessor = PreprocessingModel(
        path_data=str(_dataset(tmp_path)),
        features_specs={
            "bio": TextFeature(name="bio", feature_type=FeatureType.TEXT, **kwargs),
        },
        features_stats_path=str(tmp_path / "stats.json"),
        overwrite_stats=True,
    )
    preprocessor.build_preprocessor()
    return preprocessor


@pytest.mark.unit
class TestTextOutputModes(unittest.TestCase):
    """Every mode TextVectorization can serve from a fixed vocabulary."""

    def test_int_mode_is_padded_to_the_default_length(self):
        """The default stays a 35-token integer sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp))
            output = preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
        self.assertEqual(int(output.shape[-1]), 35)

    def test_explicit_sequence_length_is_respected(self):
        """Passing it overrides the default rather than being ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), output_sequence_length=10)
            output = preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
        self.assertEqual(int(output.shape[-1]), 10)

    def test_multi_hot_mode_builds(self):
        """This used to raise because the default length was injected anyway."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), output_mode="multi_hot")
            output = np.asarray(
                preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
            )
        # One column per vocabulary entry, and it is an indicator vector.
        self.assertNotEqual(output.shape[-1], 35)
        self.assertTrue(set(np.unique(output)).issubset({0.0, 1.0}))

    def test_count_mode_builds(self):
        """Same defect, same fix."""
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(Path(tmp), output_mode="count")
            output = np.asarray(
                preprocessor.model({"bio": tf.constant([["the hello world foo"]])})
            )
        self.assertNotEqual(output.shape[-1], 35)
        self.assertGreater(output.sum(), 0)

    def test_stop_words_are_removed(self):
        """A stop word must not survive into the encoded sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = _build(Path(tmp))
            plain_out = np.asarray(
                plain.model({"bio": tf.constant([["the hello world foo"]])})
            )
        with tempfile.TemporaryDirectory() as tmp:
            filtered = _build(Path(tmp), stop_words=["the"])
            filtered_out = np.asarray(
                filtered.model({"bio": tf.constant([["the hello world foo"]])})
            )
        self.assertFalse(np.allclose(plain_out, filtered_out))


if __name__ == "__main__":
    unittest.main()


@pytest.mark.unit
class TestCategoryEncodingIsValidated(unittest.TestCase):
    """A miscased encoding used to degrade the feature in silence."""

    def test_lowercase_encoding_is_accepted(self):
        """`"hashing"` now means HASHING rather than nothing at all."""
        from kdp.features import CategoricalFeature

        feature = CategoricalFeature(
            name="city",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding="hashing",
        )
        self.assertEqual(feature.category_encoding, "HASHING")

    def test_unknown_encoding_raises(self):
        """Anything that is not an option is refused up front."""
        from kdp.features import CategoricalFeature

        with self.assertRaises(ValueError) as ctx:
            CategoricalFeature(
                name="city",
                feature_type=FeatureType.STRING_CATEGORICAL,
                category_encoding="nonsense",
            )
        self.assertIn("category_encoding", str(ctx.exception))

    def test_lowercase_hashing_actually_hashes(self):
        """Previously this produced a width-1 lookup index instead of buckets."""
        from kdp.features import CategoricalFeature

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rng = np.random.default_rng(7)
            csv_path = tmp_path / "cat.csv"
            pd.DataFrame({"city": rng.choice(["a", "b", "c"], 100)}).to_csv(
                csv_path, index=False
            )
            keras.backend.clear_session()
            preprocessor = PreprocessingModel(
                path_data=str(csv_path),
                features_specs={
                    "city": CategoricalFeature(
                        name="city",
                        feature_type=FeatureType.STRING_CATEGORICAL,
                        category_encoding="hashing",
                        hash_bucket_size=8,
                    ),
                },
                features_stats_path=str(tmp_path / "stats.json"),
                overwrite_stats=True,
            )
            preprocessor.build_preprocessor()
            output = preprocessor.model({"city": tf.constant([["a"]])})
        self.assertEqual(int(output.shape[-1]), 8)


@pytest.mark.unit
class TestPlacementOptionsAreValidated(unittest.TestCase):
    """A placement that matches nothing used to disable the feature in silence."""

    def test_wrong_placement_raises_instead_of_doing_nothing(self):
        """`"all"` is not `"all_features"`, and never was."""
        with self.assertRaises(ValueError) as ctx:
            PreprocessingModel(features_specs={}, feature_selection_placement="all")
        self.assertIn("feature_selection_placement", str(ctx.exception))

    def test_valid_placements_are_accepted(self):
        """Every documented placement is a real option."""
        for placement in (
            "none",
            "numeric",
            "categorical",
            "text",
            "date",
            "all_features",
        ):
            model = PreprocessingModel(
                features_specs={}, feature_selection_placement=placement
            )
            self.assertEqual(model.feature_selection_placement, placement)

    def test_casing_is_normalised(self):
        """An upper-case spelling resolves rather than silently missing."""
        model = PreprocessingModel(
            features_specs={}, feature_selection_placement="ALL_FEATURES"
        )
        self.assertEqual(model.feature_selection_placement, "all_features")

    def test_attention_placement_is_validated_too(self):
        """The same comparison-by-string trap applies here."""
        with self.assertRaises(ValueError):
            PreprocessingModel(
                features_specs={}, tabular_attention_placement="everything"
            )


@pytest.mark.unit
class TestTextVectorizerOutputOptions(unittest.TestCase):
    """The enum that names the text output modes.

    `kdp.features` and `kdp.processor` each defined a class of this name, one
    with `auto()` integers and one with the real strings. Importing the wrong
    one produced an `output_mode` that no comparison in KDP matched and that
    `TextVectorization` could not use.
    """

    def test_both_modules_expose_the_same_class(self):
        """Two classes of one name is how the values diverged."""
        self.assertIs(
            kdp.features.TextVectorizerOutputOptions,
            kdp.processor.TextVectorizerOutputOptions,
        )

    def test_members_are_the_strings_keras_accepts(self):
        """The value has to be usable as `output_mode` on its own."""
        options = kdp.features.TextVectorizerOutputOptions
        self.assertEqual(options.TF_IDF, "tf_idf")
        self.assertEqual(options.INT, "int")
        self.assertEqual(options.MULTI_HOT, "multi_hot")

    def test_a_crossed_feature_is_integer_coded(self):
        """Crosses are hashed into bins, so `int` is the only honest mode."""
        self.assertEqual(kdp.features.CrossFeatureOutputOptions.INT, "int")

    def test_the_enum_works_where_the_string_does(self):
        """A caller reaching for the enum must get the documented behaviour."""
        options = kdp.features.TextVectorizerOutputOptions
        with tempfile.TemporaryDirectory() as tmp:
            preprocessor = _build(
                Path(tmp),
                output_mode=options.MULTI_HOT,
                max_tokens=16,
            )
        self.assertIsNotNone(preprocessor.model)
