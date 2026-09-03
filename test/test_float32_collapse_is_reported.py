"""A column float32 cannot hold has to be called out, not silently flattened.

Everything after the CSV reader is float32, which carries about seven
significant digits. A column whose values sit far from zero and vary only
slightly loses that variation on the way in: Unix timestamps in seconds are 128
apart in float32 near 1.6e9, so a column spread over half a minute arrives as
one or two distinct numbers. Normalization then works perfectly on what is
left, the reported standard deviation is 1.0, and the feature the model sees is
a constant.

The numbers are not wrong -- they describe the values the file really produced.
The loss is upstream and cannot be undone here, so this is a warning naming the
column, and it has to fire for exactly the columns that are hurt.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from loguru import logger

from kdp import FeatureType, PreprocessingModel
from kdp.stats import DatasetStatistics

ROWS = 3000


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(7)
    return pd.DataFrame(
        {
            # Unix seconds, spread over well under a minute.
            "epoch_seconds": generator.normal(1.6e9, 30.0, ROWS),
            # Days since the epoch: far from zero, but coarse enough to survive.
            "epoch_days": generator.normal(19_000.0, 300.0, ROWS),
            # Money in cents, up to ten million.
            "price_cents": generator.integers(0, 10**7, ROWS).astype(float),
            "ordinary": generator.normal(0.0, 1.0, ROWS),
            "constant": np.full(ROWS, 42.0),
        },
    )


class _Captured:
    """Collect loguru warnings raised while the block runs."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._handler: int | None = None

    def __enter__(self) -> "_Captured":
        self._handler = logger.add(
            lambda message: self.messages.append(str(message)),
            level="WARNING",
        )
        return self

    def __exit__(self, *_) -> None:
        if self._handler is not None:
            logger.remove(self._handler)


@pytest.mark.unit
class TestFloat32CollapseIsReported(unittest.TestCase):
    """The warning has to name the column that is lost, and only that one."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.frame = _frame()
        self.data = self.directory / "data.csv"
        self.frame.to_csv(self.data, index=False)

    def _run_statistics(self) -> _Captured:
        with _Captured() as captured:
            DatasetStatistics(
                path_data=str(self.data),
                features_stats_path=str(self.directory / "stats.json"),
                features_specs={
                    name: FeatureType.FLOAT_NORMALIZED for name in self.frame
                },
                overwrite_stats=True,
            ).main()
        return captured

    def test_the_column_float32_destroys_is_named(self) -> None:
        warnings = [m for m in self._run_statistics().messages if "float32" in m]
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("epoch_seconds", warnings[0])

    def test_columns_float32_can_hold_are_left_alone(self) -> None:
        warnings = [m for m in self._run_statistics().messages if "float32" in m]
        joined = " ".join(warnings)
        for name in ("epoch_days", "price_cents", "ordinary", "constant"):
            self.assertNotIn(name, joined)


@pytest.mark.unit
class TestTheWarningDescribesWhatActuallyHappens(unittest.TestCase):
    """The claim in the message has to hold end to end."""

    def test_the_named_column_really_does_collapse(self) -> None:
        directory = Path(tempfile.mkdtemp())
        frame = _frame()
        data = directory / "data.csv"
        frame.to_csv(data, index=False)

        preprocessor = PreprocessingModel(
            path_data=str(data),
            features_stats_path=str(directory / "stats.json"),
            features_specs={name: FeatureType.FLOAT_NORMALIZED for name in frame},
            overwrite_stats=True,
            output_mode="dict",
        )
        preprocessor.build_preprocessor()
        outputs = preprocessor.model(
            {
                name: tf.constant(frame[name].values.reshape(-1, 1), dtype=tf.float32)
                for name in frame
            },
        )

        collapsed = np.unique(np.asarray(outputs["epoch_seconds"]))
        self.assertLess(
            len(collapsed),
            10,
            "the warning claims this column collapses, and it did not",
        )
        for name in ("epoch_days", "price_cents", "ordinary"):
            kept = np.unique(np.asarray(outputs[name]))
            self.assertGreater(
                len(kept),
                ROWS // 2,
                f"{name} was not warned about but lost most of its values",
            )


if __name__ == "__main__":
    unittest.main()
