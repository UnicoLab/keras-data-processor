import math
import tensorflow as tf

from kdp.layers.date_encoding_layer import DateEncodingLayer


class TestDateEncodingLayer:
    """Test suite for DateEncodingLayer."""

    def test_cyclic_encoding(self):
        """Test cyclic encoding of date components."""
        layer = DateEncodingLayer()

        # Create sample parsed dates [year, month, day, day_of_week]
        dates = tf.constant(
            [
                [2023, 1, 15, 6],  # Sunday
                [2023, 6, 30, 4],  # Friday
                [2023, 12, 30, 5],  # Saturday
            ],
            dtype=tf.int32,
        )

        result = layer(dates)
        assert (
            result.shape
            == (
                3,
                8,
            )
        )  # [batch_size, (year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, weekday_sin, weekday_cos)]

        # Check that all values are between -1 and 1 (sine/cosine range)
        assert tf.reduce_all(tf.less_equal(tf.abs(result), 1.0))

    def test_year_normalization(self):
        """Test year normalization."""
        layer = DateEncodingLayer()

        # Test different years
        dates = tf.constant(
            [
                [2023, 1, 1, 6],
                [2024, 1, 1, 6],
                [2025, 1, 1, 6],
            ],
            dtype=tf.int32,
        )

        result = layer(dates)
        # Year encoding should be cyclic, so 2023, 2024, 2025 should have similar patterns
        assert tf.reduce_all(tf.abs(result[0, :2] - result[1, :2]) < 0.01)

    def test_cyclic_continuity(self):
        """Test that cyclic encoding is continuous at boundaries."""
        layer = DateEncodingLayer()

        # Test month transition (December to January)
        dates = tf.constant(
            [
                [2023, 12, 31, 6],
                [2024, 1, 1, 0],
            ],
            dtype=tf.int32,
        )

        result = layer(dates)
        month_encoding_dec = result[0, 2:4]  # month sine and cosine for December
        month_encoding_jan = result[1, 2:4]  # month sine and cosine for January

        # calculate the angle between the two vectors
        dot_product = (
            month_encoding_dec[0] * month_encoding_jan[0]
            + month_encoding_dec[1] * month_encoding_jan[1]
        )
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)

        # The encodings should be similar for consecutive months
        assert abs(angle_deg) <= 52  # ensure that the angle is less than 52 degrees

    def test_weekday_encoding(self):
        """Test that weekday encoding is correct and cyclic."""
        layer = DateEncodingLayer()

        # Test all days of the week
        dates = tf.constant(
            [
                [2023, 1, 1, 0],  # Sunday
                [2023, 1, 2, 1],  # Monday
                [2023, 1, 3, 2],  # Tuesday
                [2023, 1, 4, 3],  # Wednesday
                [2023, 1, 5, 4],  # Thursday
                [2023, 1, 6, 5],  # Friday
                [2023, 1, 7, 6],  # Saturday
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check that Sunday and Saturday encodings are similar (cyclic)
        sunday_encoding = result[0, 6:8]  # weekday sine and cosine for Sunday
        saturday_encoding = result[6, 6:8]  # weekday sine and cosine for Saturday

        dot_product = (
            sunday_encoding[0] * saturday_encoding[0]
            + sunday_encoding[1] * saturday_encoding[1]
        )
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        print("ANGLE DEGREES:", angle_deg)

        assert abs(angle_deg) <= 60  # ensure that the angle is less than 60 degrees

    def test_a_plain_date_still_gives_eight_columns(self):
        """The four date components encode to eight cyclic ones, as before."""
        layer = DateEncodingLayer()
        dates = tf.constant([[2023, 1, 15, 0], [2023, 7, 15, 6]], dtype=tf.int32)
        output = layer(dates)
        assert tuple(output.shape) == (2, 8)
        assert tuple(layer.compute_output_shape((2, 4))) == (2, 8)

    def test_columns_appended_by_an_earlier_layer_are_carried_through(self):
        """`SeasonLayer` runs before this one and its answer has to survive.

        The encoding returned exactly eight columns and dropped everything
        else, so the pipeline ran the season layer afterwards instead -- where
        it read column 1 of this output, the cosine of the year, as the month.
        """
        layer = DateEncodingLayer()
        with_extra = tf.constant(
            [[2023, 1, 15, 0, 1, 0, 0, 0], [2023, 7, 15, 6, 0, 0, 1, 0]],
            dtype=tf.int32,
        )
        output = layer(with_extra)
        assert tuple(output.shape) == (2, 12)
        assert tuple(layer.compute_output_shape((2, 8))) == (2, 12)

        # The carried columns arrive unchanged, after the eight encoded ones.
        carried = output.numpy()[:, 8:]
        assert carried.tolist() == [[1, 0, 0, 0], [0, 0, 1, 0]]

        # And the encoding itself is unaffected by their presence.
        plain = DateEncodingLayer()(
            tf.constant([[2023, 1, 15, 0], [2023, 7, 15, 6]], dtype=tf.int32),
        )
        assert output.numpy()[:, :8].tolist() == plain.numpy().tolist()
