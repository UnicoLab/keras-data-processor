import numpy as np
import tensorflow as tf
from parameterized import parameterized

from kdp.layers.time_series.moving_average_layer import MovingAverageLayer


class TestMovingAverageLayer(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    @parameterized.expand(
        [
            # (periods, drop_na, pad_value, keep_original)
            ([3], True, 0.0, False),
            ([3, 5], True, 0.0, False),
            ([3], False, 0.0, False),
            ([3], True, -999.0, False),
            ([3], True, 0.0, True),
        ]
    )
    def test_moving_average_layer_config(
        self, periods, drop_na, pad_value, keep_original
    ):
        """Test the configuration options for MovingAverageLayer."""
        # Create the layer
        layer = MovingAverageLayer(
            periods=periods,
            drop_na=drop_na,
            pad_value=pad_value,
            keep_original=keep_original,
        )

        # Check that the configuration is correct
        self.assertEqual(layer.periods, periods)
        self.assertEqual(layer.drop_na, drop_na)
        self.assertEqual(layer.pad_value, pad_value)
        self.assertEqual(layer.keep_original, keep_original)

    def test_single_period_drop_na_true(self):
        """Test MovingAverageLayer with a single period and drop_na=True."""
        # Create an input tensor (constant series)
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with period=3 and drop_na=True
        period = 3
        layer = MovingAverageLayer(periods=[period], drop_na=True, keep_original=False)

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # Expected output: Moving average of period 3
        # For each position i, MA(3) = (input[i] + input[i-1] + input[i-2]) / 3
        # With drop_na=True, the first (period-1) values should be dropped
        expected_ma = np.array(
            [
                (1.0 + 2.0 + 3.0) / 3,  # MA(3) for position 2
                (2.0 + 3.0 + 4.0) / 3,  # MA(3) for position 3
                (3.0 + 4.0 + 5.0) / 3,
                (4.0 + 5.0 + 6.0) / 3,
                (5.0 + 6.0 + 7.0) / 3,
                (6.0 + 7.0 + 8.0) / 3,
                (7.0 + 8.0 + 9.0) / 3,
                (8.0 + 9.0 + 10.0) / 3,  # MA(3) for position 9
            ]
        )

        # Check that the output shape is as expected
        self.assertEqual(output_np.shape, (len(input_data) - (period - 1),))

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_ma, rtol=1e-5)

    def test_multiple_periods(self):
        """Test MovingAverageLayer with multiple periods."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with multiple periods and drop_na=True
        periods = [2, 3]
        layer = MovingAverageLayer(periods=periods, drop_na=True, keep_original=False)

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # Expected output:
        # MA(2) = (input[i] + input[i-1]) / 2
        # MA(3) = (input[i] + input[i-1] + input[i-2]) / 3
        # With drop_na=True, the first (max_period-1) values should be dropped
        expected_ma2 = [
            (2.0 + 3.0) / 2,  # MA(2) for position 2
            (3.0 + 4.0) / 2,  # MA(2) for position 3
            (4.0 + 5.0) / 2,
            (5.0 + 6.0) / 2,
            (6.0 + 7.0) / 2,
            (7.0 + 8.0) / 2,  # MA(2) for position 7
        ]

        expected_ma3 = [
            (1.0 + 2.0 + 3.0) / 3,  # MA(3) for position 2
            (2.0 + 3.0 + 4.0) / 3,  # MA(3) for position 3
            (3.0 + 4.0 + 5.0) / 3,
            (4.0 + 5.0 + 6.0) / 3,
            (5.0 + 6.0 + 7.0) / 3,
            (6.0 + 7.0 + 8.0) / 3,  # MA(3) for position 7
        ]

        expected_output = np.column_stack([expected_ma2, expected_ma3])

        # Check that the output shape is as expected
        self.assertEqual(
            output_np.shape, (len(input_data) - (max(periods) - 1), len(periods))
        )

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_drop_na_false(self):
        """Test MovingAverageLayer with drop_na=False."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with period=3 and drop_na=False
        period = 3
        layer = MovingAverageLayer(
            periods=[period], drop_na=False, pad_value=0.0, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # Expected output:
        # With drop_na=False, positions where there's not enough data for a full window
        # should use partial averages
        expected_output = np.array(
            [
                1.0,  # Position 0: just the value itself
                (1.0 + 2.0) / 2,  # Position 1: average of first two values
                (1.0 + 2.0 + 3.0) / 3,  # Position 2: full window average
                (2.0 + 3.0 + 4.0) / 3,  # Position 3: full window average
                (3.0 + 4.0 + 5.0) / 3,  # Position 4: full window average
            ]
        )

        # Check that the output shape is as expected
        self.assertEqual(output_np.shape, (5,))

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_custom_pad_value(self):
        """Test MovingAverageLayer with a custom pad_value."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with a custom pad_value and drop_na=False
        pad_value = -999.0
        layer = MovingAverageLayer(
            periods=[2], drop_na=False, pad_value=pad_value, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # First element should be the original value, not pad_value
        # In our implementation the first value is just the input value
        self.assertEqual(output_np[0], 1.0)

    def test_keep_original_true(self):
        """Test MovingAverageLayer with keep_original=True."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with keep_original=True
        layer = MovingAverageLayer(periods=[3], drop_na=True, keep_original=True)

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # Expected output:
        # Original value and moving average, with first (period-1) rows dropped due to drop_na=True
        expected_output = np.array(
            [
                [3.0, 2.0],  # Original value and MA(3) of [1,2,3]
                [4.0, 3.0],  # Original value and MA(3) of [2,3,4]
                [5.0, 4.0],  # Original value and MA(3) of [3,4,5]
            ]
        )

        # Check that the output shape is as expected
        self.assertEqual(output_np.shape, (3, 2))

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_2d_input(self):
        """Test MovingAverageLayer with a 2D input tensor.

        Time runs along axis 0 -- the layer sits behind a chronologically sorted
        input in the preprocessing pipeline -- so each column is an independent
        series.
        """
        # 5 time steps, 2 series
        input_data = [
            [1.0, 6.0],
            [2.0, 7.0],
            [3.0, 8.0],
            [4.0, 9.0],
            [5.0, 10.0],
        ]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        layer = MovingAverageLayer(periods=[3], drop_na=True, keep_original=False)
        output_np = layer(input_tensor).numpy()

        # MA(3) per column: [1,2,3],[2,3,4],[3,4,5] and [6,7,8],[7,8,9],[8,9,10]
        expected_output = np.array([[2.0, 7.0], [3.0, 8.0], [4.0, 9.0]])

        self.assertEqual(output_np.shape, (3, 2))
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_output_depends_on_the_input_values(self):
        """Guard against canned outputs: different data must give different results."""
        layer = MovingAverageLayer(periods=[3], drop_na=True, keep_original=False)

        trend = layer(tf.constant([1.0, 3.0, 5.0, 7.0, 9.0])).numpy()
        noise = layer(tf.constant([10.0, 20.0, 45.0, 5.0, 0.0])).numpy()

        self.assertAllClose(trend, np.array([3.0, 5.0, 7.0]), rtol=1e-5)
        self.assertAllClose(noise, np.array([25.0, 70.0 / 3.0, 50.0 / 3.0]), rtol=1e-5)

    def test_shorter_series_than_window_yields_no_rows(self):
        """With drop_na=True and no full window available the output is empty."""
        layer = MovingAverageLayer(periods=[5], drop_na=True, keep_original=False)
        output_np = layer(tf.constant([1.0, 2.0, 3.0])).numpy()
        self.assertEqual(output_np.shape, (0,))

    def test_compute_output_shape_matches_call(self):
        """The declared output shape agrees with what call() produces."""
        for periods, drop_na, keep_original in [
            ([3], True, False),
            ([3], False, False),
            ([2, 3], True, False),
            ([3], True, True),
        ]:
            with self.subTest(
                periods=periods, drop_na=drop_na, keep_original=keep_original
            ):
                layer = MovingAverageLayer(
                    periods=periods, drop_na=drop_na, keep_original=keep_original
                )
                data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                self.assertEqual(
                    tuple(layer(data).shape),
                    tuple(layer.compute_output_shape((8,))),
                )

    def test_config(self):
        """Test that the layer can be serialized and deserialized."""
        # Create a layer with custom configuration
        original_layer = MovingAverageLayer(
            periods=[3, 7, 14],
            drop_na=False,
            pad_value=-1.0,
            keep_original=True,
            name="test_moving_average_layer",
        )

        # Get config
        config = original_layer.get_config()

        # Create a new layer from config
        new_layer = MovingAverageLayer.from_config(config)

        # Check that the config was preserved
        self.assertEqual(new_layer.periods, [3, 7, 14])
        self.assertEqual(new_layer.drop_na, False)
        self.assertEqual(new_layer.pad_value, -1.0)
        self.assertEqual(new_layer.keep_original, True)
        self.assertEqual(new_layer.name, "test_moving_average_layer")
