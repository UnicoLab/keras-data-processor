import re
import string
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="kdp.layers")
class TextPreprocessingLayer(keras.layers.Layer):
    def __init__(self, stop_words: list, **kwargs: dict) -> None:
        """Initializes a TextPreprocessingLayer.

        Args:
            stop_words (list): A list of stop words to remove.
            **kwargs: Additional keyword arguments for the layer.
        """
        super().__init__(**kwargs)
        self.stop_words = stop_words
        # Define punctuation and stop words patterns as part of the configuration
        self.punctuation_pattern = re.escape(string.punctuation)
        self.stop_words_pattern = r"|".join(
            [re.escape(word) for word in self.stop_words],
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Preprocesses the input tensor.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The preprocessed tensor.
        """
        x = tf.strings.lower(x)
        x = tf.strings.regex_replace(x, f"[{self.punctuation_pattern}]", " ")
        stop_words_regex = rf"\b({self.stop_words_pattern})\b\s?"
        x = tf.strings.regex_replace(x, stop_words_regex, " ")
        return tf.strings.regex_replace(x, r"\s+", " ")

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        config = super().get_config()
        # Only what `__init__` takes. The two patterns are derived from
        # `stop_words`, and writing them here made every saved model carry
        # arguments the constructor would not accept.
        config.update({"stop_words": self.stop_words})
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TextPreprocessingLayer":
        """Instantiates a TextPreprocessingLayer from its configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The TextPreprocessingLayer instance.
        """
        # Models saved before this release carry the two derived patterns,
        # which `__init__` never accepted: loading one failed outright with
        # "Unrecognized keyword arguments passed to TextPreprocessingLayer".
        # They are rebuilt from `stop_words`, so they are dropped here and
        # those files load.
        config = dict(config)
        config.pop("punctuation_pattern", None)
        config.pop("stop_words_pattern", None)
        return cls(**config)
