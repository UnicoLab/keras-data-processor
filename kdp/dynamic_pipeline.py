import tensorflow as tf


class DynamicPreprocessingPipeline:
    """Chains Keras preprocessing layers over a dictionary of features.

    Every layer is addressed by its ``name``. When the pipeline runs, a layer
    reads its input from the entry that shares its name if that entry exists,
    and otherwise from the output of the layer that precedes it. The result is
    always written back under the layer's own name, so intermediate results
    stay available to later layers and to the caller.

    That single rule covers both common shapes:

    * supplying one entry per layer runs the layers independently over their
      own inputs;
    * supplying only the first layer's entry chains the layers, each one
      consuming what the previous layer produced.

    Example:
        ```python
        pipeline = DynamicPreprocessingPipeline(
            [ScalingLayer(name="scaling"), LogLayer(name="log")]
        )
        # "log" is absent from the data, so it consumes the scaling output.
        out = pipeline.transform({"scaling": tf.constant([[1.0], [2.0]])})
        out["scaling"], out["log"]
        ```
    """

    def __init__(self, layers: list) -> None:
        """Initializes the pipeline with a list of preprocessing layers.

        Args:
            layers (list): A list of TensorFlow preprocessing layers. Each layer
                must have a unique ``name``, which doubles as the key it reads
                from and writes to.

        Raises:
            ValueError: If two layers share the same name.
        """
        self.layers = layers
        self.dependency_map = self._analyze_dependencies()

    def _analyze_dependencies(self) -> dict:
        """Determines which key each layer consumes.

        Returns:
            dict: A mapping of each layer's name to the name of the entry it
                reads when the pipeline runs. A layer reads its own key when the
                data provides it, otherwise it falls back to the previous
                layer's output. The first layer always reads its own key.

        Raises:
            ValueError: If two layers share the same name.
        """
        dependencies = {}
        previous_name = None
        for layer in self.layers:
            if layer.name in dependencies:
                raise ValueError(
                    f"Duplicate layer name {layer.name!r} in pipeline. Each layer "
                    "needs a unique name because names are used as data keys."
                )
            # The key is only known at run time (it depends on what the data
            # actually contains), so record both candidates in priority order.
            dependencies[layer.name] = (
                (layer.name,) if previous_name is None else (layer.name, previous_name)
            )
            previous_name = layer.name
        return dependencies

    def transform(self, features: dict) -> dict:
        """Applies every layer to the feature dictionary.

        Args:
            features (dict): Mapping of feature names to tensors. It is not
                modified; a new dictionary is returned.

        Returns:
            dict: The input entries plus one entry per layer, holding that
                layer's output.

        Raises:
            KeyError: If a layer has no entry to read from, i.e. neither its own
                key nor the previous layer's output is available.
        """
        current_data = dict(features)
        for layer in self.layers:
            candidates = self.dependency_map[layer.name]
            source = next((key for key in candidates if key in current_data), None)
            if source is None:
                raise KeyError(
                    f"Layer {layer.name!r} has no input: none of {list(candidates)} "
                    f"is present in the data (available keys: "
                    f"{sorted(current_data)})."
                )
            current_data[layer.name] = layer(current_data[source])
        return current_data

    def initialize_and_transform(self, features: dict) -> dict:
        """Applies every layer to the feature dictionary.

        Kept as an alias of :meth:`transform` for backwards compatibility.

        Args:
            features (dict): Mapping of feature names to tensors.

        Returns:
            dict: The input entries plus one entry per layer.
        """
        return self.transform(features)

    def process(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Processes the dataset through the pipeline using the tf.data API.

        Args:
            dataset (tf.data.Dataset): A dataset whose elements are dictionaries
                of features.

        Returns:
            tf.data.Dataset: The processed dataset, with each layer's output
                stored under the layer's name.
        """
        return dataset.map(self.transform)
