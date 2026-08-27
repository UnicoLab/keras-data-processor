from collections.abc import Callable

import tensorflow as tf
from loguru import logger

from kdp.layers_factory import PreprocessorLayerFactory
from kdp.dynamic_pipeline import DynamicPreprocessingPipeline


class ProcessingStep:
    def __init__(
        self,
        layer_creator: Callable[..., tf.keras.layers.Layer],
        **layer_kwargs,
    ) -> None:
        """Initialize a processing step.

        Args:
            layer_creator (Callable[..., tf.keras.layers.Layer]): A callable that creates a layer.
            **layer_kwargs: Additional keyword arguments for the layer creator.
        """
        self.layer = layer_creator(**layer_kwargs)

    def process(self, input_data) -> tf.keras.layers.Layer:
        """Apply the processing step to the input data.

        Args:
            input_data: The input data to process.
        """
        return self.layer(input_data)

    def connect(self, input_layer) -> tf.keras.layers.Layer:
        """Connect this step's layer to an input layer and return the output layer.

        Args:
            input_layer: The input layer to connect to.
        """
        return self.layer(input_layer)

    @property
    def name(self) -> str:
        """Return the name of the layer."""
        return self.layer.name


class Pipeline:
    def __init__(self, steps: list[ProcessingStep] = None, name: str = "") -> None:
        """Initialize a pipeline with a list of processing steps.

        Args:
            steps (list[ProcessingStep]): A list of processing steps.
            name (str): The name of the pipeline.
        """
        logger.info(f"🔂 Initializing New Pipeline for: {name}")
        self.steps = steps or []

    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline.

        Args:
            step (ProcessingStep): The processing step to add.
        """
        logger.info(f"Adding new preprocessing layer: {step.name} to the pipeline ➕")
        self.steps.append(step)

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the pipeline steps by connecting each step in sequence, starting from the input layer.

        Args:
            input_layer: The input layer to start the chain from.
        """
        output_layer = input_layer
        for step in self.steps:
            output_layer = step.connect(output_layer)
        return output_layer

    def transform(self, input_data: tf.Tensor) -> tf.Tensor:
        """Apply the pipeline to the input data.

        Args:
            input_data: The input data to process.

        Returns:
            tf.Tensor: The processed data.
        """
        output_data = input_data
        for step in self.steps:
            output_data = step.process(output_data)
        return output_data


class FeaturePreprocessor:
    def __init__(self, name: str, use_dynamic: bool = False) -> None:
        """Initializes a feature preprocessor.

        Args:
            name (str): The name of the feature preprocessor.
            use_dynamic (bool): Whether to use the dynamic preprocessing pipeline.
        """
        self.name = name
        self.use_dynamic = use_dynamic
        if not self.use_dynamic:
            self.pipeline = Pipeline(name=name)
        else:
            self.layers = []  # for dynamic pipeline

        # For backwards compatibility with tests
        self.processing_steps = []

    def add_processing_step(
        self,
        layer_creator: Callable[..., tf.keras.layers.Layer] = None,
        **layer_kwargs,
    ) -> None:
        """Add a preprocessing layer to the feature preprocessor pipeline.
        If using the standard pipeline, a ProcessingStep is added.
        Otherwise, the layer is added to a list for dynamic handling.

        Args:
            layer_creator (Callable[..., tf.keras.layers.Layer]): A callable that creates a layer.
                If not provided, the default layer creator is used.
            **layer_kwargs: Additional keyword arguments for the layer creator.
        """
        layer_creator = layer_creator or PreprocessorLayerFactory.create_layer

        # For backwards compatibility with tests
        self.processing_steps.append(layer_kwargs)

        if self.use_dynamic:
            layer = layer_creator(**layer_kwargs)
            logger.info(f"Adding {layer.name} to dynamic preprocessing pipeline")
            self.layers.append(layer)
        else:
            step = ProcessingStep(layer_creator=layer_creator, **layer_kwargs)
            self.pipeline.add_step(step=step)

    def _run_dynamic(self, input_data):
        """Run the dynamic pipeline over a single tensor and return its output.

        The dynamic pipeline keys its data by layer name, so the input is handed
        to the first layer under that layer's name and the last layer's output is
        returned.

        Args:
            input_data: The tensor (or symbolic Keras tensor) to process.

        Returns:
            The output of the final layer, or ``input_data`` if there are no layers.
        """
        if not self.layers:
            return input_data

        dynamic_pipeline = DynamicPreprocessingPipeline(self.layers)
        output_dict = dynamic_pipeline.transform({self.layers[0].name: input_data})
        return output_dict[self.layers[-1].name]

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chains the processing steps starting from the given input_layer.

        For a static pipeline, this delegates to the internal Pipeline's chain() method.
        For the dynamic pipeline, it constructs the dynamic pipeline on the fly.
        """
        if not self.use_dynamic:
            return self.pipeline.chain(input_layer)
        return self._run_dynamic(input_layer)

    def transform(self, input_data: tf.Tensor) -> tf.Tensor:
        """Process the input data through the pipeline.
        For the dynamic pipeline, wrap input in a dictionary and extract final output.

        Args:
            input_data: The input data to process.

        Returns:
            tf.Tensor: The processed data.
        """
        if not self.use_dynamic:
            return self.pipeline.transform(input_data)
        return self._run_dynamic(input_data)
