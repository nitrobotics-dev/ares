from pydantic import BaseModel, model_validator
import tensorflow as tf
import numpy as np


class TensorConverterMixin(BaseModel):
    """
    TFDS returns tensors; we want everything in numpy arrays or
    base python types to work with other parts of the codebase.
    """

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def convert_tensors_to_python(cls, data):
        def convert_value(value):
            if isinstance(value, tf.Tensor):
                # Convert to numpy first
                value = value.numpy()
                # Convert to base Python type if it's a scalar
                if np.isscalar(value):
                    if isinstance(value, (np.bool_)):
                        return bool(value)
                    elif isinstance(value, np.floating):
                        return float(value)
                    elif isinstance(value, np.integer):
                        return int(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            return value

        return {k: convert_value(v) for k, v in data.items()}


class EpisodeMetadata(TensorConverterMixin, BaseModel):
    file_path: str


class OpenXEmbodimentStepObservation(TensorConverterMixin, BaseModel):
    image: np.ndarray
    state: np.ndarray = None
    depth: np.ndarray | None = None
    highres_image: np.ndarray | None = None


class OpenXEmbodimentStep(TensorConverterMixin, BaseModel):
    action: np.ndarray
    discount: float
    is_first: bool
    is_last: bool
    is_terminal: bool
    language_embedding: np.ndarray
    language_instruction: str
    observation: OpenXEmbodimentStepObservation
    reward: float


class OpenXEmbodimentEpisode(TensorConverterMixin, BaseModel):
    episode_metadata: EpisodeMetadata
    steps: list[OpenXEmbodimentStep]
