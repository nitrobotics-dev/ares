from pydantic import BaseModel, model_validator
import tensorflow as tf
import numpy as np


class TensorConverterMixin(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # mixin to convert tf.EagerTensor to Python types
    @model_validator(mode="after")
    def convert_tensors_to_python(self):
        for field_name, value in self.model_dump().items():
            if field_name in self.model_fields:
                if isinstance(value, tf.Tensor):
                    # Convert to numpy first
                    value = value.numpy()
                    # Convert to base Python type if it's a scalar
                    if np.isscalar(value):
                        if isinstance(value, (np.bool_)):
                            value = bool(value)
                        elif isinstance(value, np.floating):
                            value = float(value)
                        elif isinstance(value, np.integer):
                            value = int(value)
                    setattr(self, field_name, value)
        return self


class EpisodeMetadata(TensorConverterMixin, BaseModel):
    file_path: str | tf.Tensor


class OpenXEmbodimentStepObservation(TensorConverterMixin, BaseModel):
    image: np.ndarray | tf.Tensor
    state: np.ndarray | tf.Tensor = None
    depth: np.ndarray | tf.Tensor | None = None
    highres_image: np.ndarray | tf.Tensor | None = None


class OpenXEmbodimentStep(TensorConverterMixin, BaseModel):
    action: np.ndarray | tf.Tensor
    discount: float | tf.Tensor
    is_first: bool | tf.Tensor
    is_last: bool | tf.Tensor
    is_terminal: bool | tf.Tensor
    language_embedding: np.ndarray | tf.Tensor
    language_instruction: str | tf.Tensor
    observation: OpenXEmbodimentStepObservation
    reward: float | tf.Tensor


class OpenXEmbodimentEpisode(TensorConverterMixin, BaseModel):
    episode_metadata: EpisodeMetadata
    steps: list[OpenXEmbodimentStep]
