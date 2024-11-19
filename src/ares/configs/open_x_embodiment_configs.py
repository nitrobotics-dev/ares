from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel, model_validator


class TensorConverterMixin(BaseModel):
    """
    TFDS returns tensors; we want everything in numpy arrays or
    base python types to work with other parts of the codebase.
    """

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def convert_tensors_to_python(cls, data: dict) -> dict:
        def convert_value(value: Any) -> Any:
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


class OpenXEmbodimentEpisodeMetadata(TensorConverterMixin, BaseModel):
    file_path: str


class OpenXEmbodimentStepObservation(TensorConverterMixin, BaseModel):
    image: np.ndarray
    state: np.ndarray | None = None
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
    episode_metadata: OpenXEmbodimentEpisodeMetadata
    steps: list[OpenXEmbodimentStep]


PATH_TO_SPREADSHEET = "/workspaces/ares/data/oxe.csv"
HEADER_ROW = 16


def get_oxe_dataframe() -> pd.DataFrame:
    return pd.read_csv(PATH_TO_SPREADSHEET, header=HEADER_ROW)


def get_dataset_information(dataset_name: str) -> pd.DataFrame:
    df = get_oxe_dataframe()
    return dict(df[df["Registered Dataset Name"] == dataset_name].iloc[0])
