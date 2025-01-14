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
    success: bool | None = None


class OpenXEmbodimentStepObservation(TensorConverterMixin, BaseModel):
    image: np.ndarray
    state: np.ndarray | None = None
    depth: np.ndarray | None = None
    highres_image: np.ndarray | None = None
    wrist_image: np.ndarray | None = None
    end_effector_state: np.ndarray | None = None

    @model_validator(mode="after")
    def swap_in_highres_image(self) -> "OpenXEmbodimentStepObservation":
        # use the highres image as image if available
        if self.highres_image is not None:
            self.image = self.highres_image
        return self

    @model_validator(mode="before")
    def concat_state(cls, data: dict) -> dict:
        if "state" not in data:
            extra_state_keys = [
                "end_effector_cartesian_pos",
                "end_effector_cartesian_velocity",
                "joint_pos",
            ]
            state_arrays = [data[k] for k in extra_state_keys if k in data]
            if state_arrays:
                data["state"] = np.concatenate(state_arrays)
            else:
                data["state"] = None
        return data


class OpenXEmbodimentStep(TensorConverterMixin, BaseModel):
    action: np.ndarray
    discount: float | None = None
    is_first: bool
    is_last: bool
    is_terminal: bool
    language_embedding: np.ndarray | None = None
    language_instruction: str | None = None
    observation: OpenXEmbodimentStepObservation
    reward: float

    # TODO: hack to remap fields between datasets
    @model_validator(mode="before")
    @classmethod
    def remap_fields(cls, data: dict) -> dict:
        # Handle observation field remapping
        if "observation" in data and isinstance(data["observation"], dict):
            obs = data["observation"]

            # Move natural_language_instruction if it exists in observation
            if "natural_language_instruction" in obs:
                data["language_instruction"] = obs.pop("natural_language_instruction")
            if "natural_language_embedding" in obs:
                data["language_embedding"] = obs.pop("natural_language_embedding")

            # Add more field remapping here as needed
            action = data["action"]
            if isinstance(action, dict):
                if "gripper_closedness_action" in action:  # jaco_play
                    data["action"] = np.concatenate(
                        [
                            action["world_vector"],
                            action["gripper_closedness_action"],
                            action["terminate_episode"],
                        ]
                    )
        return data


class OpenXEmbodimentEpisode(TensorConverterMixin, BaseModel):
    episode_metadata: OpenXEmbodimentEpisodeMetadata | None = None
    steps: list[OpenXEmbodimentStep]


PATH_TO_SPREADSHEET = "/workspaces/ares/data/oxe.csv"
HEADER_ROW = 16


def get_oxe_dataframe() -> pd.DataFrame:
    return pd.read_csv(PATH_TO_SPREADSHEET, header=HEADER_ROW)


def get_dataset_information(dataset_name: str) -> pd.DataFrame:
    df = get_oxe_dataframe()
    return dict(df[df["Registered Dataset Name"] == dataset_name].iloc[0])
