"""
Main configuration classes for ARES. This configurations are used throughout the repository in order to standardize the data across many datasets.
We use pydantic to define the configurations and manually construct many subclasses of these classes.

The configuration starts with the toplevel `Rollout` class, which contains all information about a given episode. The Rollout class contains
a Robot, Environment, Task, and Trajectory, which all contain many fields or subconfigs. All configs inherit from the `BaseConfig` class, which
contains utility helpers for flattening fields and getting nested attributes. 

We make judicious use of pydantic's `Field` object to add metadata to the fields, such as the valid values for a field, or the description of the field.
These fields include the type, description, and pattern, which we can use both to validate data *and* as instructions for models to generate data. 
Fields that are not required are given a default value of `None`, and fields with the suffix `estimate` will be inferred by a model. For example, see the 
`COLOR_PATTERN` below, which is used to validate the `color_estimate` field in the `Robot` class and thus force all color estimates to be in the list of valid colors.
Likewise, for many patterns, we add `PATTERN_EXTRA_CHOICES` to allow for some flexibility in the data, such as allowing for "other" or "unknown" values.

The end of the file includes helpers to transform a BaseConfig object into a labelling instructions and an example dictionary. Additionally, see 
`ares.configs.pydantic_sql_helpers` to see how we dynamically create SQLModel classes from pydantic models, which requires a bit of extra work.
"""

import json
import typing as t
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

PATTERN_EXTRA_CHOICES = "|other|unknown"
COLOR_PATTERN = f"^(white|black|gray|red|green|blue|yellow|purple|orange|brown|pink|gray{PATTERN_EXTRA_CHOICES})$"


class BaseConfig(BaseModel):
    def flatten_fields(self, prefix: str = "") -> dict[str, t.Any]:
        flattened = {}
        for field_name, field_value in self.model_dump().items():
            if isinstance(field_value, dict):
                flattened.update(
                    {f"{prefix}{field_name}_{k}": v for k, v in field_value.items()}
                )
            elif isinstance(field_value, list):
                # Convert lists to JSON strings
                flattened[f"{prefix}{field_name}"] = json.dumps(field_value)
            else:
                flattened[f"{prefix}{field_name}"] = field_value
        return flattened

    def get_nested_attr(self, attr_path: str) -> t.Any | None:
        """
        Get nested attribute using flattened notation by leveraging flatten_fields.
        e.g. rollout.get_nested_attr("task_language_instruction") returns rollout.task.language_instruction.
        """
        flattened = self.flatten_fields()
        if attr_path not in flattened:
            return None
        return flattened[attr_path]

    @model_validator(mode="before")
    def convert_sequences_to_json(cls, data: dict) -> dict:
        # Convert any sequence fields to JSON strings
        # this is a workaround for converting to sql formats
        for field, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                # Convert numpy arrays to lists first if needed
                value = data[field]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                data[field] = json.dumps(value)
        return data


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_several_dicts(dicts: list[dict]) -> dict:
    merged = dicts[0].copy()
    for d in dicts[1:]:
        merged = merge_dicts(merged, d)
    return merged


def merge_config_sources(
    object_cls: t.Type[BaseConfig],
    hardcoded_info: dict,
    structured_info: dict,
    extra_kwargs: dict,
) -> BaseConfig:
    """Merge multiple sources of configuration information and create a config object.

    Args:
        object_cls: The BaseConfig class to instantiate
        hardcoded_info: Dictionary containing hardcoded configuration values
        structured_info: Dictionary containing structured configuration values
        extra_kwargs: Additional keyword arguments to override other sources

    Returns:
        Instantiated config object with merged information
    """
    merged_info = merge_several_dicts(
        [
            hardcoded_info.get(object_cls.__name__.lower(), {}),
            structured_info.get(object_cls.__name__.lower(), {}),
            extra_kwargs,
        ]
    )
    return object_cls(**merged_info)


class Robot(BaseConfig):
    embodiment: str
    gripper: str | None = None
    morphology: str
    action_space: str
    rgb_cams: int
    depth_cams: int
    wrist_cams: int
    color_estimate: str = Field(
        description="The main color of the robot. 'other' implies a color that is not in the list of valid colors; 'unknown' implies that the color of the robot is not known.",
        pattern=COLOR_PATTERN,
    )
    camera_angle_estimate: str = Field(
        description="The angle of the camera. 'other' implies an angle that is not in the list of valid angles.",
        pattern=f"^(front|side|top|angled|wrist{PATTERN_EXTRA_CHOICES})$",
    )


class Environment(BaseConfig):
    name: str
    lighting_estimate: str = Field(
        description="Lighting conditions in the environment",
        pattern=f"^(dim|normal|bright{PATTERN_EXTRA_CHOICES})$",
    )
    simulation_estimate: bool = Field(
        description="Whether the frames are from a simulation (True) or the real world (False)"
    )
    data_collection_method: str | None = None
    background_estimate: str = Field(description="The background of the environment")
    surface_estimate: str = Field(
        description="The surface that the task is taking place on",
        pattern=f"^(wood|metal|plastic|glass|concrete|carpet|tile|rubber|fabric|composite|marble|granite|cardboard|foam{PATTERN_EXTRA_CHOICES})$",
    )
    focus_objects_estimate: str = Field(
        description="The object(s) is the robot supposed to interact with.",
        default_factory=list,
    )
    distractor_objects_estimate: str = Field(
        description="Objects present in the scene that the robot is NOT supposed to interact with.",
        default_factory=list,
    )
    people_estimate: bool = Field(
        description="Whether there are people present in the scene or not.",
    )
    static_estimate: bool = Field(
        description="Whether the scene is static (meaning no motion in the background) or dynamic (motion in the background). This ignores the motion of the robot and objects it interacts with.",
    )

    @model_validator(mode="before")
    def validate_surface(cls, values: dict) -> dict:
        for k in ["surface_estimate", "background_estimate", "lighting_estimate"]:
            if k in values:
                values[k] = values[k].lower()
        return values


class Task(BaseConfig):
    language_instruction: str
    language_instruction_type: str
    success_criteria: str | None = None
    success: float | None = None
    success_estimate: float = Field(
        description="An estimate of the success of the task",
        ge=0,
        le=1,
    )
    complexity_category_estimate: str = Field(
        description="The complexity of the task",
        pattern=f"^(simple|medium|complex{PATTERN_EXTRA_CHOICES})$",
    )
    complexity_score_estimate: float = Field(
        description="The complexity score of the task",
        ge=0,
        le=1,
    )
    rarity_estimate: float = Field(
        description="The subjective interestingness of the episode. 0 means it is not interesting at all, 1 means it is very interesting. Episodes may be interesting for different reasons, e.g. the robot is performing a difficult task, the robot is interacting with a rare object, the robot is interacting with a person, a novel edge case, etc.",
        ge=0,
        le=1,
    )

    @model_validator(mode="after")
    def check_success(self) -> "Task":
        if (
            self.success is not None
            and not np.isnan(self.success)
            and not pd.isna(self.success)
            and not 0 <= self.success <= 1
        ):
            raise ValueError("Success must be between 0 and 1, inclusive")
        return self


class Trajectory(BaseConfig):
    actions: str  # JSON string of list[list[float]]
    is_first: int | None  # index of first step
    is_last: int | None  # index of last step
    is_terminal: int | None  # index of terminal step
    states: str | None  # JSON string of list[list[float]]
    rewards: str | None = None  # JSON string of list[float]
    reward_step: int | None = (
        None  # if -1; never reach reward. if > 0, the index of the first step to recieve reward of 1. If None, we do not have reward data.
    )

    @property
    def actions_array(self) -> np.ndarray:
        """Get actions as a numpy array instead of JSON string."""
        return np.array(json.loads(self.actions))

    @property
    def states_array(self) -> np.ndarray | None:
        """Get states as a numpy array instead of JSON string."""
        if self.states is None:
            return None
        return np.array(json.loads(self.states))

    @property
    def rewards_array(self) -> np.ndarray | None:
        """Get rewards as a numpy array instead of JSON string."""
        if self.rewards is None:
            return None
        return np.array(json.loads(self.rewards))


class Rollout(BaseConfig):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    creation_time: datetime | None
    ingestion_time: datetime
    path: str
    filename: str
    dataset_name: str
    dataset_filename: str
    dataset_formalname: str
    description_estimate: str | None = Field(
        description="A detailed description of the entire episode, meaning everything that happens in the images. Include analysis of the task the robot is completing, including success criteria and performance.",
    )
    length: int
    robot: Robot
    environment: Environment
    task: Task
    trajectory: Trajectory
    split: str | None = None

    @property
    def full_path(self) -> str:
        return f"{self.dataset_filename}/{self.filename}"


def pydantic_to_field_instructions(
    model_cls: type[BaseModel],
    exclude_fields: dict | None = None,
    prefix: str = "",
    required_only: bool = False,
) -> list[str]:
    exclude_fields = exclude_fields or {}
    field_instructions = []
    skip_fields = {"id", "ingestion_time", "creation_time"}

    for field_name, field in model_cls.model_fields.items():
        # Skip auto-generated and optional fields
        if field_name in skip_fields:
            continue

        # Skip optional fields if required_only is True
        if required_only and (not field.is_required() or field.default is None):
            continue

        # Check if field exists in exclude_fields (both nested and top-level)
        obj_name = model_cls.__name__.lower()
        if obj_name in exclude_fields and field_name in exclude_fields[obj_name]:
            continue

        # Handle nested models recursively
        if hasattr(field.annotation, "model_fields"):
            nested_exclude = exclude_fields.get(field_name.lower(), {})
            if isinstance(nested_exclude, dict):
                nested_instructions = pydantic_to_field_instructions(
                    field.annotation,
                    exclude_fields,
                    prefix=f"{prefix}{field_name}.",
                    required_only=required_only,
                )
                field_instructions.extend(nested_instructions)
        else:
            # Add field description if available
            field_info = f"{prefix}{field_name}: {str(field.annotation)}"
            if field.description:
                field_info += f" - {field.description}"
            if field.metadata:
                for meta in field.metadata:
                    if hasattr(meta, "pattern"):
                        field_info += f" (valid values: {meta.pattern})"
                    if hasattr(meta, "ge"):
                        field_info += f" (minimum value: {meta.ge})"
                    if hasattr(meta, "le"):
                        field_info += f" (maximum value: {meta.le})"
                    if hasattr(meta, "multiple_of"):
                        field_info += f" (multiple of: {meta.multiple_of})"

            field_instructions.append(f"    - {field_info}")
    return field_instructions


def pydantic_to_example_dict(
    model_cls: type[BaseModel],
    exclude_fields: dict | None = None,
    required_only: bool = False,
) -> dict:
    exclude_fields = exclude_fields or {}
    # Get the field instructions first
    field_instructions = pydantic_to_field_instructions(
        model_cls, exclude_fields, required_only=required_only
    )

    # Convert the instructions into a nested dictionary
    example_dict = {}
    for instruction in field_instructions:
        # Strip the leading "  - " and split into path and type
        path = instruction.strip()[2:].split(":")[0].strip()

        # Handle nested paths
        parts = path.split(".")
        current = example_dict
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = "..."
    return example_dict
