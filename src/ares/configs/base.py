import json
import typing as t
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator


class BaseConfig(BaseModel):
    def flatten_fields(self, prefix: str = "") -> t.Dict[str, t.Any]:
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


class Robot(BaseConfig):
    embodiment: str
    gripper: str
    morphology: str
    action_space: str
    rgb_cams: int
    depth_cams: int
    wrist_cams: int


class Environment(BaseConfig):
    name: str
    lighting: str = Field(
        description="Lighting conditions in the environment",
        # pattern="^(dim|normal|bright)$",
    )
    # background: str = Field(description="Description of the setting or background")
    simulation: bool = Field(
        description="Whether the input is from a simulation (True) or the real world (False)"
    )
    # object: str = Field(
    #     description="Description of the object the robot is interacting with"
    # )
    # object_shape_color ??


class Task(BaseConfig):
    language_instruction: str
    language_instruction_type: str
    success_criteria: str | None = None
    success: float | None = None

    @model_validator(mode="after")
    def check_success(self) -> "Task":
        if self.success is not None and not 0 <= self.success <= 1:
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

    @model_validator(mode="before")
    def convert_sequences_to_json(cls, data: dict) -> dict:
        # Convert any list fields to JSON strings
        for field in ["actions", "states", "rewards"]:
            if isinstance(data.get(field), (list, np.ndarray)):
                # Convert numpy arrays to lists first if needed
                value = data[field]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                data[field] = json.dumps(value)
        return data

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
    description: str | None = Field(
        description="A detailed description of the input video. Include analysis of the task the robot is completing, including success criteria and performance.",
        default=None,
    )
    length: int
    robot: Robot
    environment: Environment
    task: Task
    trajectory: Trajectory


def pydantic_to_field_instructions(
    model_cls: type[BaseModel],
    exclude_fields: t.Dict = {},
    prefix: str = "",
    required_only: bool = False,
) -> list[str]:
    field_instructions = []
    skip_fields = {"id", "ingestion_time", "creation_time"}

    for field_name, field in model_cls.model_fields.items():
        # Skip auto-generated and optional fields
        if field_name in skip_fields:
            continue

        # Skip optional fields if required_only is True
        if required_only and (not field.is_required or field.default is None):
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

            # TODO: add more metadata! we can support
            # - str fields: min length, max length
            # - numeric fields: gt, ge, lt, le, multiple_of
            # - list fields: min length, max length, unique
            # - enum fields! e.g. LightingConditions
            if field.metadata:
                for meta in field.metadata:
                    if hasattr(meta, "pattern"):
                        field_info += f" (valid values: {meta.pattern})"

            field_instructions.append(f"    - {field_info}")
    return field_instructions


def pydantic_to_example_dict(
    model_cls: type[BaseModel], exclude_fields: t.Dict = {}, required_only: bool = False
) -> dict:
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
