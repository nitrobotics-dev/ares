import typing as t
import uuid
from datetime import datetime

import numpy as np
from pydantic import BaseModel, model_validator
from sqlmodel import Field


class BaseConfig(BaseModel):
    def flatten_fields(self, prefix: str = "") -> t.Dict[str, t.Any]:
        flattened = {}
        for field_name, field_value in self.model_dump().items():
            if isinstance(field_value, dict):
                flattened.update(
                    {f"{prefix}{field_name}_{k}": v for k, v in field_value.items()}
                )
            else:
                flattened[f"{prefix}{field_name}"] = field_value
        return flattened


class Robot(BaseConfig):
    # name: str --> "generic"
    # sensor: str --> just single camera for now
    embodiment: str


class Environment(BaseConfig):
    name: str
    lighting: str
    simulation: bool


class Task(BaseConfig):
    dataset_name: str
    language_instruction: str
    success_criteria: str
    success: float

    @model_validator(mode="after")
    def check_success(self) -> "Task":
        if not 0 <= self.success <= 1:
            raise ValueError("Success must be between 0 and 1, inclusive")
        return self


class Trajectory(BaseConfig):
    actions: list[list[float]]  # what to do? (N, D)
    is_first: int | None  # index of first step
    is_last: int | None  # index of last step
    is_terminal: int | None  # index of terminal step
    states: list[list[float]] | None  # (N, D)


class Rollout(BaseConfig):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    creation_time: datetime
    ingestion_time: datetime
    path: str
    dataset_name: str
    length: int
    robot: Robot
    environment: Environment
    task: Task
    trajectory: Trajectory


def pydantic_to_field_instructions(
    model_cls: type[BaseModel], exclude_fields: t.Dict = {}, prefix: str = ""
) -> list[str]:
    field_instructions = []
    for field_name, field in model_cls.model_fields.items():
        # Get the nested exclude_fields dict if it exists
        nested_exclude = (
            exclude_fields.get(field_name, {})
            if isinstance(exclude_fields, dict)
            else {}
        )

        # Skip if this field is excluded (has a non-dict value in exclude_fields)
        if field_name in exclude_fields and not isinstance(
            exclude_fields[field_name], dict
        ):
            continue

        # Handle nested models recursively
        if hasattr(field.annotation, "model_fields"):
            nested_instructions = pydantic_to_field_instructions(
                field.annotation, nested_exclude, prefix=f"{prefix}{field_name}."
            )
            field_instructions.extend(nested_instructions)
        else:
            field_instructions.append(f"    - {prefix}{field_name}: {str(field)}")
    return field_instructions


def pydantic_to_example_dict(
    model_cls: type[BaseModel], exclude_fields: t.Dict = {}
) -> dict:
    example_dict = {}
    for field_name, field in model_cls.model_fields.items():
        # Get the nested exclude_fields dict if it exists
        nested_exclude = (
            exclude_fields.get(field_name, {})
            if isinstance(exclude_fields, dict)
            else {}
        )

        # Skip if this field is excluded (has a non-dict value in exclude_fields)
        if field_name in exclude_fields and not isinstance(
            exclude_fields[field_name], dict
        ):
            continue

        # Handle nested models recursively
        if hasattr(field.annotation, "model_fields"):
            nested_dict = pydantic_to_example_dict(field.annotation, nested_exclude)
            if nested_dict:  # Only add if not empty
                example_dict[field_name] = nested_dict
        else:
            if hasattr(field.annotation, "__args__"):  # For Literal types
                example_dict[field_name] = field.annotation.__args__[0]
            else:
                example_dict[field_name] = "..."
    return example_dict
