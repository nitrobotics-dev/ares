import typing as t
import uuid

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
    name: str
    sensor: str


class Environment(BaseConfig):
    name: str
    lighting: str
    simulation: bool


class Task(BaseConfig):
    name: str
    description: str
    success_criteria: str
    success: float
    language_instruction: str

    @model_validator(mode="after")
    def check_success(self):
        if not 0 <= self.success <= 1:
            raise ValueError("Success must be between 0 and 1, inclusive")
        return self


class Trajectory(BaseConfig):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    path: str
    dataset_name: str
    length: int
    robot: Robot
    environment: Environment
    task: Task
