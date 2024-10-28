import uuid
from pydantic import BaseModel, model_validator
import typing as t
from sqlmodel import SQLModel, Field
from sqlmodel import SQLModel, Field
from pydantic import BaseModel
from typing import Any, Type
import uuid


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
    type: str
    sensor: str


class Environment(BaseConfig):
    name: str
    type: str
    lighting: str
    simulation: bool


class Task(BaseConfig):
    name: str
    description: str
    success_criteria: str
    success: float

    @model_validator(mode="after")
    def check_success(self):
        if not 0 <= self.success <= 1:
            raise ValueError("Success must be between 0 and 1, inclusive")
        return self


class Trajectory(BaseConfig):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    robot: Robot
    environment: Environment
    task: Task


# dynamically build flattened SQLModel class
def create_flattened_model(data_model: Type[BaseModel]) -> Type[SQLModel]:
    fields: t.Dict[str, Any] = {
        "__annotations__": {},
        "__tablename__": "trajectory",
    }

    # Add id field explicitly as primary key
    fields["__annotations__"]["id"] = uuid.UUID
    fields["id"] = Field(default_factory=uuid.uuid4, primary_key=True)

    # recursively extract fields
    def flatten_fields(prefix: str, model: Type[BaseModel]):
        for field_name, field_type in model.__annotations__.items():
            if field_name == "id":  # Skip id field as we've handled it above
                continue
            if issubclass(field_type, BaseModel):
                flatten_fields(f"{prefix}{field_name}_", field_type)
            else:
                field_key = f"{prefix}{field_name}"
                fields["__annotations__"][field_key] = field_type
                fields[field_key] = Field()

    flatten_fields("", data_model)
    return type("TrajectorySQLModel", (SQLModel,), fields, table=True)


TrajectorySQLModel = create_flattened_model(Trajectory)
