import typing as t
import uuid
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import Engine, String, Float, Boolean, DateTime, Integer
from sqlalchemy.orm import Session

from ares.configs import Robot, Environment, Task, Trajectory


BASE_ROBOT_DB_PATH = "sqlite:///robot_data.db"
TEST_ROBOT_DB_PATH = "sqlite:///test_robot_data.db"


# dynamically build flattened SQLModel class
def create_flattened_model(data_model: t.Type[BaseModel]) -> t.Type[SQLModel]:
    fields: t.Dict[str, t.Any] = {
        "__annotations__": {},
        "__tablename__": "trajectory",
    }

    # Add id field explicitly as primary key
    fields["__annotations__"]["id"] = uuid.UUID
    fields["id"] = Field(default_factory=uuid.uuid4, primary_key=True)

    # recursively extract fields
    def flatten_fields(prefix: str, model: t.Type[BaseModel]):
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


def setup_database(path: str = BASE_ROBOT_DB_PATH) -> Engine:
    engine = create_engine(path)
    SQLModel.metadata.create_all(engine)
    return engine


def add_trajectory(engine: Engine, trajectory: Trajectory):
    trajectory_sql_model = TrajectorySQLModel(**trajectory.flatten_fields(""))
    with Session(engine) as session:
        session.add(trajectory_sql_model)
        session.commit()


if __name__ == "__main__":
    from test_configs import TRAJ1, TRAJ2

    engine = setup_database(path=TEST_ROBOT_DB_PATH)
    add_trajectory(engine, TRAJ1)
    add_trajectory(engine, TRAJ2)

    sess = Session(engine)
    res = sess.query(TrajectorySQLModel).filter(TrajectorySQLModel.task_success > 0.5)
    breakpoint()
