import typing as t
import uuid

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session
from sqlmodel import Field, Session, SQLModel, create_engine

from ares.configs.base import Rollout

SQLITE_PREFIX = "sqlite:///"
BASE_ROBOT_DB_PATH = SQLITE_PREFIX + "robot_data.db"
TEST_ROBOT_DB_PATH = SQLITE_PREFIX + "test_robot_data.db"


# dynamically build flattened SQLModel class
def create_flattened_model(
    data_model: t.Type[BaseModel], non_nullable_fields: list[str] = ["id"]
) -> t.Type[SQLModel]:
    fields: t.Dict[str, t.Any] = {
        "__annotations__": {},
        "__tablename__": "rollout",
    }

    # Add id field explicitly as primary key
    fields["__annotations__"]["id"] = uuid.UUID
    fields["id"] = Field(default_factory=uuid.uuid4, primary_key=True)

    # recursively extract fields
    def flatten_fields(prefix: str, model: t.Type[BaseModel]) -> None:
        for field_name, field in model.model_fields.items():
            if field_name == "id":  # Skip id field as we've handled it above
                continue

            field_type = field.annotation
            if field_type is None:
                continue

            # Handle list types by converting them to JSON strings
            origin_type = t.get_origin(field_type)
            if origin_type is not None and origin_type in (list, t.List):
                fields["__annotations__"][f"{prefix}{field_name}"] = str
                if f"{prefix}{field_name}" not in non_nullable_fields:
                    fields[f"{prefix}{field_name}"] = Field(default=None, nullable=True)
                else:
                    fields[f"{prefix}{field_name}"] = Field()
                continue
            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                # Handle nested BaseModel
                flatten_fields(f"{prefix}{field_name}_", field_type)
                continue

            # Handle the field
            field_key = f"{prefix}{field_name}"
            fields["__annotations__"][field_key] = field_type
            if field_key not in non_nullable_fields:
                fields[field_key] = Field(nullable=True)
            else:
                fields[field_key] = Field()

    flatten_fields("", data_model)
    return type("RolloutSQLModel", (SQLModel,), fields, table=True)


# creates the flattened SQLModel class dynamically from the Rollout config
# note that all fields are nullable by default, except for id and path
# RolloutSQLModel = create_flattened_model(Rollout, non_nullable_fields=["id", "path"])


def setup_database(path: str = BASE_ROBOT_DB_PATH) -> Engine:
    engine = create_engine(path)
    SQLModel.metadata.create_all(engine)
    return engine


def add_rollout(engine: Engine, rollout: Rollout, RolloutSQLModel: SQLModel) -> None:
    rollout_sql_model = RolloutSQLModel(**rollout.flatten_fields(""))
    with Session(engine) as session:
        session.add(rollout_sql_model)
        session.commit()


def add_rollouts(engine: Engine, rollouts: t.List[Rollout]) -> None:
    # use add_all; potentially update to bulk_save_objects
    with Session(engine) as session:
        session.add_all([RolloutSQLModel(**t.flatten_fields("")) for t in rollouts])
        session.commit()


# query helpers
# Database queries
def get_rollouts(engine: Engine) -> pd.DataFrame:
    """Get all rollouts from the database as a pandas DataFrame."""
    with Session(engine) as session:
        query = text(
            """
            SELECT *
            FROM rollout
            ORDER BY id
        """
        )
        df = pd.read_sql(query, session.connection())
    return df


if __name__ == "__main__":
    from ares.configs.test_configs import ROLL1, ROLL2

    RolloutSQLModel = create_flattened_model(
        Rollout, non_nullable_fields=["id", "path"]
    )
    engine = setup_database(path=TEST_ROBOT_DB_PATH)
    add_rollout(engine, ROLL1)
    add_rollout(engine, ROLL2)

    sess = Session(engine)
    res = sess.query(RolloutSQLModel).filter(RolloutSQLModel.task_success > 0.5)
    breakpoint()
