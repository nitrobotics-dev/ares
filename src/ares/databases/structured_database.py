import typing as t

import pandas as pd
from sqlalchemy import Engine, text
from sqlmodel import Session, SQLModel, create_engine

from ares.configs.base import Rollout
from ares.configs.pydantic_sql_helpers import create_flattened_model

SQLITE_PREFIX = "sqlite:///"
BASE_ROBOT_DB_PATH = SQLITE_PREFIX + "robot_data.db"
TEST_ROBOT_DB_PATH = SQLITE_PREFIX + "test_robot_data.db"

RolloutSQLModel = create_flattened_model(Rollout)


def setup_database(RolloutSQLModel: SQLModel, path: str = BASE_ROBOT_DB_PATH) -> Engine:
    engine = create_engine(path)
    RolloutSQLModel.metadata.create_all(engine)
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

    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    add_rollout(engine, ROLL1, RolloutSQLModel)
    add_rollout(engine, ROLL2, RolloutSQLModel)

    sess = Session(engine)
    res = sess.query(RolloutSQLModel).filter(RolloutSQLModel.task_success > 0.5)
    breakpoint()
