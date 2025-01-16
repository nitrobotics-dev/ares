import typing as t
import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy import Column, Engine, MetaData, inspect, select, text
from sqlmodel import Session, SQLModel, create_engine

from ares.configs.base import Rollout
from ares.configs.pydantic_sql_helpers import create_flattened_model, recreate_model

SQLITE_PREFIX = "sqlite:///"
BASE_ROBOT_DB_PATH = SQLITE_PREFIX + "robot_data.db"
TEST_ROBOT_DB_PATH = SQLITE_PREFIX + "test_robot_data.db"

RolloutSQLModel = create_flattened_model(Rollout)


def setup_database(RolloutSQLModel: SQLModel, path: str = BASE_ROBOT_DB_PATH) -> Engine:
    engine = create_engine(path)
    inspector = inspect(engine)

    if not inspector.has_table("rollout"):
        # If table doesn't exist, create it with all columns
        RolloutSQLModel.metadata.create_all(engine)
    else:
        # Get existing columns
        existing_columns = {col["name"] for col in inspector.get_columns("rollout")}

        # Get new columns from the model
        model_columns = set(RolloutSQLModel.model_fields.keys())

        # Find columns to add
        columns_to_add = model_columns - existing_columns

        if columns_to_add:
            # Create MetaData instance
            metadata = MetaData()
            metadata.reflect(bind=engine)

            # Get the table
            table = metadata.tables["rollout"]

            # Add new columns
            with engine.begin() as conn:
                for col_name in columns_to_add:
                    # Get column definition from model
                    col_type = RolloutSQLModel.model_fields[col_name].annotation
                    conn.execute(
                        text(
                            f"ALTER TABLE rollout ADD COLUMN {col_name} {get_sql_type(col_type)} NULL"
                        )
                    )
                print(f"Added new columns: {columns_to_add}")

    return engine


def get_sql_type(python_type: type) -> str:
    """Convert Python/Pydantic types to SQLite types."""
    type_map = {
        str: "TEXT",
        int: "INTEGER",
        float: "REAL",
        bool: "BOOLEAN",
        datetime: "TIMESTAMP",
        uuid.UUID: "TEXT",
        # Add more type mappings as needed
    }
    # Handle Optional types
    origin = t.get_origin(python_type)
    if origin is t.Union:
        args = t.get_args(python_type)
        # Get the non-None type
        python_type = next(arg for arg in args if arg is not type(None))

    return type_map.get(python_type, "TEXT")


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
def get_rollouts_as_df(engine: Engine) -> pd.DataFrame:
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

        # Get expected columns from current model
        expected_columns = set(RolloutSQLModel.model_fields.keys())

        # Add missing columns with NaN values
        for col in expected_columns - set(df.columns):
            df[col] = pd.NA

        return df


def get_rollout_by_name(
    engine: Engine, dataset_formalname: str, filename: str
) -> t.Optional[Rollout]:
    with Session(engine) as session:
        query = select(RolloutSQLModel).where(
            RolloutSQLModel.dataset_formalname == dataset_formalname,
            RolloutSQLModel.filename == filename,
        )
        row = session.exec(query).first()
        if row is None:
            return None
        return recreate_model(row[0], Rollout)


def get_dataset_rollouts(engine: Engine, dataset_formalname: str) -> list[Rollout]:
    with Session(engine) as session:
        query = select(RolloutSQLModel).where(
            RolloutSQLModel.dataset_formalname == dataset_formalname,
        )
        rows = session.exec(query).all()
        rollouts = []
        for row in rows:
            try:
                rollouts.append(recreate_model(row[0], Rollout))
            except Exception as e:
                print(f"Error recreating model: {e}")
        return rollouts


def get_all_rollouts(engine: Engine) -> list[Rollout]:
    with Session(engine) as session:
        query = select(RolloutSQLModel)
        rows = session.exec(query).all()
        return [recreate_model(row[0], Rollout) for row in rows]


def db_to_df(engine: Engine) -> pd.DataFrame:
    query = select(RolloutSQLModel)
    df = pd.read_sql(query, engine)
    return df


def get_partial_df(engine: Engine, columns: list[str]) -> pd.DataFrame:
    # Convert column names to actual SQLModel column references
    model_columns = [getattr(RolloutSQLModel, col) for col in columns]
    query = select(RolloutSQLModel).with_only_columns(*model_columns)
    df = pd.read_sql(query, engine)
    return df


def setup_rollouts(
    engine: Engine,
    dataset_formalname: str,
    filenames: list[str] | None = None,
) -> list[Rollout]:
    # either get filenames from db or filenames for specific ones
    if filenames is None:
        rollouts = get_dataset_rollouts(engine, dataset_formalname)
    else:
        rollout_attempts = [
            get_rollout_by_name(engine, dataset_formalname, fname)
            for fname in filenames
        ]
        rollouts = [r for r in rollout_attempts if r is not None]
    return rollouts


def add_column_with_vals_and_defaults(
    engine: Engine,
    new_column_name: str,
    python_type: type,
    default_value: any = None,
    key_mapping_col_names: list[str] = None,
    specific_key_mapping_values: dict[tuple, any] = None,
) -> None:
    """
    Add a new column to the rollout table with optional default and specific values.
    If column already exists, just updates the values.
    """
    with engine.begin() as conn:
        # Check if column exists
        inspector = inspect(engine)
        existing_columns = {col["name"] for col in inspector.get_columns("rollout")}

        # Only add column if it doesn't exist
        if new_column_name not in existing_columns:
            sql_type = get_sql_type(python_type)
            conn.execute(
                text(f"ALTER TABLE rollout ADD COLUMN {new_column_name} {sql_type}")
            )

        # Set default value if provided
        if default_value is not None:
            conn.execute(
                text(f"UPDATE rollout SET {new_column_name} = :value"),
                {"value": default_value},
            )

        # Set specific values based on key_mapping_col_names
        if specific_key_mapping_values and key_mapping_col_names:
            for key_tuple, value in specific_key_mapping_values.items():
                # Build WHERE clause dynamically based on key_mapping_col_names
                where_conditions = " AND ".join(
                    f"{col_name} = :{col_name}" for col_name in key_mapping_col_names
                )

                # Create params dict by zipping column names with key tuple values
                params = dict(zip(key_mapping_col_names, key_tuple))
                params["value"] = value

                conn.execute(
                    text(
                        f"UPDATE rollout SET {new_column_name} = :value "
                        f"WHERE {where_conditions}"
                    ),
                    params,
                )


def get_rollouts_by_ids(engine: Engine, ids: list[uuid.UUID]) -> list[Rollout]:
    with Session(engine) as session:
        query = select(RolloutSQLModel).where(RolloutSQLModel.id.in_(ids))
        rows = session.exec(query).all()
        return [recreate_model(row[0], Rollout) for row in rows]


if __name__ == "__main__":
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    df = db_to_df(engine)
    breakpoint()
    # add_rollout(engine, ROLL1, RolloutSQLModel)
    # add_rollout(engine, ROLL2, RolloutSQLModel)

    # sess = Session(engine)
    # res = sess.query(RolloutSQLModel).filter(RolloutSQLModel.task_success > 0.5)
    # breakpoint()
