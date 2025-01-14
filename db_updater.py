from pathlib import Path

from ares.configs.base import Rollout
from ares.configs.pydantic_sql_helpers import create_flattened_model, recreate_model
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    add_column_with_vals_and_defaults,
    db_to_df,
    get_all_rollouts,
    setup_database,
)

engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
rollouts = get_all_rollouts(engine)


if __name__ == "__main__":
    id_keys = ["dataset_name", "path"]
    # new_col_key_stem = ["success", "rewards", "reward_step"]
    # new_cols_flat_types = [bool, str, int]
    # default_vals = [None, None, None]
    new_col_key_stem = ["filename"]
    new_cols_flat_names = ["filename"]
    new_cols_flat_types = [str]
    default_vals = [None]

    for i in range(len(new_cols_flat_names)):
        input_mapping = dict()
        for rollout in rollouts:
            input_mapping[tuple(getattr(rollout, k) for k in id_keys)] = getattr(
                rollout,
                new_col_key_stem[i],
            )

        add_column_with_vals_and_defaults(
            engine=engine,
            new_column_name=new_cols_flat_names[i],
            python_type=new_cols_flat_types[i],
            default_value=default_vals[i],
            key_mapping_col_names=id_keys,
            specific_key_mapping_values=input_mapping,
        )
        print(
            f"added {new_cols_flat_names[i]}: e.g. {list(input_mapping.values())[:5]}"
        )
