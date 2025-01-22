from pathlib import Path

import numpy as np

from ares.configs.base import Rollout
from ares.configs.open_x_embodiment_configs import get_dataset_information
from ares.configs.pydantic_sql_helpers import create_flattened_model, recreate_model
from ares.constants import DATASET_NAMES
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    add_column_with_vals_and_defaults,
    db_to_df,
    get_all_rollouts,
    get_dataset_rollouts,
    get_rollout_by_name,
    setup_database,
)

engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
rollouts = get_all_rollouts(engine)

DATASET_INFOS = dict()
for data_names in DATASET_NAMES:
    dfilename = data_names["dataset_filename"]
    dformalname = data_names["dataset_formalname"]
    dataset_info = get_dataset_information(dfilename)
    DATASET_INFOS[dformalname] = dataset_info


if __name__ == "__main__":
    id_keys = ["dataset_name", "path"]

    # new_col_key_stem = ["split"]
    new_cols_flat_names = ["environment_data_collection_method"]
    new_cols_flat_types = [str]
    default_vals = [None]

    for i in range(len(new_cols_flat_names)):
        input_mapping = dict()
        for rollout in rollouts:
            new_val = DATASET_INFOS[rollout.dataset_formalname]["Data Collect Method"]
            input_mapping[tuple(getattr(rollout, k) for k in id_keys)] = new_val

        print(f"prepped {len(input_mapping)} to add to db:")
        print(f"e.g. {set(np.random.choice(list(input_mapping.values()), 50))}")
        print(f"under new name {new_cols_flat_names[i]}")
        print("...confirm?")
        breakpoint()  # break to check things look right

        add_column_with_vals_and_defaults(
            engine=engine,
            new_column_name=new_cols_flat_names[i],
            python_type=new_cols_flat_types[i],
            default_value=default_vals[i],
            key_mapping_col_names=id_keys,
            specific_key_mapping_values=input_mapping,
        )
        print(f"added {new_cols_flat_names[i]}")
