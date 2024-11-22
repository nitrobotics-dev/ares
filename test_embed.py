from collections import defaultdict
from typing import Dict

import numpy as np
from sqlalchemy.orm import Session

from ares.configs.base import Rollout
from ares.configs.pydantic_sql_helpers import recreate_model
from ares.databases.embedding_database import (
    BASE_EMBEDDING_DB_PATH,
    TEST_EMBEDDING_DB_PATH,
    FaissIndex,
    IndexManager,
)
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)


def rollout_to_index_name(rollout: Rollout) -> str:
    return f"{rollout.dataset_name}-{rollout.robot.embodiment}"


def rollout_to_embedding_pack(rollout: Rollout) -> Dict[str, np.ndarray | None]:
    name = rollout_to_index_name(rollout)
    return {
        f"{name}-states": rollout.trajectory.states_array,
        f"{name}-actions": rollout.trajectory.actions_array,
    }


TEST_TIME_STEPS = 100
LIMIT = 2**31 - 1  # Maximum 32-bit signed integer

engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
sess = Session(engine)
rows = sess.query(RolloutSQLModel).limit(LIMIT).all()
rollouts = [recreate_model(r, Rollout) for r in rows]

index_manager = IndexManager(TEST_EMBEDDING_DB_PATH, FaissIndex)
unique_dataset_robots_list = list({r.robot.embodiment: r for r in rollouts}.values())

for unique_rollout in unique_dataset_robots_list:
    name = rollout_to_index_name(unique_rollout)
    for key in ["states", "actions"]:
        feature_dim = getattr(unique_rollout.trajectory, f"{key}_array").shape[-1]
        index_manager.init_index(
            name + f"-{key}",
            feature_dim,
            TEST_TIME_STEPS,
            norm_means=None,
            norm_stds=None,
        )


for rollout in rollouts:
    rollout_embedding_pack = rollout_to_embedding_pack(rollout)
    for index_name, matrix in rollout_embedding_pack.items():
        index_manager.add_matrix(index_name, matrix, str(rollout.id))


print(f"Added {len(rollouts)} rollouts to {len(index_manager.indices)} indices")
print(f"Metadata: {index_manager.metadata}")

# for each row, lets try retrieving a noisy version of itself
successes = []
for rollout in rollouts:
    name = f"{rollout.dataset_name}-{rollout.robot.embodiment}"
    for key in ["states", "actions"]:
        query_matrix = getattr(rollout.trajectory, f"{key}_array")
        # add noise by dropping random time steps (will get re-interpolated inside query)
        query_matrix = query_matrix[:: np.random.randint(1, 5)]
        # query_matrix = query_matrix + np.random.normal(0, 0.1, query_matrix.shape)
        distances, ids, vecs = index_manager.search_matrix(
            name + f"-{key}", query_matrix, 2
        )
        if str(rollout.id) in ids:
            successes.append(True)
        else:
            successes.append(False)

print(f"Success rate: {np.mean(successes)}")
breakpoint()
