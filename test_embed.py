from collections import defaultdict
from typing import Dict

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from ares.configs.base import Rollout
from ares.configs.pydantic_sql_helpers import recreate_model
from ares.databases.embedding_database import (
    BASE_EMBEDDING_DB_PATH,
    TEST_EMBEDDING_DB_PATH,
    FaissIndex,
    IndexManager,
    rollout_to_embedding_pack,
    rollout_to_index_name,
)
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)
from ares.models.llm import NOMIC_EMBEDDER as EMBEDDER

TEST_TIME_STEPS = 100
LIMIT = 2**31 - 1  # Maximum 32-bit signed integer

engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
sess = Session(engine)
rows = sess.query(RolloutSQLModel).limit(LIMIT).all()
rollouts = [recreate_model(r, Rollout) for r in rows]

index_manager = IndexManager(TEST_EMBEDDING_DB_PATH, FaissIndex)
unique_dataset_robots_list = list({r.robot.embodiment: r for r in rollouts}.values())

# for unique_rollout in unique_dataset_robots_list:
#     for key in ["states", "actions"]:
#         name = rollout_to_index_name(unique_rollout, suffix=key)
#         feature_dim = getattr(unique_rollout.trajectory, f"{key}_array").shape[-1]
#         index_manager.init_index(
#             name + f"-{key}",
#             feature_dim,
#             TEST_TIME_STEPS,
#             norm_means=None,  # TODO --> need to normalize across sensors on 1 bot
#             norm_stds=None,  # TODO
#         )

# for task, description, video --> no env/robot specific! just one distribution
embedding_dim = EMBEDDER.embed("dummy").shape[0]
for key in ["task", "description"]:  # "video"!!!
    name = key
    index_manager.init_index(
        name,
        feature_dim=embedding_dim,
        time_steps=1,
        norm_means=None,
        norm_stds=None,
    )
    for rollout in tqdm(rollouts, desc=f"Embedding {key}"):
        inp = (
            rollout.task.language_instruction
            if key == "description"
            else rollout.task.success_criteria
        )
        embedding = EMBEDDER.embed(inp)
        index_manager.add_vector(name, embedding, str(rollout.id))

breakpoint()

for rollout in rollouts:
    rollout_embedding_pack = rollout_to_embedding_pack(rollout)
    for index_name, matrix in rollout_embedding_pack.items():
        if not (
            matrix is None
            or (isinstance(matrix, list) and all(x is None for x in matrix))
            or len(matrix.shape) != 2
        ):
            index_manager.add_matrix(index_name, matrix, str(rollout.id))

# breakpoint()
# print(f"Added {len(rollouts)} rollouts to {len(index_manager.indices)} indices")
# print(f"Metadata: {index_manager.metadata}")

# # index_manager.save()

# # for each row, lets try retrieving a noisy version of itself
# successes = []
# search_k = 10
# for rollout in rollouts:
#     for key in ["states", "actions"]:
#         name = rollout_to_index_name(rollout, suffix=key)
#         query_matrix = getattr(rollout.trajectory, f"{key}_array")
#         if query_matrix is None or len(query_matrix.shape) != 2:
#             continue
#         # add noise by dropping random time steps (will get re-interpolated inside query)
#         query_matrix = query_matrix[:: np.random.randint(1, 5)]
#         try:
#             query_matrix = query_matrix + np.random.normal(0, 0.1, query_matrix.shape)
#         except Exception as e:
#             breakpoint()
#         distances, ids, vecs = index_manager.search_matrix(name, query_matrix, search_k)
#         if str(rollout.id) in ids:
#             successes.append(True)
#         else:
#             successes.append(False)

# print(f"Success rate: {np.mean(successes)}")
# breakpoint()
