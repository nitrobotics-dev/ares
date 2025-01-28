import asyncio
import os

from ares.configs.open_x_embodiment_configs import get_dataset_information
from ares.constants import ARES_DATA_DIR, ARES_OXE_DIR, DATASET_NAMES
from ares.databases.annotation_database import ANNOTATION_DB_PATH
from ares.databases.embedding_database import EMBEDDING_DB_PATH
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
    setup_rollouts,
)
from ares.models.shortcuts import get_nomic_embedder
from scripts.annotating.annotation_base import orchestrate_annotating
from scripts.annotating.run_grounding import GroundingModalAnnotatingFn
from scripts.run_structured_ingestion import (
    build_dataset,
    run_structured_database_ingestion,
)
from scripts.run_trajectory_embedding_ingestion import (
    run_embedding_database_ingestion_per_dataset,
)

if __name__ == "__main__":
    vlm_name = "gpt-4o-mini"
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    embedder = get_nomic_embedder()

    for i, dataset_info in enumerate(DATASET_NAMES):

        dataset_filename = dataset_info["dataset_filename"]
        dataset_formalname = dataset_info["dataset_formalname"]
        builder, dataset_dict = build_dataset(dataset_filename, ARES_OXE_DIR)
        print(
            f"working on {dataset_formalname} with splits {list(dataset_dict.keys())}"
        )

        for split in dataset_dict.keys():
            ds = dataset_dict[split]
            print(f"found {len(ds)} episodes in {split}")
            dataset_info = get_dataset_information(dataset_filename)

            # hardcode a few additional fields
            dataset_info["Dataset Filename"] = dataset_filename
            dataset_info["Dataset Formalname"] = dataset_formalname
            dataset_info["Split"] = split

            # run structured ingestion
            # new_rollout_ids = None
            fails, new_rollout_ids = asyncio.run(
                run_structured_database_ingestion(
                    ds,
                    dataset_info,
                    dataset_formalname,
                    vlm_name,
                    engine,
                    dataset_filename,
                )
            )

            # # we cant accumulate rollouts and episodes in memory at the same time, so save rollouts
            # # to db and videos to disk then reconstitute rollouts for indexing!
            rollouts = setup_rollouts(engine, dataset_formalname)
            if new_rollout_ids is not None:
                rollouts = [r for r in rollouts if r.id in new_rollout_ids]
            if len(rollouts) == 0:
                breakpoint()
            run_embedding_database_ingestion_per_dataset(
                rollouts, embedder, index_path=EMBEDDING_DB_PATH
            )

            # run grounding annotation with modal
            orchestrate_annotating(
                engine_path=ROBOT_DB_PATH,
                ann_db_path=ANNOTATION_DB_PATH,
                annotating_fn=GroundingModalAnnotatingFn(),
                rollout_ids=[r.id for r in rollouts],
                failures_path=os.path.join(
                    ARES_DATA_DIR,
                    "annotating_failures",
                    f"grounding_{dataset_filename}_{split}.pkl",
                ),
            )
