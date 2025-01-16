import os
import time
import traceback
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from sqlalchemy import Engine
from tqdm import tqdm

from ares.configs.base import Rollout
from ares.configs.open_x_embodiment_configs import (
    OpenXEmbodimentEpisode,
    get_dataset_information,
)
from ares.databases.embedding_database import (
    TEST_EMBEDDING_DB_PATH,
    TEST_TIME_STEPS,
    FaissIndex,
    IndexManager,
    rollout_to_embedding_pack,
)
from ares.databases.structured_database import (
    SQLITE_PREFIX,
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    add_rollout,
    setup_database,
    setup_rollouts,
)
from ares.models.extractor import InformationExtractor, RandomInformationExtractor
from ares.models.shortcuts import Embedder, get_nomic_embedder
from ares.name_remapper import DATASET_NAMES
from ares.utils.image_utils import ARES_DATASET_VIDEO_PATH, save_video
from scripts.run_structured_ingestion import (
    build_dataset,
    run_structured_database_ingestion,
)
from scripts.run_trajectory_embedding_ingestion import run_embedding_database_ingestion

if __name__ == "__main__":
    hf_base = "jxu124/OpenX-Embodiment"
    random_extractor = RandomInformationExtractor()  # HACK!!
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    data_dir = "/workspaces/ares/data/oxe/"

    for dataset_info in DATASET_NAMES:
        dataset_filename = dataset_info["dataset_filename"]
        dataset_formalname = dataset_info["dataset_formalname"]
        builder, dataset_dict = build_dataset(dataset_filename, data_dir)
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

            run_structured_database_ingestion(
                ds,
                dataset_info,
                dataset_formalname,
                random_extractor,
                engine,
                dataset_filename,
            )

            # we cant accumulate rollouts and episodes in memory at the same time, so save rollouts
            # to db and videos to disk then reconstitute rollouts for indexing!
            run_embedding_database_ingestion(engine, dataset_formalname)

            # TODO: run grounding here!

        breakpoint()
