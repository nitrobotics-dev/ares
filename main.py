import tensorflow_datasets as tfds

from ares.configs.open_x_embodiment_configs import get_dataset_information
from ares.constants import ARES_OXE_DIR, DATASET_NAMES
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
    setup_rollouts,
)
from ares.models.extractor import VLMInformationExtractor
from ares.models.shortcuts import get_gpt_4o_mini, get_nomic_embedder
from scripts.run_grounding_annotation_with_modal import app as modal_app
from scripts.run_grounding_annotation_with_modal import run_modal_grounding
from scripts.run_structured_ingestion import (
    build_dataset,
    run_structured_database_ingestion,
)
from scripts.run_trajectory_embedding_ingestion import (
    run_embedding_database_ingestion_per_dataset,
)

if __name__ == "__main__":
    # vlm = get_gpt_4o()
    vlm = get_gpt_4o_mini()
    vlm_extractor = VLMInformationExtractor(vlm)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
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
            run_structured_database_ingestion(
                ds,
                dataset_info,
                dataset_formalname,
                vlm_extractor,
                engine,
                dataset_filename,
            )

            # we cant accumulate rollouts and episodes in memory at the same time, so save rollouts
            # to db and videos to disk then reconstitute rollouts for indexing!
            rollouts = setup_rollouts(engine, dataset_formalname)
            run_embedding_database_ingestion_per_dataset(rollouts, embedder)

            # run grounding annotation with modal
            with modal_app.run():
                run_modal_grounding(dataset_filename=dataset_filename, split=split)
