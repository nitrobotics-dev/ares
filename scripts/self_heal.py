"""
Errors happen! We want a service to self-heal -- that is, ensure that our databases are synced.
This script is a two-step process: 
1. Run `find-heal` to find which rollouts are missing from the embedding database and annotation database. This saves a list of ids to disk to be used in the next step.
2. Run `exec-heal` to ingest the missing rollouts into the embedding database and update the annotation database.
This ensures our databases are in sync.
"""

import os
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from ares.constants import ARES_DATA_DIR
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.embedding_database import (
    EMBEDDING_DB_PATH,
    META_INDEX_NAMES,
    FaissIndex,
    IndexManager,
    rollout_to_index_name,
)
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    get_partial_df,
    get_rollout_by_name,
    setup_database,
)

from .annotating.run_grounding import run_modal_grounding
from .run_trajectory_embedding_ingestion import (
    main as run_trajectory_embedding_ingestion,
)

HEALING_EXCEPTIONS = {
    "utokyo_saytap_converted_externally_to_rlds": ["grounding"],
    "CMU Franka Exploration": ["CMU Franka Exploration-Franka-states"],
    "USC Jaco Play": ["USC Jaco Play-Jaco 2-states"],
}
HEAL_INFO_DIR = os.path.join(ARES_DATA_DIR, "heal_info")


@click.command("find-heal")
@click.option("--heal-info-dir", type=str, default=HEAL_INFO_DIR)
def find_heal_opportunities(heal_info_dir: str) -> str:
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    ann_db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)
    embedding_db = IndexManager(EMBEDDING_DB_PATH, FaissIndex)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    heal_dir = os.path.join(heal_info_dir, time_str)
    os.makedirs(heal_dir, exist_ok=True)

    # collect all rollout IDs from structured database (engine)
    id_cols = ["id", "dataset_filename", "dataset_formalname", "filename"]
    rollout_df = get_partial_df(engine, id_cols)
    dataset_formalname_to_df = {
        k: v for k, v in rollout_df.groupby("dataset_formalname")
    }
    dataset_filename_to_df = {k: v for k, v in rollout_df.groupby("dataset_filename")}

    # check embedding database
    to_update_embedding_index_ids = []
    for dataset_formalname, id_df in dataset_formalname_to_df.items():
        if "embedding" in HEALING_EXCEPTIONS.get(dataset_formalname, []):
            continue
        example_rollout = get_rollout_by_name(
            engine, dataset_formalname, id_df["filename"].iloc[0]
        )
        potential_index_names = [
            rollout_to_index_name(example_rollout, suffix)
            for suffix in ["states", "actions"]
        ] + META_INDEX_NAMES  # description, task
        for index_name in potential_index_names:
            if index_name in HEALING_EXCEPTIONS.get(dataset_formalname, []):
                continue
            if index_name not in embedding_db.indices:
                missing_ids = id_df["id"].tolist()
                existing_index_ids = []
            else:
                existing_index = embedding_db.indices[index_name]
                existing_index_ids = existing_index.get_all_ids()
                # add any missing ids to update list
                missing_ids = set(id_df["id"].astype(str).tolist()) - set(
                    existing_index_ids.tolist()
                )
            if len(missing_ids) > 0:
                n_existing = len(existing_index_ids)
                pct_missing = (
                    100
                    * len(missing_ids)
                    / (n_existing if n_existing > 0 else len(missing_ids))
                )
                print(
                    f"Found {len(missing_ids)} missing ids for index {index_name} out of {n_existing} existing ids; {pct_missing:.2f}% missing from dataset {dataset_formalname}"
                )
                to_update_embedding_index_ids.extend(missing_ids)

    update_embedding_ids_path = os.path.join(heal_dir, "update_embedding_ids.txt")
    with open(update_embedding_ids_path, "w") as f:
        for id in to_update_embedding_index_ids:
            f.write(f"{id}\n")
    print(
        f"Found {len(to_update_embedding_index_ids)} ids to update in embedding database; saving to disk at {update_embedding_ids_path}"
    )

    print("\n\n" + "=" * 100 + "\n\n")
    # to update grounding
    to_update_grounding_ids = []
    existing_video_ids = pd.Series(ann_db.get_video_ids())
    for dataset_filename, id_df in dataset_filename_to_df.items():
        if "grounding" in HEALING_EXCEPTIONS.get(dataset_filename, []):
            to_update_grounding_ids.extend(id_df["id"].tolist())
        # check if videos exists -- if not, add to list (will add video and grounding)
        found_video_ids = (id_df["dataset_filename"] + "/" + id_df["filename"]).apply(
            lambda x: str(Path(x).with_suffix(".mp4"))
        )
        mask = ~found_video_ids.isin(existing_video_ids)
        if mask.any():
            print(f"Found {mask.sum()} missing videos for dataset {dataset_filename}")
            to_update_grounding_ids.extend(id_df[mask]["id"].astype(str).tolist())

        # Handle videos that exist but are missing annotations
        has_video_mask = found_video_ids.isin(existing_video_ids)
        videos_with_annotations = pd.Series(ann_db.get_annotation_ids())
        missing_annotations_mask = ~found_video_ids[has_video_mask].isin(
            videos_with_annotations
        )
        if missing_annotations_mask.any():
            print(
                f"Found {missing_annotations_mask.sum()} videos missing annotations for dataset {dataset_filename}"
            )
            to_update_grounding_ids.extend(
                id_df[has_video_mask][missing_annotations_mask]["id"]
                .astype(str)
                .tolist()
            )

    update_grounding_ids_path = os.path.join(heal_dir, "update_grounding_ids.txt")
    to_update_grounding_ids = list(set(to_update_grounding_ids))  # remove duplicates
    with open(update_grounding_ids_path, "w") as f:
        for id in to_update_grounding_ids:
            f.write(f"{id}\n")
    print(
        f"Found {len(to_update_grounding_ids)} ids to update in grounding database; saving to disk at {update_grounding_ids_path}"
    )
    print(f"TIME DIR: {time_str}")


@click.command("exec-heal")
@click.option("--time-dir", type=str, required=True)
def execute_heal(time_dir: str):
    heal_dir = os.path.join(HEAL_INFO_DIR, time_dir)

    # run embedding ingestion via click's command from our embedding ingestion script
    update_embedding_ids_path = os.path.join(heal_dir, "update_embedding_ids.txt")
    run_trajectory_embedding_ingestion.callback(
        engine_url=ROBOT_DB_PATH,
        dataset_formalname=None,
        from_id_file=update_embedding_ids_path,
        index_path=EMBEDDING_DB_PATH,
    )

    # update grounding database
    update_grounding_ids_path = os.path.join(heal_dir, "update_grounding_ids.txt")
    run_modal_grounding(
        engine_path=ROBOT_DB_PATH,
        ann_db_path=ANNOTATION_DB_PATH,
        retry_failed_path=update_grounding_ids_path,
    )

    print(f"Finished healing")


@click.group()
def cli():
    """Self-healing utilities for database synchronization"""
    pass


cli.add_command(find_heal_opportunities)
cli.add_command(execute_heal)

if __name__ == "__main__":
    cli()
