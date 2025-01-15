import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from sqlalchemy import Engine, case, func, text
from sqlalchemy.orm import Session
from sqlmodel import SQLModel, select
from tqdm import tqdm

from ares.configs.base import Rollout
from ares.configs.open_x_embodiment_configs import (
    OpenXEmbodimentEpisode,
    OpenXEmbodimentEpisodeMetadata,
    get_dataset_information,
)
from ares.configs.pydantic_sql_helpers import recreate_model
from ares.databases.embedding_database import (
    TEST_EMBEDDING_DB_PATH,
    FaissIndex,
    IndexManager,
    rollout_to_embedding_pack,
)
from ares.databases.structured_database import (
    SQLITE_PREFIX,
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    add_column_with_vals_and_defaults,
    add_rollout,
    setup_database,
    setup_rollouts,
)
from ares.models.extractor import RandomInformationExtractor
from ares.models.shortcuts import Embedder, get_nomic_embedder
from ares.name_remapper import DATASET_NAMES
from ares.utils.image_utils import ARES_DATASET_VIDEO_PATH, save_video


def build_dataset(
    dataset_name: str, data_dir: str
) -> tuple[tfds.builder, tfds.datasets]:
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()
    dataset_dict = builder.as_dataset()
    return builder, dataset_dict


def construct_openxembodiment_episode(
    episode: OpenXEmbodimentEpisode, dataset_info: dict
) -> OpenXEmbodimentEpisode:
    raw_steps = list(ep["steps"])
    if "episode_metadata" not in ep:
        ep["episode_metadata"] = dict(file_path=f"episode_{i}.npy")
    episode = OpenXEmbodimentEpisode(**ep)
    steps = episode.steps

    if episode.episode_metadata is None:
        # construct our own metadata
        episode.episode_metadata = OpenXEmbodimentEpisodeMetadata(
            file_path=f"episode_{i}.npy",  # to mock extension
        )
    return episode


def maybe_save_video(
    episode: OpenXEmbodimentEpisode, dataset_filename: str, path: str
) -> None:
    video = [step.observation.image for step in episode.steps]
    fname = str(Path(path.removeprefix("/")).with_suffix(""))
    if not os.path.exists(
        os.path.join(ARES_DATASET_VIDEO_PATH, dataset_filename, fname + ".mp4")
    ):
        save_video(video, dataset_filename, fname)


def ingest_trajectory_matrices(
    rollouts: list[Rollout], index_manager: IndexManager
) -> None:
    # collect all embedding packs for states, actions (pre-existing)
    embedding_packs = []
    for rollout in rollouts:
        embedding_packs.append(rollout_to_embedding_pack(rollout))

    # collect all the embeddings and get normalizing constants
    for k in embedding_packs[0].keys():
        embeddings = np.concatenate([pack[k] for pack in embedding_packs])
        print(f"found {embeddings.shape} for {k}; (N,K)")
        # find normalizing constants
        means = np.mean(embeddings, axis=0)
        stds = np.std(embeddings, axis=0)
        feature_dim = embeddings.shape[1]
        print(f"found means {means.shape} and stds {stds.shape}")
        # setup index if not already existing
        if k not in index_manager.indices.keys():
            index_manager.init_index(
                k,
                feature_dim,
                TEST_TIME_STEPS,
                norm_means=means,  # normalize with dimension-specific means
                norm_stds=stds,  # normalize with dimension-specific stds
            )
        # add the embeddings to the index! these will be normalized
        for pack in tqdm(embedding_packs, desc=f"Ingesting {k} embeddings"):
            # some datasets do not provide this information
            if isinstance(pack.get(k), np.ndarray):
                index_manager.add_matrix(k, pack[k], str(rollout.id))


def ingest_language_embeddings(
    rollouts: list[Rollout], index_manager: IndexManager, embedder: Embedder
) -> None:
    feature_dim = embedder.embed("test").shape[0]
    for name in ["task", "description"]:
        for rollout in tqdm(rollouts, desc=f"Ingesting {name} embeddings"):
            if name not in index_manager.indices.keys():
                index_manager.init_index(
                    name,
                    feature_dim,
                    TEST_TIME_STEPS,
                    norm_means=None,  # no need to normalize
                    norm_stds=None,  # no need to normalize
                    extra_metadata={"model": embedder.name},
                )

            # HACK!!!! fix the lang instructuion and success criteria weirdness
            inp = (
                rollout.task.language_instruction
                if name == "description"
                else rollout.task.success_criteria
            )
            # some datasets do not provide this information
            if inp is None:
                continue
            embedding = embedder.embed(inp)
            index_manager.add_vector(name, embedding, str(rollout.id))


TEST_TIME_STEPS = 100


if __name__ == "__main__":
    hf_base = "jxu124/OpenX-Embodiment"

    for dataset_info in DATASET_NAMES:
        dataset_filename = dataset_info["dataset_filename"]
        dataset_formalname = dataset_info["dataset_formalname"]

        data_dir = "/workspaces/ares/data/oxe/"
        builder, dataset_dict = build_dataset(dataset_filename, data_dir)
        print(f"working on {dataset_formalname}")
        for split in dataset_dict.keys():
            ds = dataset_dict[split]
            print(f"working on {split} out of {list(dataset_dict.keys())}")
            dataset_info = get_dataset_information(dataset_filename)
            # hardcode a few additional fields
            dataset_info["Dataset Filename"] = dataset_filename
            dataset_info["Dataset Formalname"] = dataset_formalname
            dataset_info["Split"] = split

            print(len(ds))
            random_extractor = RandomInformationExtractor()  # HACK!!

            # os.remove(TEST_ROBOT_DB_PATH.replace(SQLITE_PREFIX, ""))
            engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
            tic = time.time()
            n_rollouts = 0

            # iterate through and add rollouts to the database
            for i, ep in tqdm(
                enumerate(ds), desc=f"Ingesting {dataset_formalname} rollouts"
            ):
                try:
                    if i == 0:
                        print(list(ep["steps"])[0]["observation"].keys())
                        breakpoint()

                    # construct the OpenXEmbodiment Episode
                    episode = construct_openxembodiment_episode(ep, dataset_info)

                    # complete the raw information with the rollout request (random for now)
                    rollout = random_extractor.extract(
                        episode=episode, dataset_info=dataset_info
                    )
                    # potentially save the video as mp4 and frames
                    maybe_save_video(
                        episode, dataset_filename, episode.episode_metadata.file_path
                    )

                    # add the rollout to the database
                    add_rollout(engine, rollout, RolloutSQLModel)
                    n_rollouts += 1
                except Exception as e:
                    print(
                        f"Error processing episode {i} during rollout extraction: {e}"
                    )
                    print(traceback.format_exc())
                    breakpoint()

            if n_rollouts == 0:
                breakpoint()
            print(f"Structured database new rollouts: {n_rollouts}")
            total_time = time.time() - tic
            print(f"Structured database time: {total_time}")
            print(f"Structured database mean time: {total_time / n_rollouts}")

            # we cant accumulate rollouts and episodes in memory at the same time, so save rollouts
            # to db and videos to disk then reconstitute rollouts for indexing!
            rollouts = setup_rollouts(engine, dataset_formalname)
            index_manager = IndexManager(TEST_EMBEDDING_DB_PATH, index_class=FaissIndex)

            tic = time.time()
            # add the trajectory matrices to the index and get normalizing constants for this dataset
            # (states and actions)
            ingest_trajectory_matrices(rollouts, index_manager)
            index_manager.save()

            # add task and description embeddings to the index
            # (task, description)
            embedder = get_nomic_embedder()
            ingest_language_embeddings(rollouts, index_manager, embedder)
            index_manager.save()

            breakpoint()
            print(f"Embedding database new rollouts: {len(rollouts)}")
            total_time = time.time() - tic
            print(f"Embedding database time: {total_time}")
            print(f"Embedding database mean time: {total_time / len(rollouts)}")

            print(
                {
                    k: v
                    for k, v in index_manager.metadata.items()
                    if dataset_formalname in k
                }
            )

        breakpoint()
