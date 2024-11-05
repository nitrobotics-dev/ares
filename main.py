import logging
import os
import time

import datasets
import imageio
import numpy as np
import tensorflow_datasets as tfds
from IPython import display
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from ares.configs.base import Environment, Robot, Task, Trajectory
from ares.configs.open_x_embodiment_configs import OpenXEmbodimentEpisode
from ares.database import (
    SQLITE_PREFIX,
    TEST_ROBOT_DB_PATH,
    TrajectorySQLModel,
    add_trajectories,
    add_trajectory,
    setup_database,
)
from ares.extractor import RandomInformationExtractor


def build_dataset(
    dataset_name: str, data_dir: str
) -> tuple[tfds.builder, tfds.datasets]:
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()
    datasets = builder.as_dataset()
    return builder, datasets


if __name__ == "__main__":
    hf_base = "jxu124/OpenX-Embodiment"
    # dataset_name = "ucsd_kitchen_dataset_converted_externally_to_rlds"
    dataset_name = "cmu_play_fusion"
    data_dir = "/workspaces/ares/data"

    builder, datasets = build_dataset(dataset_name, data_dir)
    dataset_info = builder.info
    ds = datasets["train"]

    random_extractor = RandomInformationExtractor()

    os.remove(TEST_ROBOT_DB_PATH.replace(SQLITE_PREFIX, ""))
    engine = setup_database(path=TEST_ROBOT_DB_PATH)

    trajectories = []
    add_and_commit_times = []

    for i, ep in tqdm(enumerate(ds)):
        episode = OpenXEmbodimentEpisode(**ep)
        trajectory = random_extractor.extract(
            episode=episode, dataset_info=dataset_info
        )
        trajectories.append(trajectory)
        # just track this
        start_time = time.time()
        add_trajectory(engine, trajectory)
        add_and_commit_times.append(time.time() - start_time)
        if i > 50:
            break

    print(
        f"mean (sum) --> add and commit time: {np.mean(add_and_commit_times), np.sum(add_and_commit_times)}"
    )

    os.remove(TEST_ROBOT_DB_PATH.replace(SQLITE_PREFIX, ""))
    engine = setup_database(path=TEST_ROBOT_DB_PATH)

    tic = time.time()
    add_trajectories(engine, trajectories)
    bunch_time = time.time() - tic

    print(f"time to add all trajectories: {np.mean(bunch_time), np.sum(bunch_time)}")

    sess = Session(engine)
    # row_count = sess.execute(
    #     select(func.count()).select_from(TrajectorySQLModel)
    # ).scalar_one()
    # res = (
    #     sess.query(TrajectorySQLModel)
    #     .filter(TrajectorySQLModel.task_success > 0.5)
    #     .all()
    # )
    # print(f"mean wins: {len(res) / row_count}")
    #  res = sess.scalars(sess.query(TrajectorySQLModel.task_language_instruction)).all()
    breakpoint()
