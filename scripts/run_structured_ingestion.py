import os
import time
import traceback
from pathlib import Path

import click
import numpy as np
import tensorflow_datasets as tfds
from sqlalchemy import Engine
from tqdm import tqdm

from ares.configs.base import Rollout
from ares.configs.open_x_embodiment_configs import (
    OpenXEmbodimentEpisode,
    construct_openxembodiment_episode,
    get_dataset_information,
)
from ares.databases.structured_database import (
    RolloutSQLModel,
    add_rollout,
    setup_database,
)
from ares.models.extractor import InformationExtractor
from ares.utils.image_utils import ARES_DATASET_VIDEO_PATH, save_video


def build_dataset(
    dataset_name: str, data_dir: str
) -> tuple[tfds.builder, tfds.datasets]:
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()
    dataset_dict = builder.as_dataset()
    return builder, dataset_dict


def maybe_save_video(
    episode: OpenXEmbodimentEpisode, dataset_filename: str, path: str
) -> None:
    video = [step.observation.image for step in episode.steps]
    fname = str(Path(path.removeprefix("/")).with_suffix(""))
    if not os.path.exists(
        os.path.join(ARES_DATASET_VIDEO_PATH, dataset_filename, fname + ".mp4")
    ):
        save_video(video, dataset_filename, fname)


def run_structured_database_ingestion(
    ds: tfds.datasets,
    dataset_info: dict,
    dataset_formalname: str,
    extractor: InformationExtractor,
    engine: Engine,
    dataset_filename: str,
) -> None:
    tic = time.time()
    n_rollouts = 0
    for i, ep in tqdm(enumerate(ds), desc=f"Ingesting {dataset_formalname} rollouts"):
        try:
            if i == 0:
                print(list(ep["steps"])[0]["observation"].keys())

            # construct the OpenXEmbodiment Episode
            episode = construct_openxembodiment_episode(ep, dataset_info, i)

            # complete the raw information with the rollout request (random for now)
            rollout = extractor.extract(episode=episode, dataset_info=dataset_info)
            # potentially save the video as mp4 and frames
            maybe_save_video(
                episode, dataset_filename, episode.episode_metadata.file_path
            )

            # add the rollout to the database
            add_rollout(engine, rollout, RolloutSQLModel)
            n_rollouts += 1
        except Exception as e:
            print(f"Error processing episode {i} during rollout extraction: {e}")
            print(traceback.format_exc())
            breakpoint()

    if n_rollouts == 0:
        breakpoint()
    print(f"Structured database new rollouts: {n_rollouts}")
    total_time = time.time() - tic
    print(f"Structured database time: {total_time}")
    print(f"Structured database mean time: {total_time / n_rollouts}")


@click.command()
@click.option(
    "--dataset-filename",
    type=str,
    required=True,
    help="Filename of the dataset to process",
)
@click.option(
    "--dataset-formalname",
    type=str,
    required=True,
    help="Formal name of the dataset to process",
)
@click.option(
    "--data-dir",
    type=str,
    required=True,
    help="Directory to store the dataset",
)
@click.option(
    "--engine-url",
    type=str,
    required=True,
    help="SQLAlchemy database URL",
)
def main(
    dataset_filename: str,
    dataset_formalname: str,
    data_dir: str,
    engine_url: str,
) -> None:
    extractor = InformationExtractor()  # HACK
    engine = setup_database(RolloutSQLModel, path=engine_url)
    builder, dataset_dict = build_dataset(dataset_filename, data_dir)
    for split in dataset_dict.keys():
        ds = dataset_dict[split]
        dataset_info = get_dataset_information(dataset_filename)
        dataset_info["Dataset Filename"] = dataset_filename
        dataset_info["Dataset Formalname"] = dataset_formalname
        dataset_info["Split"] = split

        run_structured_database_ingestion(
            ds,
            dataset_info,
            dataset_formalname,
            extractor,
            engine,
            dataset_filename,
        )
        print(f"Ingested {len(ds)} rollouts for split {split}")


if __name__ == "__main__":
    main()
