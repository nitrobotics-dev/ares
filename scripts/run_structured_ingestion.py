"""
Helper script to ingest structured data into the database. This script is used in `main` in order to ingest a dataset into the database. We do this by converting a
dataset into a RLDS-style TFDS, extracting the rollouts from the TFDS, and then adding the rollouts to the database.
"""

import asyncio
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import click
import tensorflow_datasets as tfds
from sqlalchemy import Engine
from tqdm import tqdm

from ares.configs.open_x_embodiment_configs import (
    OpenXEmbodimentEpisode,
    construct_openxembodiment_episode,
    get_dataset_information,
)
from ares.constants import ARES_VIDEO_DIR
from ares.databases.structured_database import (
    RolloutSQLModel,
    add_rollout,
    get_partial_df,
    setup_database,
)
from ares.models.extractor import InformationExtractor, VLMInformationExtractor
from ares.models.shortcuts import get_vlm
from ares.utils.image_utils import save_video


@dataclass
class BatchResult:
    n_new: int = 0
    n_skipped: int = 0
    fails: list[dict] = field(default_factory=list)
    new_ids: set[uuid.UUID] = field(default_factory=set)


def build_dataset(
    dataset_name: str, data_dir: str
) -> tuple[tfds.builder, tfds.datasets]:
    """Build and prepare the dataset."""
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()
    dataset_dict = builder.as_dataset()
    return builder, dataset_dict


def maybe_save_video(episode: OpenXEmbodimentEpisode, dataset_filename: str) -> None:
    """Save video if it doesn't already exist."""
    video = [step.observation.image for step in episode.steps]
    path = episode.episode_metadata.file_path
    fname = str(Path(path.removeprefix("/")).with_suffix(""))
    if not os.path.exists(
        os.path.join(ARES_VIDEO_DIR, dataset_filename, fname + ".mp4")
    ):
        save_video(video, dataset_filename, fname)


async def process_batch(
    episodes: list[tuple[int, dict]],  # (index, episode) pairs
    dataset_info: dict,
    extractor: InformationExtractor,
    engine: Engine,
    dataset_filename: str,
    existing_paths: set[str],
) -> BatchResult:
    """Process a batch of episodes with parallel VLM extraction."""
    result = BatchResult()
    valid_episodes = []

    # Filter and construct episodes
    for i, ep in episodes:
        try:
            episode = construct_openxembodiment_episode(ep, dataset_info, i)
            if episode.episode_metadata.file_path in existing_paths:
                result.n_skipped += 1
                continue
            valid_episodes.append((i, episode))
        except Exception as e:
            result.fails.append(
                {"index": i, "error": e, "traceback": traceback.format_exc()}
            )

    if not valid_episodes:
        return result

    # Save videos before VLM extraction
    for _, episode in valid_episodes:
        try:
            maybe_save_video(episode, dataset_filename)
        except Exception as e:
            result.fails.append(
                {
                    "error": e,
                    "traceback": traceback.format_exc(),
                    "path": episode.episode_metadata.file_path,
                }
            )

    # Parallel VLM extraction
    try:
        rollouts = await extractor.extract_batch(
            episodes=[ep for _, ep in valid_episodes], dataset_info=dataset_info
        )

        # Sequential database operations
        for (i, episode), rollout in zip(valid_episodes, rollouts):
            try:
                if isinstance(rollout, dict):
                    # there was an error in processing
                    result.fails.append(rollout)
                    continue
                add_rollout(engine, rollout, RolloutSQLModel)
                result.new_ids.add(rollout.id)
                result.n_new += 1
            except Exception as e:
                result.fails.append(
                    {
                        "index": i,
                        "error": e,
                        "traceback": traceback.format_exc(),
                        "path": episode.episode_metadata.file_path,
                    }
                )
    except Exception as e:
        result.fails.append({"error": e, "traceback": traceback.format_exc()})

    return result


async def run_structured_database_ingestion(
    ds: tfds.datasets,
    dataset_info: dict,
    dataset_formalname: str,
    vlm_name: str,
    engine: Engine,
    dataset_filename: str,
    batch_size: int = 200,
) -> tuple[list[dict], set[uuid.UUID]]:
    """Process dataset in batches with parallel VLM requests."""
    tic = time.time()
    total_result = BatchResult()

    # Get existing paths once
    existing_id_df = get_partial_df(engine, ["dataset_filename", "path"])
    existing_paths = set(
        existing_id_df[existing_id_df["dataset_filename"] == dataset_filename][
            "path"
        ].tolist()
    )

    # Create a single VLM instance to be shared across all batches
    vlm = get_vlm(vlm_name)
    extractor = VLMInformationExtractor(vlm)

    # Process in streaming batches
    current_batch = []

    for i, ep in tqdm(enumerate(ds), desc=f"Ingesting {dataset_formalname} rollouts"):
        if i == 0:
            print(list(ep["steps"])[0]["observation"].keys())

        current_batch.append((i, ep))

        if len(current_batch) >= batch_size:
            result = await process_batch(
                current_batch,
                dataset_info,
                extractor,
                engine,
                dataset_filename,
                existing_paths,
            )
            total_result.n_new += result.n_new
            total_result.n_skipped += result.n_skipped
            total_result.fails.extend(result.fails)
            total_result.new_ids.update(result.new_ids)
            current_batch = []
            if result.n_new == 0 and result.n_skipped == 0 and len(result.fails) != 0:
                breakpoint()
    # Process final batch if any
    if current_batch:
        result = await process_batch(
            current_batch,
            dataset_info,
            extractor,
            engine,
            dataset_filename,
            existing_paths,
        )
        total_result.n_new += result.n_new
        total_result.n_skipped += result.n_skipped
        total_result.fails.extend(result.fails)
        total_result.new_ids.update(result.new_ids)

    if total_result.n_new == 0 and total_result.n_skipped == 0:
        breakpoint()

    print(f"Structured database new rollouts: {total_result.n_new}")
    total_time = time.time() - tic
    print(f"Structured database time: {total_time}")
    if total_result.n_new > 0:
        print(f"Structured database mean time: {total_time / total_result.n_new}")

    return total_result.fails, total_result.new_ids


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
    vlm_name = "gpt-4o-mini"
    engine = setup_database(RolloutSQLModel, path=engine_url)
    builder, dataset_dict = build_dataset(dataset_filename, data_dir)
    for split in dataset_dict.keys():
        ds = dataset_dict[split]
        dataset_info = get_dataset_information(dataset_filename)
        dataset_info["Dataset Filename"] = dataset_filename
        dataset_info["Dataset Formalname"] = dataset_formalname
        dataset_info["Split"] = split

        asyncio.run(
            run_structured_database_ingestion(
                ds,
                dataset_info,
                dataset_formalname,
                vlm_name,
                engine,
                dataset_filename,
            )
        )
        print(f"Ingested {len(ds)} rollouts for split {split}")


if __name__ == "__main__":
    main()
