import asyncio
import os
import typing as t

from tqdm import tqdm

from ares.constants import (
    ARES_DATA_DIR,
    ARES_OXE_DIR,
    DATASET_NAMES,
    get_dataset_info_by_key,
)
from ares.databases.annotation_database import ANNOTATION_DB_PATH
from ares.databases.embedding_database import EMBEDDING_DB_PATH
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)
from ares.extras.pi_demo_utils import PI_DEMO_TASKS
from ares.models.shortcuts import get_nomic_embedder
from ares.utils.image_utils import get_video_frames
from main import run_ingestion_pipeline

dataset_filename = "pi_demos"
split = "test"
dataset_info = get_dataset_info_by_key("dataset_filename", dataset_filename)
dataset_formalname = dataset_info["dataset_formalname"]

full_dataset_info = {
    "Dataset": dataset_formalname,
    "Dataset Filename": dataset_filename,
    "Dataset Formalname": dataset_formalname,
    "Split": split,
    "Robot": None,
    "Robot Morphology": None,
    "Gripper": None,
    "Action Space": None,
    "# RGB Cams": None,
    "# Depth Cams": None,
    "# Wrist Cams": None,
    "Language Annotations": "Natural",
    "Data Collect Method": "Expert Policy",
    "Scene Type": None,
    "Citation": "year={2024}",
}


def prep_for_oxe_episode(task_info: dict, success_flag: str) -> dict | None:
    """
    Force the PI Demo videos and task information into the OpenXEmbodimentEpisode format.
    """
    filename = f"{task_info['filename_prefix']}_{success_flag}"
    try:
        frames = get_video_frames(dataset_filename="pi_demos", filename=filename)
    except Exception as e:
        print(f"Error getting video frames for {filename}: {e}")
        return None
    metadata = {"file_path": filename, "success": success_flag == "success"}
    steps = []
    for i, frame in enumerate(frames):
        observation = {
            "image": frame,
        }
        steps.append(
            {
                "image": frame,
                "action": None,
                "state": None,
                "is_first": i == 0,
                "is_last": i == len(frames) - 1,
                "is_terminal": False,
                "language_embedding": None,
                "language_instruction": task_info["task"],
                "observation": observation,
            }
        )
    return {"episode_metadata": metadata, "steps": steps}


class PiDemoIngestion:
    def __init__(self, task_infos: list[dict], success_flags: list[str]):
        self.task_infos = task_infos
        self.success_flags = success_flags
        self._episodes = []
        for task_info in tqdm(task_infos):
            for success_flag in success_flags:
                episode = prep_for_oxe_episode(task_info, success_flag)
                if episode is not None:
                    self._episodes.append(episode)
        self._index = 0

    def __iter__(self) -> "PiDemoIngestion":
        self._index = 0
        return self

    def __next__(self) -> dict:
        if self._index >= len(self._episodes):
            raise StopIteration
        episode = self._episodes[self._index]
        self._index += 1
        return episode

    def __len__(self) -> int:
        return len(self._episodes)


if __name__ == "__main__":
    vlm_name = "gpt-4o"
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    embedder = get_nomic_embedder()
    task_infos = list(PI_DEMO_TASKS.values())
    # the PI Demo videos are enormous, so we can only ingest them one-at-a-time
    for task_info in tqdm(task_infos):
        for flag in ["success", "fail"]:
            print(task_info)
            ds = PiDemoIngestion([task_info], [flag])
            run_ingestion_pipeline(
                ds,
                full_dataset_info,
                dataset_formalname,
                vlm_name,
                engine,
                dataset_filename,
                embedder,
                split,
            )
