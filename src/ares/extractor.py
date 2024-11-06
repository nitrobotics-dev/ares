import os
import re
import string
import typing as t
from datetime import datetime

import numpy as np
from tensorflow_datasets.core import DatasetInfo

from ares.configs.base import (
    Environment,
    Robot,
    Rollout,
    Task,
    Trajectory,
    pydantic_to_example_dict,
    pydantic_to_field_instructions,
)
from ares.configs.open_x_embodiment_configs import OpenXEmbodimentEpisode
from ares.llm import LLM


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_several_dicts(dicts: list[dict]) -> dict:
    merged = dicts[0].copy()
    for d in dicts[1:]:
        merged = merge_dicts(merged, d)
    return merged


# hard coded tfds builder info extraction
def hard_coded_dataset_info_extraction(dataset_info: DatasetInfo) -> dict:
    # get year from citation
    year = None
    if "year" in dataset_info.citation:
        match = re.search(r"year=\{(\d{4})\}", dataset_info.citation)
        if match:
            year = int(match.group(1))

    return {"rollout": {"dataset_name": dataset_info.name, "creation_time": year}}


def hard_coded_episode_info_extraction(episode: OpenXEmbodimentEpisode) -> dict:
    steps = episode.steps
    actions = np.stack([step.action for step in steps]).tolist()
    firsts = [step.is_first for step in steps]
    lasts = [step.is_last for step in steps]
    terminals = [step.is_terminal for step in steps]
    is_first = np.where(firsts)[0][0] if np.any(firsts) else None
    is_last = np.where(lasts)[0][-1] if np.any(lasts) else None
    is_terminal = np.where(terminals)[0][-1] if np.any(terminals) else None
    states = np.stack([step.observation.state for step in steps]).tolist()
    return {
        "rollout": {"path": episode.episode_metadata.file_path},
        "trajectory": {
            "actions": actions,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "states": states,
        },
        "task": {"language_instruction": steps[0].language_instruction},
    }


class InformationExtractor:
    def __init__(self) -> None:
        pass

    def extract(
        self, episode: OpenXEmbodimentEpisode, dataset_info: DatasetInfo
    ) -> Rollout:
        raise NotImplementedError


class RandomInformationExtractor(InformationExtractor):
    """
    Creates a random information extractor that extracts random information from the episode.
    The created object is filled with random string values for testing purposes.
    """

    def random_string(self, length_bound: int = 10) -> str:
        return "".join(
            np.random.choice(
                string.ascii_lowercase.split(),
                size=np.random.randint(1, length_bound + 1),
            )
        )

    def extract(
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        robot_kwargs: t.Dict[str, t.Any] = {},
        environment_kwargs: t.Dict[str, t.Any] = {},
        task_kwargs: t.Dict[str, t.Any] = {},
        llm_kwargs: t.Dict[str, t.Any] = {},
    ) -> Rollout:
        robot = Robot(
            name=robot_kwargs.get("name", self.random_string()),
            sensor=robot_kwargs.get(
                "sensor", np.random.choice(["camera", "wrist", "gripper"])
            ),
        )
        environment = Environment(
            name=self.random_string(),
            lighting=self.random_string(),
            simulation=np.random.choice([True, False]),
        )
        task = Task(
            name=self.random_string(),
            description=self.random_string(),
            success_criteria=self.random_string(),
            success=np.random.uniform(0, 1),
            language_instruction=episode.steps[0].language_instruction,
        )

        return Rollout(
            creation_time=datetime.now(),
            ingestion_time=datetime.now(),
            path=episode.episode_metadata.file_path,
            robot=robot,
            environment=environment,
            task=task,
            length=len(episode.steps),
            dataset_name=dataset_info.name,
            trajectory=Trajectory(),
        )


class LLMInformationExtractor(InformationExtractor):
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        robot_kwargs: t.Dict[str, t.Any] = {},
        environment_kwargs: t.Dict[str, t.Any] = {},
        task_kwargs: t.Dict[str, t.Any] = {},
        llm_kwargs: t.Dict[str, t.Any] = {},
    ) -> Rollout:
        dataset_info_dict = hard_coded_dataset_info_extraction(dataset_info)
        episode_info_dict = hard_coded_episode_info_extraction(episode)
        hardcoded_info = merge_dicts(dataset_info_dict, episode_info_dict)

        # with our hardcoded extraction, we need to tell the LLM which fields NOT to extract
        images = [step.observation.image for step in episode.steps]
        info = {
            "task": episode.steps[0].language_instruction,
            "field_instructions": pydantic_to_field_instructions(
                Rollout, exclude_fields=hardcoded_info
            ),
            "response_format": "",
            "example_response_format": pydantic_to_example_dict(
                Rollout, exclude_fields=hardcoded_info
            ),
        }
        structured_info: t.Dict[str, t.Any] = self.llm.ask(
            **llm_kwargs, images=images, info=info
        )

        # get all the robot stuff
        robot_info = merge_several_dicts(
            [hardcoded_info["robot"], structured_info["robot"], robot_kwargs]
        )
        # get all the environment stuff
        environment_info = merge_several_dicts(
            [
                hardcoded_info["environment"],
                structured_info["environment"],
                environment_kwargs,
            ]
        )
        # get all the task stuff
        task_info = merge_several_dicts(
            [hardcoded_info["task"], structured_info["task"], task_kwargs]
        )
        # get all the trajectory stuff
        trajectory_info = merge_several_dicts(
            [hardcoded_info["trajectory"], structured_info["trajectory"]]
        )

        # instantiate all the objects
        robot = Robot(**robot_info)
        environment = Environment(**environment_info)
        task = Task(**task_info)
        trajectory = Trajectory(**trajectory_info)

        return Rollout(
            creation_time=dataset_info_dict.get("creation_time", datetime.now()),
            ingestion_time=datetime.now(),
            path=episode_info_dict["path"],
            dataset_name=dataset_info_dict["dataset_name"],
            length=len(episode.steps),
            robot=robot,
            environment=environment,
            task=task,
            trajectory=trajectory,
        )
