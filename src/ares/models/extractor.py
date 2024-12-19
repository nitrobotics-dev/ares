import json
import re
import string
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow_datasets.core import DatasetInfo

from ares.configs.base import (
    BaseConfig,
    Environment,
    Robot,
    Rollout,
    Task,
    Trajectory,
    pydantic_to_example_dict,
    pydantic_to_field_instructions,
)
from ares.configs.open_x_embodiment_configs import OpenXEmbodimentEpisode
from ares.models.llm import LLM


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


def hard_coded_dataset_info_extraction_spreadsheet(dataset_info: dict) -> dict:
    year = None
    if "Citation" in dataset_info and not pd.isna(dataset_info["Citation"]):
        match = re.search(r"year\s*=\s*\{(\d{4})\}", dataset_info["Citation"])
        if match:
            year = int(match.group(1))
    return {
        "rollout": {
            "dataset_name": dataset_info["Dataset"],
            "creation_time": year,
            "ingestion_time": datetime.now(),
        },
        "robot": {
            "embodiment": dataset_info["Robot"],
            "gripper": dataset_info["Gripper"],
            "morphology": dataset_info["Robot Morphology"],
            "action_space": dataset_info["Action Space"],
            "rgb_cams": dataset_info["# RGB Cams"],
            "depth_cams": dataset_info["# Depth Cams"],
            "wrist_cams": dataset_info["# Wrist Cams"],
        },
        "environment": {
            "name": dataset_info["Scene Type"],
            "simulation": dataset_info["Data Collect Method"] == "Human VR",
        },
        "task": {
            "language_instruction_type": dataset_info["Language Annotations"],
        },
    }


# hard coded tfds builder info extraction
def hard_coded_dataset_info_extraction_tfds(dataset_info: DatasetInfo) -> dict:
    # get year from citation
    year = None
    if "year" in dataset_info.citation:
        match = re.search(r"year\s*=\s*\{(\d{4})\}", dataset_info.citation)
        if match:
            year = int(match.group(1))
    # TODO: can also get month, try to be a bit more precise

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
        "rollout": {
            "path": episode.episode_metadata.file_path,
            "length": len(steps),
        },
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
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        task_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        llm_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
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
                list(string.ascii_lowercase),
                size=np.random.randint(1, length_bound + 1),
            )
        )

    def finish_random_object(
        self, object: t.Type[BaseConfig], kwargs: t.Dict[str, t.Any]
    ) -> BaseConfig:
        # Get all fields from the model class
        fields = object.model_fields
        filled_kwargs = kwargs.copy()

        # Fill in missing fields with random values based on their type
        for field_name, field_info in fields.items():
            if field_name not in filled_kwargs:
                field_type = field_info.annotation

                # Handle different field types
                if field_type == str:
                    filled_kwargs[field_name] = self.random_string()
                elif field_type == int:
                    filled_kwargs[field_name] = np.random.randint(0, 10)
                elif field_type == float:
                    filled_kwargs[field_name] = np.random.uniform(0, 1)
                elif field_type == bool:
                    filled_kwargs[field_name] = bool(np.random.choice([True, False]))
                elif field_type == datetime:
                    filled_kwargs[field_name] = datetime.now()
                elif t.get_origin(field_type) == list:
                    # For lists, create a random-length list of random values
                    elem_type = t.get_args(field_type)[0]
                    length = np.random.randint(1, 5)
                    if elem_type == str:
                        filled_kwargs[field_name] = [
                            self.random_string() for _ in range(length)
                        ]
                    elif elem_type in (int, float):
                        filled_kwargs[field_name] = np.random.rand(length).tolist()
                    else:
                        filled_kwargs[field_name] = []

        # Create and return the object with all fields filled
        return object(**filled_kwargs)

    def extract(
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        task_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        llm_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> Rollout:
        robot_kwargs = robot_kwargs or {}
        environment_kwargs = environment_kwargs or {}
        task_kwargs = task_kwargs or {}
        llm_kwargs = llm_kwargs or {}

        dataset_info_dict = hard_coded_dataset_info_extraction_spreadsheet(dataset_info)
        episode_info_dict = hard_coded_episode_info_extraction(episode)
        hardcoded_info = merge_dicts(dataset_info_dict, episode_info_dict)

        # Create component objects in a loop
        components = {
            "robot": Robot,
            "environment": Environment,
            "task": Task,
            "trajectory": Trajectory,
        }
        objects = {
            name: self.finish_random_object(cls, hardcoded_info[name])
            for name, cls in components.items()
        }

        # Create final rollout with all components
        rollout = self.finish_random_object(
            Rollout,
            {**hardcoded_info["rollout"], **objects},
        )
        return rollout


class LLMInformationExtractor(InformationExtractor):
    def __init__(self, llm: LLM):
        self.llm = llm

    def finish_llm_object(
        self,
        object: t.Type[BaseConfig],
        hardcoded_info: dict,
        structured_info: dict,
        extra_kwargs: dict,
    ) -> BaseConfig:
        # Merge all sources of information for this object
        merged_info = merge_several_dicts(
            [
                hardcoded_info.get(object.__name__.lower(), {}),
                structured_info.get(object.__name__.lower(), {}),
                extra_kwargs,
            ]
        )
        return object(**merged_info)

    def extract(
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        task_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        llm_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> Rollout:
        # Initialize kwargs
        robot_kwargs = robot_kwargs or {}
        environment_kwargs = environment_kwargs or {}
        task_kwargs = task_kwargs or {}
        llm_kwargs = llm_kwargs or {}

        # Get hardcoded information
        dataset_info_dict = hard_coded_dataset_info_extraction_spreadsheet(dataset_info)
        episode_info_dict = hard_coded_episode_info_extraction(episode)
        hardcoded_info = merge_dicts(dataset_info_dict, episode_info_dict)

        # Get LLM-extracted information
        images = [step.observation.image for step in episode.steps]
        # HACK
        if len(images) > 10:
            # select 10 evenly spaced images --> update with FPS sampling
            images = images[:: len(images) // 10]

        info = {
            "task": episode.steps[0].language_instruction,
            "field_instructions": (
                "Please provide the following required information:\n"
                + "\n".join(
                    pydantic_to_field_instructions(
                        Rollout, exclude_fields=hardcoded_info, required_only=True
                    )
                )
            ),
            "example_response_format": pydantic_to_example_dict(
                Rollout, exclude_fields=hardcoded_info, required_only=True
            ),
        }
        breakpoint()
        messages, response = self.llm.ask(
            prompt_filename=llm_kwargs.get("prompt_filename", "test_prompt.jinja2"),
            images=images,
            info=info,
        )
        # Parse the response content as JSON if it's a string
        content = response.choices[0].message.content
        # Remove markdown code block formatting
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_info = json.loads(content) if isinstance(content, str) else content
        # Create component objects in a loop
        components = {
            "robot": (Robot, robot_kwargs),
            "environment": (Environment, environment_kwargs),
            "task": (Task, task_kwargs),
            "trajectory": (Trajectory, {}),
        }
        objects = {
            name: self.finish_llm_object(cls, hardcoded_info, structured_info, kwargs)
            for name, (cls, kwargs) in components.items()
        }

        # Create final rollout with all components
        # Include both hardcoded rollout info and top-level fields from structured_info
        rollout_kwargs = {
            **hardcoded_info["rollout"],
            **{k: v for k, v in structured_info.items() if not isinstance(v, dict)},
            **objects,
        }
        breakpoint()
        return Rollout(**rollout_kwargs)
