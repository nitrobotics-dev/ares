import json
import re
import string
import traceback
import typing as t
from datetime import datetime
from pathlib import Path

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
    merge_config_sources,
    merge_dicts,
    merge_several_dicts,
    pydantic_to_example_dict,
    pydantic_to_field_instructions,
)
from ares.configs.open_x_embodiment_configs import OpenXEmbodimentEpisode
from ares.models.base import VLM
from ares.utils.image_utils import load_video_frames


def hard_coded_dataset_info_extraction_spreadsheet(dataset_info: dict) -> dict:
    year = None
    if "Citation" in dataset_info and not pd.isna(dataset_info["Citation"]):
        match = re.search(r"year\s*=\s*\{(\d{4})\}", dataset_info["Citation"])
        if match:
            year = int(match.group(1))
    return {
        "rollout": {
            "dataset_name": dataset_info["Dataset"],
            "dataset_formalname": dataset_info["Dataset Formalname"],
            "dataset_filename": dataset_info["Dataset Filename"],
            "creation_time": year,
            "ingestion_time": datetime.now(),
            "split": dataset_info["Split"],
        },
        "robot": {
            "embodiment": dataset_info["Robot"],
            "gripper": (
                dataset_info["Gripper"]
                if not pd.isna(dataset_info["Gripper"])
                else "None"
            ),
            "morphology": dataset_info["Robot Morphology"],
            "action_space": dataset_info["Action Space"],
            "rgb_cams": dataset_info["# RGB Cams"],
            "depth_cams": dataset_info["# Depth Cams"],
            "wrist_cams": dataset_info["# Wrist Cams"],
        },
        "environment": {
            "name": dataset_info["Scene Type"],
            "simulation": dataset_info["Data Collect Method"] == "Human VR",
            "data_collection_method": dataset_info["Data Collect Method"],
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

    return {"rollout": {"dataset_name": dataset_info.name, "creation_time": year}}


def hard_coded_episode_info_extraction(episode: OpenXEmbodimentEpisode) -> dict:
    # gather data
    steps = episode.steps
    firsts = [step.is_first for step in steps]
    lasts = [step.is_last for step in steps]
    terminals = [step.is_terminal for step in steps]
    rewards = [step.reward for step in steps]
    # get indices instead of steps for first, last, terminal
    is_first = np.where(firsts)[0][0] if np.any(firsts) else None
    is_last = np.where(lasts)[0][-1] if np.any(lasts) else None
    is_terminal = np.where(terminals)[0][-1] if np.any(terminals) else None
    success = episode.episode_metadata.success
    reward_step = None
    if any([x is not None for x in rewards]):
        if np.any(np.array(rewards) == 1):
            reward_step = np.where(np.array(rewards) == 1)[0][0]
        else:
            reward_step = -1
    # gather trajectory data
    actions = np.stack([step.action for step in steps]).tolist()
    states = np.stack([step.observation.state for step in steps]).tolist()
    path = episode.episode_metadata.file_path.removeprefix("/")
    return {
        "rollout": {
            "path": path,
            "filename": str(Path(path).with_suffix("")),
            "length": len(steps),
        },
        "trajectory": {
            "actions": actions,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "states": states,
            "rewards": rewards,
            "reward_step": reward_step,
        },
        "task": {
            "language_instruction": steps[0].language_instruction,
            "success": success,
        },
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
        model_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> Rollout:
        raise NotImplementedError


class VLMInformationExtractor(InformationExtractor):
    def __init__(self, vlm: VLM):
        self.vlm = vlm

    def _prepare_prompt_info(
        self,
        episode: OpenXEmbodimentEpisode,
        hardcoded_info: dict,
    ) -> dict:
        """Prepare the prompt information for VLM."""
        return {
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

    def _create_rollout(
        self,
        hardcoded_info: dict,
        structured_info: dict,
        component_kwargs: dict,
    ) -> Rollout:
        """Create a rollout from structured and hardcoded information."""
        components = {
            "robot": (Robot, component_kwargs.get("robot_kwargs", {})),
            "environment": (
                Environment,
                component_kwargs.get("environment_kwargs", {}),
            ),
            "task": (Task, component_kwargs.get("task_kwargs", {})),
            "trajectory": (Trajectory, component_kwargs.get("trajectory_kwargs", {})),
        }

        objects = {
            name: merge_config_sources(
                ConfigCls, hardcoded_info, structured_info, kwargs
            )
            for name, (ConfigCls, kwargs) in components.items()
        }

        rollout_kwargs = {
            **hardcoded_info["rollout"],
            **{k: v for k, v in structured_info.items() if not isinstance(v, dict)},
            **objects,
        }
        return Rollout(**rollout_kwargs)

    async def extract_batch(
        self,
        episodes: list[OpenXEmbodimentEpisode],
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        task_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        model_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        trajectory_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> list[Rollout]:
        """Process a batch of episodes in parallel."""
        component_kwargs = {
            "robot_kwargs": robot_kwargs or {},
            "environment_kwargs": environment_kwargs or {},
            "task_kwargs": task_kwargs or {},
            "model_kwargs": model_kwargs or {},
            "trajectory_kwargs": trajectory_kwargs or {},
        }

        # Get hardcoded information
        dataset_info_dict = hard_coded_dataset_info_extraction_spreadsheet(dataset_info)
        episode_info_dicts = [hard_coded_episode_info_extraction(ep) for ep in episodes]
        hardcoded_infos = [
            merge_dicts(dataset_info_dict, ep_info) for ep_info in episode_info_dicts
        ]
        print(f"found {len(hardcoded_infos)} hardcoded infos")
        # Prepare prompts and images
        prompts = []
        images_list = []
        rollouts = []
        for episode, hardcoded_info in zip(episodes, hardcoded_infos):
            try:
                # Load video frames
                dataset_filename = dataset_info_dict["rollout"]["dataset_filename"]
                episode_fname = hardcoded_info["rollout"]["filename"]
                images, _ = load_video_frames(
                    dataset_filename,
                    episode_fname,
                    target_fps=1,
                    include_last_frame=True,
                )
                images_list.append(images)

                # Prepare prompt
                prompts.append(self._prepare_prompt_info(episode, hardcoded_info))
            except Exception as e:
                print(f"Error preparing prompt: {e}")
                print(traceback.format_exc())
                error_dict = {
                    "path": hardcoded_info["rollout"]["path"],
                    "error_pattern": "prompt_preparation_failure",
                    "error": traceback.format_exc(),
                }
                # Add None placeholders to keep lists aligned
                prompts.append(None)
                images_list.append(None)
                rollouts.append(error_dict)

        # Filter out failed prompts before batch processing
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        valid_images = [images_list[i] for i in valid_indices]

        # Batch process with VLM (only if we have valid prompts)
        responses = []
        if valid_prompts:
            print(f"batching {len(valid_prompts)} prompts")
            results = await self.vlm.ask_batch_async(
                infos=valid_prompts,
                prompt_filename=component_kwargs["model_kwargs"].get(
                    "prompt_filename", "extractor_prompt.jinja2"
                ),
                images_list=valid_images,
            )
            # Unpack the responses from the results
            responses = [response for _, response in results]

        # Process responses and create rollouts
        for i, response in enumerate(responses):
            try:
                content = response.choices[0].message.content.strip()
                content = content.removeprefix("```json").removesuffix("```").strip()
                structured_info = (
                    json.loads(content) if isinstance(content, str) else content
                )

                rollout = self._create_rollout(
                    hardcoded_info=hardcoded_infos[valid_indices[i]],
                    structured_info=structured_info,
                    component_kwargs=component_kwargs,
                )
                rollouts.append(rollout)

            except Exception as e:
                print(f"Error parsing response: {e}")
                print(traceback.format_exc())
                error_dict = {
                    "path": hardcoded_infos[valid_indices[i]]["rollout"]["path"],
                    "error_pattern": "extraction_failure",
                    "error": traceback.format_exc(),
                }
                rollouts.append(error_dict)
        return rollouts


class RandomInformationExtractor(InformationExtractor):
    """
    Creates a random information extractor that extracts random information from the episode.
    The created object is filled with random string/ints/floats/bools/datetimes/lists etc.

    This should only be used for testing purposes.
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
        model_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> Rollout:
        robot_kwargs = robot_kwargs or {}
        environment_kwargs = environment_kwargs or {}
        task_kwargs = task_kwargs or {}
        model_kwargs = model_kwargs or {}

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
