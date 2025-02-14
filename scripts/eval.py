""" 
Script to evaluate the performance of leading API models on the Physical Intelligence Demos dataset. See `notebooks/eval_nb.ipynb` for more details.
We generate a success constraint for the task based on the first frame and then evaluate the performance of the model at the task.
We evaluate two methods: 
- `video`: analyze the stream of frames all at once, aka [image 1, image 2, image 3, ...]
- `frame_descriptions`: generate a description for each frame, then process only text descriptions. [image 1] -> description 1, [image 2] -> description 2, ... and then process [description 1, description 2, ...]

Additionally, we evaluate different VLMs (openai, anthropic, google, etc.) and different FPS (0, 0.25, 0.5, 1, 2), as well as mean/median of vote performance. Note that 0 FPS means only the first and last frames are used.

All functions are implemented asynchronously to allow for parallel processing of tasks, with each VLM maintaining its own semaphore to avoid rate limiting.
"""

import asyncio
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import Field

from ares.configs.base import BaseConfig, Rollout, pydantic_to_field_instructions
from ares.constants import ARES_DATA_DIR, get_dataset_info_by_key
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
    setup_rollouts,
)
from ares.models.base import VLM, parse_response, parse_responses
from ares.utils.image_utils import load_video_frames

IMAGE_TILE_SIZE = (512, 512)
MAX_N_FRAMES = 40


# define an EvalConfig pydantic model to parse the output of the API models, define the input types, and output format
class EvalConfig(BaseConfig):
    description: str = Field(
        description="A thorough description of everything that happens in the sequence of images. Focus on the robot's actions and how it performs the task."
    )
    analysis: str = Field(
        description="A detailed analysis determining which success_criteria were met and which were not."
    )
    performance: float = Field(
        description="A float between 0 and 1, where 1 is the robot successfully performed the task and 0 is the robot did not successfully perform the task. Performance should never be 0.5 as that is the threshold for a pass/fail.",
        ge=0,
        le=1,
    )


async def dynamic_constraint_generation_async(
    vlm: VLM, task: str, frames: list[np.ndarray]
) -> str:
    """
    Helper function for generating success constraints for a task based on the first frame and the task description.
    """
    messages, res = await vlm.ask_async(
        info=dict(task=task),
        prompt_filename="success_constraint_generation.jinja2",
        images=[frames[0]],
    )
    return parse_response(res.choices[0], load_json=False)


async def get_frame_description(
    vlm: VLM,
    task: str,
    frame: np.ndarray,
    frame_idx: int,
    success_constraints_str: str,
) -> str:
    messages, res = await vlm.ask_async(
        info=dict(task=task, success_constraints=success_constraints_str),
        prompt_filename="task_frame_description.jinja2",
        images=[frame],
    )
    return f"Frame {frame_idx}: {res.choices[0].message.content}"


async def simple_single_video_eval_async(
    vlm: VLM,
    task: str,
    frames: list[np.ndarray],
    success_constraints_str: str,
    output_format: str,
) -> list[dict | str | None]:
    """
    Helper function for evaluating the robot's performance at a task by analyzing the stream of frames all at once.
    """
    messages, res = await vlm.ask_async(
        info=dict(
            task=task,
            output_format=output_format,
            success_constraints=success_constraints_str,
        ),
        prompt_filename="simple_video_eval.jinja2",
        images=frames,
        model_kwargs=dict(n=5) if "claude" not in vlm.name else None,
    )
    outputs = parse_responses(res, load_json=True)
    return outputs


async def simple_single_frame_description_eval_async(
    vlm: VLM,
    task: str,
    frames: list[np.ndarray],
    success_constraints_str: str,
    output_format: str,
) -> list[dict | str | None]:
    """
    Helper function for describing each frame in the sequence of images and then summarizing the descriptions.
    This yields min(N, MAX_N_FRAMES) descriptions which are then all passed to the VLM to get the performance score.
    """

    descriptions = await asyncio.gather(
        *[
            get_frame_description(vlm, task, frame, i, success_constraints_str)
            for i, frame in enumerate(frames)
        ]
    )
    descriptions_str = "\n".join(descriptions)
    messages, res = await vlm.ask_async(
        info=dict(
            task=task,
            success_constraints=success_constraints_str,
            descriptions=descriptions_str,
            output_format=output_format,
        ),
        prompt_filename="summarization_frame_eval.jinja2",
        model_kwargs=dict(n=5) if "claude" not in vlm.name else None,
    )
    outputs = parse_responses(res, load_json=True)
    return outputs


async def process_task_async(
    vlm: VLM,
    dataset_filename: str,
    rollout: Rollout,
    fps: float,
    method: str,
    output_format: str,
) -> dict | None:
    try:
        frames, frame_indices = load_video_frames(
            dataset_filename, rollout.filename, target_fps=fps
        )
        success_constraints_str = await dynamic_constraint_generation_async(
            vlm, rollout.task.language_instruction, frames
        )
        if method == "video":
            outputs = await simple_single_video_eval_async(
                vlm,
                rollout.task.language_instruction,
                frames,
                success_constraints_str,
                output_format,
            )
        elif method == "frame_descriptions":
            outputs = await simple_single_frame_description_eval_async(
                vlm,
                rollout.task.language_instruction,
                frames,
                success_constraints_str,
                output_format,
            )
        else:
            print(f"Invalid method: {method}")
            return None

        votes = [x["performance"] if isinstance(x, dict) else np.nan for x in outputs]
        return dict(
            vlm=vlm.name,
            task=rollout.task.language_instruction,
            fps=fps,
            performance=votes,
            mean_performance=(np.nanmean(votes) if any(votes) else np.nan),
            median_performance=(np.nanmedian(votes) if any(votes) else np.nan),
            method=method,
            success_flag=rollout.task.success,
        )
    except Exception as e:
        print(f"Error processing task {rollout.task.language_instruction}: {e}")
        print(traceback.format_exc())
        return None


async def main(
    rollouts: list[Rollout],
    dataset_filename: str,
    vlm_options: list[VLM],
    methods: list[str],
    output_format: str,
    fps_options: list[float],
) -> list[dict | None]:
    all_results = []
    for method in methods:
        for vlm in vlm_options:
            tasks_to_process = []
            for rollout in rollouts:
                for fps in fps_options:
                    tasks_to_process.append(
                        process_task_async(
                            vlm,
                            dataset_filename,
                            rollout,
                            fps,
                            method,
                            output_format,
                        )
                    )

            # Gather all tasks for this specific VLM
            results = await asyncio.gather(*tasks_to_process, return_exceptions=True)
            results = [r for r in results if r is not None]

            # Save results for this VLM
            if results:
                path = os.path.join(
                    ARES_DATA_DIR,
                    f"eval_dump/eval_results_{vlm.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{method}.csv",
                )
                df = pd.DataFrame(results)
                df.to_csv(path, index=False)
                print(f"saved results to {path}")
            all_results.extend(results)
    return all_results


if __name__ == "__main__":

    from ares.models.shortcuts import (
        get_claude_3_5_sonnet,
        get_gemini_2_flash,
        get_gemini_15_flash,
        get_gemini_15_pro,
        get_gpt_4o,
        get_gpt_4o_mini,
        get_gpt_o1_mini,
    )

    # FPS of 0 just means first and last frames
    fps_options = [0, 0.25, 0.5, 1, 2]
    dataset_filename = "pi_demos"
    dataset_info = get_dataset_info_by_key("dataset_filename", dataset_filename)
    dataset_formalname = dataset_info["dataset_formalname"]

    # vlm_options = [get_gpt_4o_mini()]
    # vlm_options = [get_gpt_4o()]
    vlm_options = [get_claude_3_5_sonnet()]
    # methods = ["frame_descriptions", "video"]
    methods = ["video"]
    output_format = "\n".join(pydantic_to_field_instructions(EvalConfig))

    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    rollouts = setup_rollouts(engine, dataset_formalname)

    all_results = asyncio.run(
        main(
            rollouts,
            dataset_filename,
            vlm_options,
            methods,
            output_format,
            fps_options,
        )
    )
    print(f"found {len(all_results)} results")
    breakpoint()
