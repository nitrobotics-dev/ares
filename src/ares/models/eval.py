import asyncio
import json
import os
import traceback
import typing as t
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from litellm import completion_cost
from pydantic import BaseModel, Field
from tqdm import tqdm

from ares.configs.base import pydantic_to_field_instructions
from ares.models.base import VLM
from ares.utils.image_utils import load_video_frames
from ares.utils.task_utils import PI_DEMO_PATH, PI_DEMO_TASKS

IMAGE_TILE_SIZE = (512, 512)


def save_frames_to_mp4(frames: list[np.ndarray], fname: str) -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    video_writer = cv2.VideoWriter(
        fname,
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def easy_get_frames(
    dataset_filename: str, task: str, success_flag: str, fps: int | float
) -> list[np.ndarray]:
    # small hack -- if FPS is 0, just access first and last frame!
    fname = f"{PI_DEMO_TASKS[task]['filename_prefix']}_{success_flag}.mp4"
    frames, frame_indices = load_video_frames(
        dataset_filename, fname, target_fps=fps if fps != 0 else 1
    )
    if fps == 0:
        frames = [frames[0], frames[-1]]

    MAX_N_FRAMES = 40  # HACK: fix when higher tier TPM limits
    if len(frames) > MAX_N_FRAMES:
        print(
            f"received {len(frames)} frames; downsampling to 40 frames. do you still need me?"
        )
        middle_indices = np.linspace(1, len(frames) - 2, MAX_N_FRAMES - 2, dtype=int)
        frames = [frames[0]] + [frames[i] for i in middle_indices] + [frames[-1]]
    return frames


async def dynamic_constraint_generation_async(
    vlm: VLM, task: str, frames: list[np.ndarray]
) -> str:
    messages, res = await vlm.ask_async(
        info=dict(
            task=PI_DEMO_TASKS[task]["task"],
        ),
        prompt_filename="success_constraint_generation.jinja2",
        images=[frames[0]],
    )
    return res.choices[0].message.content


def parse_content(choice: dict) -> dict:
    content = choice.message.content
    content = content.strip().removeprefix("```json").removesuffix("```").strip()
    structured_info = json.loads(content) if isinstance(content, str) else content
    return structured_info


async def simple_single_video_eval_async(
    vlm: VLM,
    task: str,
    frames: list[np.ndarray],
    success_constraints_str: str,
    output_format: str,
) -> dict:
    messages, res = await vlm.ask_async(
        info=dict(
            task=PI_DEMO_TASKS[task]["task"],
            output_format=output_format,
            success_constraints=success_constraints_str,
        ),
        prompt_filename="simple_video_eval.jinja2",
        images=frames,
        model_kwargs=dict(n=5) if "claude" not in vlm.name else None,
    )
    return parse_responses(res)


async def simple_single_frame_description_eval_async(
    vlm: VLM,
    task: str,
    frames: list[np.ndarray],
    success_constraints_str: str,
    output_format: str,
) -> dict:
    # Get descriptions for all frames concurrently (rate limiting handled by VLM)
    async def get_frame_description(frame: np.ndarray, frame_idx: int) -> str:
        messages, res = await vlm.ask_async(
            info=dict(task=task, success_constraints=success_constraints_str),
            prompt_filename="task_frame_description.jinja2",
            images=[frame],
        )
        return f"Frame {frame_idx}: {res.choices[0].message.content}"

    descriptions = await asyncio.gather(
        *[get_frame_description(frame, i) for i, frame in enumerate(frames)]
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
    return parse_responses(res)


def parse_responses(res: t.Any) -> dict:
    """Helper to parse responses and collect outputs"""
    print(f"got answer; cost {completion_cost(res)}")
    outputs = defaultdict(list)
    for i, choice in enumerate(res.choices):
        try:
            structured_info = parse_content(choice)
            for k, v in structured_info.items():
                outputs[k].append(v)
            if i == 0:
                for k, v in structured_info.items():
                    print(f"- {k}: {v}")
        except Exception as e:
            print(f"Failed to parse JSON from response: {e}")
            print(f"Response: {choice.message.content}")
    return outputs


async def process_task_async(
    vlm: VLM,
    dataset_filename: str,
    task: str,
    fps: float,
    success_flag: str,
    method: str,
) -> dict:
    try:
        frames = easy_get_frames(dataset_filename, task, success_flag, fps)
        success_constraints_str = await dynamic_constraint_generation_async(
            vlm, task, frames
        )
        print(f"success constraints: {success_constraints_str}\n")

        if method == "video":
            outputs = await simple_single_video_eval_async(
                vlm, task, frames, success_constraints_str, output_format
            )
        elif method == "frame_descriptions":
            outputs = await simple_single_frame_description_eval_async(
                vlm, task, frames, success_constraints_str, output_format
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        votes = [x if x is not None else np.nan for x in outputs.get("performance", [])]
        return dict(
            vlm=vlm.name,
            task=task,
            success_flag=success_flag,
            fps=fps,
            performance=votes,
            mean_performance=(np.nanmean(votes) if any(votes) else np.nan),
            median_performance=(np.nanmedian(votes) if any(votes) else np.nan),
            method=method,
        )
    except Exception as e:
        print(f"Error processing task {task}: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    dataset_filename = "pi_demos"

    # fps_options = [1]
    fps_options = [0, 0.25, 0.5, 1, 2]
    # fps_options = [2]
    # fps_options = [0]  # 2]

    tasks = [
        "Eggs in carton",
        "Grocery Bagging",
        "Toast out of toaster",
        "Towel fold",
        "Stack bowls",
        "Tupperware in microwave",
        "Laundry fold (shirts)",
        "Laundry fold (shorts)",
        "Paper towel in holder",
        "Food in to go box",
    ]

    from ares.models.shortcuts import (
        get_claude_3_5_sonnet,
        get_gemini_2_flash,
        get_gemini_15_flash,
        get_gemini_15_pro,
        get_gpt_4o,
        get_gpt_4o_mini,
        get_gpt_o1_mini,
    )

    vlm_options = [
        # get_claude_3_5_sonnet(),
        # get_gpt_4o(),
        get_gpt_4o_mini(),
        # get_gemini_15_pro(),
        # n_frames = [1, 5, 10, 20]
        # get_gpt_o1_mini(),
        # vlm = get_gpt_4o_mini()
        # get_gemini_2_flash(),
        # get_gemini_15_flash(),
    ]
    success_flags = ["success", "fail"]
    # methods = ["frame_descriptions", "video"]
    # methods = ["video"]
    methods = ["frame_descriptions"]
    # for provider
    output_format = """
    - description: str. A thorough description of everything that happens in the sequence of images. Focus on the robot's actions and how it performs the task.
    - analysis: str. A detailed analysis determining which success_criteria were met and which were not.
    - performance: float between 0 and 1, where 1 is the robot successfully performed the task and 0 is the robot did not successfully perform the task. Performance should never be 0.5 as that is the threshold for a pass/fail.
    """.strip()

    prediction_rows = []

    async def main() -> list[dict]:
        all_results = []

        # Process each method sequentially to avoid concurrent calls to same VLM
        for method in methods:
            # Process all VLMs in parallel for this method
            vlm_tasks = []
            for vlm in vlm_options:
                tasks_to_process = []
                for task in tasks:
                    for fps in fps_options:
                        for success_flag in success_flags:
                            tasks_to_process.append(
                                process_task_async(
                                    vlm,
                                    dataset_filename,
                                    task,
                                    fps,
                                    success_flag,
                                    method,
                                )
                            )

                # Gather all tasks for this specific VLM
                results = await asyncio.gather(
                    *tasks_to_process, return_exceptions=True
                )
                results = [r for r in results if r is not None]

                # Save results for this VLM
                if results:
                    path = f"/workspaces/ares/data/eval_dump/eval_results_{vlm.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{method}.csv"
                    df = pd.DataFrame(results)
                    df.to_csv(path, index=False)
                    print(f"saved results to {path}")
                all_results.extend(results)

        return all_results

    all_results = asyncio.run(main())
    print(f"found {len(all_results)} results")
    breakpoint()
