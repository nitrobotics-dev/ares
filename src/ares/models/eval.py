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

from ares.configs.base import pydantic_to_field_instructions
from ares.image_utils import load_video_frames
from ares.models.base import VLM
from ares.task_utils import PI_DEMO_PATH, PI_DEMO_TASKS

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


def easy_get_frames(task: str, success_flag: str, fps: int) -> list[np.ndarray]:
    fname = f"{PI_DEMO_TASKS[task]['filename_prefix']}_{success_flag}.mp4"
    frames, frame_indices = load_video_frames(dataset_name, fname, target_fps=fps)

    MAX_N_FRAMES = 35  # HACK: fix when higher tier TPM limits
    if len(frames) > MAX_N_FRAMES:
        print(f"received {len(frames)} frames; downsampling to 40 frames")
        middle_indices = np.linspace(1, len(frames) - 2, MAX_N_FRAMES - 2, dtype=int)
        frames = [frames[0]] + [frames[i] for i in middle_indices] + [frames[-1]]
    return frames


def dynamic_constraint_generation(vlm: VLM, task: str, frames: list[np.ndarray]) -> str:
    # generate success constraints
    messages, res = vlm.ask(
        info=dict(
            task=PI_DEMO_TASKS[task]["task"],
        ),
        prompt_filename="success_constraint_generation.jinja2",
        images=[frames[0]],
    )
    success_constraints_str = res.choices[0].message.content
    return success_constraints_str


def simple_single_eval(
    vlm: VLM, task: str, frames: list[np.ndarray], success_constraints_str: str
) -> dict:
    messages, res = vlm.ask(
        info=dict(
            task=PI_DEMO_TASKS[task]["task"],
            output_format=output_format,
            success_constraints=success_constraints_str,
        ),
        prompt_filename="simple_eval.jinja2",
        images=frames,
        model_kwargs=dict(n=5),
    )
    print(f"got answer; cost {completion_cost(res)}")
    outputs = defaultdict(list)
    try:
        for choice in res.choices:
            content = choice.message.content
            content = (
                content.strip().removeprefix("```json").removesuffix("```").strip()
            )
            structured_info = (
                json.loads(content) if isinstance(content, str) else content
            )
            for k, v in structured_info.items():
                outputs[k].append(v)

        votes = outputs["performance"]
        print(f"SUCCESS FLAG: {success_flag.upper()}")
        print(f"VOTES: {votes}")
        print(f"MEAN: {np.mean(votes)}; MEDIAN: {np.median(votes)}")

    except Exception as e:
        print(f"Failed to parse JSON from response: {e}")
        print(f"Response: {res.choices[0].message.content}")
        breakpoint()

    return outputs


if __name__ == "__main__":
    dataset_name = "pi_demos"

    # fps = 1
    # fps = 0.5
    # fps = 0.25
    fps_options = [1]

    tasks = [
        # tasks = PI_DEMO_TASKS.keys()
        # task = "Paper towel in holder"
        # task = "Toast out of toaster"
        # "Laundry fold (shirts)"
        "Laundry fold (shorts)"
    ]

    from ares.models.shortcuts import (
        get_claude_3_5_sonnet,
        get_gemini_2_flash,
        get_gemini_15_pro,
        get_gpt_4o,
        get_gpt_4o_mini,
        get_gpt_o1_mini,
    )

    vlm_options = [
        # vlm = get_gpt_4o_mini()
        # vlm = get_gemini_2_flash()
        # vlm = get_claude_3_5_sonnet()
        get_gpt_4o(),
        # vlm = get_gemini_15_pro()
        # n_frames = [1, 5, 10, 20]
        # vlm = get_gpt_o1_mini()
        # vlm = get_gpt_4o_mini()
    ]
    success_flags = ["success", "fail"]
    # success_flags = ["fail"]

    # for provider
    output_format = """
    - description: str. A thorough description of everything that happens in the video. Focus on the robot's actions and how it performs the task.
    - analysis: str. A detailed analysis determining which success_criteria were met and which were not.
    - performance: float between 0 and 1, where 1 is the robot successfully performed the task and 0 is the robot did not successfully perform the task. Performance should never be 0.5 as that is the threshold for a pass/fail.
    """.strip()

    prediction_rows = []

    for task in tasks:
        for vlm in vlm_options:
            for fps in fps_options:
                for success_flag in success_flags:
                    frames = easy_get_frames(task, success_flag, fps)
                    success_constraints_str = dynamic_constraint_generation(
                        vlm, task, frames
                    )
                    print(f"success constraints: {success_constraints_str}\n")
                    # evaluate the task given success constraints
                    outputs = simple_single_eval(
                        vlm, task, frames, success_constraints_str
                    )
                    prediction_rows.append(
                        dict(
                            vlm=vlm,
                            task=task,
                            success_flag=success_flag,
                            fps=fps,
                            performance=outputs["performance"],
                            mean_performance=np.mean(outputs["performance"]),
                            median_performance=np.median(outputs["performance"]),
                        )
                    )

                    fname = f"/tmp/eval_output/{task}_{success_flag}_{fps}.mp4".replace(
                        " ", "_"
                    )
                    save_frames_to_mp4(frames, fname)
                    print(f"saved frames to {fname}")

                    breakpoint()

    df = pd.DataFrame(prediction_rows)
    df.to_csv(
        f"/tmp/eval_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        index=False,
    )
