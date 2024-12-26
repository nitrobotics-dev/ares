import json
import os
import traceback
import typing as t

import cv2
import numpy as np
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


if __name__ == "__main__":
    dataset_name = "pi_demos"

    # fps = 1
    fps = 0.5
    # fps = 0.25

    # tasks = PI_DEMO_TASKS.keys()
    task = "Paper towel in holder"

    from ares.models.shortcuts import (  # get_gpt_o1_mini,
        get_claude_3_5_sonnet,
        get_gemini_2_flash,
        get_gemini_15_pro,
        get_gpt_4o,
        get_gpt_4o_mini,
    )

    # vlm = get_gpt_4o_mini()
    vlm = get_gemini_2_flash()
    # vlm = get_claude_3_5_sonnet()
    vlm = get_gpt_4o()
    # vlm = get_gemini_15_pro()
    # n_frames = [1, 5, 10, 20]
    # vlm = get_gpt_o1_mini()
    # vlm = get_gpt_4o_mini()
    # n_frames = [5]
    # success_flags = ["success", "fail"]
    # success_flags = ["fail"]
    prediction_tracker = []
    label_tracker = []
    # for provider
    output_format = """
    - description: str. A thorough description of everything that happens in the video. Focus on the robot's actions and how it performs the task.
    - analysis: str. A detailed analysis determining which success_criteria were met and which were not.
    - performance: float between 0 and 1, where 1 is the robot successfully performed the task and 0 is the robot did not successfully perform the task. Performance should never be 0.5 as that is the threshold for a pass/fail.
    """.strip()

    for n_frame in n_frames:
        for success_flag in success_flags:
            fname = f"{PI_DEMO_TASKS[task]['filename_prefix']}_{success_flag}.mp4"
            frames, frame_indices = load_video_frames(
                dataset_name, fname, target_fps=fps
            )

            # generate success constraints
            messages, res = vlm.ask(
                info=dict(
                    task=PI_DEMO_TASKS[task]["task"],
                ),
                prompt_filename="success_constraint_generation.jinja2",
                images=[frames[0]],
            )
            success_constraints_str = res.choices[0].message.content
            print(f"success constraints: {success_constraints_str}\n")

            # evaluate the task given success constraints
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
            votes = []
            try:
                for choice in res.choices:
                    content = choice.message.content
                    content = (
                        content.strip()
                        .removeprefix("```json")
                        .removesuffix("```")
                        .strip()
                    )
                    structured_info = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                    votes.append(structured_info["performance"])

                print(f"SUCCESS FLAG: {success_flag.upper()}")
                print(f"VOTES: {votes}")
                print(f"MEAN: {np.mean(votes)}; MEDIAN: {np.median(votes)}")

            # for k, v in structured_info.items():
            #     print(f"\n - {k}: {v}")

            except Exception as e:
                print(f"Failed to parse JSON from response: {e}")
                print(f"Response: {res.choices[0].message.content}")
                breakpoint()

            fname = f"/tmp/eval_output/{task}_{success_flag}_{fps}.mp4".replace(
                " ", "_"
            )
            save_frames_to_mp4(frames, fname)
            print(f"saved frames to {fname}")

            breakpoint()
    breakpoint()
    prediction_tracker = np.array(prediction_tracker)
    label_tracker = np.array(label_tracker)
    print(prediction_tracker)
    print(label_tracker)
