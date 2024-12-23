import json
import os
import traceback
import typing as t

import numpy as np
from litellm import completion_cost
from pydantic import BaseModel, Field

from ares.configs.base import pydantic_to_field_instructions
from ares.image_utils import load_video_frames
from ares.models.base import VLM
from ares.task_utils import PI_DEMO_PATH, PI_DEMO_TASKS

IMAGE_TILE_SIZE = (512, 512)


if __name__ == "__main__":
    dataset_name = "pi_demos"

    # tasks = PI_DEMO_TASKS.keys()
    task = "Paper towel in holder"

    from ares.models.shortcuts import (
        get_claude_3_5_sonnet,
        get_gemini_2_flash,
        get_gpt_4o,
        get_gpt_4o_mini,
    )

    # vlm = get_gpt_4o_mini()
    # vlm = get_gemini_2_flash()
    # vlm = get_claude_3_5_sonnet()
    vlm = get_gpt_4o()
    # n_frames = [1, 5, 10, 20]
    n_frames = [5]
    success_flags = ["success", "fail"]
    # success_flags = ["fail"]
    prediction_tracker = []
    label_tracker = []

    # model_kwargs = dict(n=1)

    # for task
    # for provider
    output_format = """
    - success_criteria: str. A short description of the success criteria of the task. Be very specific.
    - description: str. A thorough description of everything that happens in the video. Focus on the robot's actions and how it performs the task.
    - analysis: str. A detailed analysis determining which success_criteria were met and which were not.
    - performance: float between 0 and 1, where 1 is the robot successfully performed the task and 0 is the robot did not successfully perform the task. Performance should never be 0.5.
    """.strip()

    for n_frame in n_frames:
        for success_flag in success_flags:
            fname = f"{PI_DEMO_TASKS[task]['filename_prefix']}_{success_flag}.mp4"
            frames, frame_indices = load_video_frames(dataset_name, fname, target_fps=1)
            # sample n_frame in order from frames
            # frames = frames[:: len(frames) // n_frame]

            messages, res = vlm.ask(
                info=dict(
                    task=PI_DEMO_TASKS[task]["task"], output_format=output_format
                ),
                prompt_filename="simple_eval.jinja2",
                images=frames,
            )
            # breakpoint()
            print(f"got answer; cost {completion_cost(res)}")
            try:
                content = res.choices[0].message.content
                content = (
                    content.strip().removeprefix("```json").removesuffix("```").strip()
                )
                structured_info = (
                    json.loads(content) if isinstance(content, str) else content
                )
                for k, v in structured_info.items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"Failed to parse JSON from response: {e}")
                print(f"Response: {res.choices[0].message.content}")
                breakpoint()
            # extract ```python .... ``` and json.load it
            # Extract Python code block from response
            # match = re.search(
            #     r"```python\s*(.*?)\s*```", res.choices[0].message.content, re.DOTALL
            # )
            # if not match:
            #     breakpoint()
            #     print("No Python code block found in response")
            #     continue

            # code_str = match.group(1)

            # try:
            #     # Parse the JSON string from the code block
            #     # prediction = json.loads(code_str)
            #     # code_str = code_str.replace("'", '"')
            #     # prediction = ast.literal_eval(code_str)
            #     description = RolloutDescription(**json.loads(content))
            #     # RolloutDescription(**json.loads((res.choices[i].message.content))
            #     prediction_tracker.append(description.success_score)
            # except Exception as e:
            #     print("Failed to parse JSON from response")
            #     print(traceback.format_exc())
            #     print(f"Original error: {e}")
            #     print(f"Code string that failed to parse: {content}")
            #     breakpoint()
            #     continue
            # label_tracker.append(1 if success_flag == "success" else 0)
            breakpoint()
    breakpoint()
    prediction_tracker = np.array(prediction_tracker)
    label_tracker = np.array(label_tracker)
    print(prediction_tracker)
    print(label_tracker)
