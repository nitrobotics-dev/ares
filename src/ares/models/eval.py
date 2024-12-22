import json
import os
import traceback
import typing as t

import numpy as np
from litellm import completion_cost
from pydantic import BaseModel, Field

from ares.configs.base import pydantic_to_field_instructions
from ares.image_utils import choose_and_preprocess_frames, split_video_to_frames
from ares.models.base import VLM
from ares.task_utils import PI_DEMO_PATH, PI_DEMO_TASKS


class RolloutDescription(BaseModel):
    task_success_constraints: str = Field(
        description="""
    Generate a list of constraints that must be met for the TASKto be successful based on the first image. 
    Be very specific -- you should be able to judge the success of the following frames based on these constraints.
    """.strip()
    )
    description: str = Field(
        # max_length=1000,
        description="""
        A detailed description of the robot's actions over the course of the provided images.
        Don't include fluff like 'Let's describe...'. Just describe the episode. Make this very detailed and extensive. Roughly 1000 tokens.
        """.strip(),
    )
    robot_setup: t.Literal["one arm", "two arms"]
    environment: t.Literal["floor", "table", "other"]
    lighting_conditions: t.Literal["normal", "dim", "bright"]
    # task: str = Field(max_length=50, description="Short task description")
    success_str: str = Field(
        # max_length=1000,
        description="""
    A detailed description of whether or not the robot successfully completes the task. Be very specific and critical about whether or not the robot has met the intended
    goal state of the task and include lots of details pertaining to partial success. In order to be successful, the robot must have completed the task in a way that is
    consistent with the task description. Any error or deviation from the task description is a failure. Roughly 100 tokens.
    """.strip(),
    )
    success_score: float = Field(
        description="""
    A float score between 0 and 1, representing the success of the task.
    A score of 0 means the task was not completed at all, and a score of
    1 means the task was absolutely completed.
    """.strip()
    )


# Build instruction string dynamically from model fields
field_instructions = pydantic_to_field_instructions(RolloutDescription)

# Build instructions string, will go into prompt jinja2 template
instructions = """
Look at the images provided and consider the following task description:
TASK: {task}

Create a response to the task by answering the following questions:
{field_instructions}

Make sure to ground any descriptions in the images they came from!
You can do this by specifying image 0, 1, 2, etc.
Do not hallucinate any details that cannot be grounded in a certain image!
""".strip()

# Build example response dict dynamically from model fields
response_format_instructions = """
Respond with a python dict that fulfills the above specifications.
The python dict should be able to be loaded with JSON, and then instantiated into the RolloutDescription object, e.g. RolloutDescription(**json.loads(output_dict)).
""".strip()


def get_frames(
    task: str, success: str, n_frames: t.Optional[int] = None
) -> list[np.ndarray]:
    video_path = os.path.join(
        PI_DEMO_PATH, f"{PI_DEMO_TASKS[task]['filename_prefix']}_{success}.mp4"
    )
    if not os.path.exists(video_path):
        raise OSError
    all_frames = split_video_to_frames(video_path)
    print(f"split video into {len(all_frames)} frames")
    specified_frames: list[int] | None = None
    frames = choose_and_preprocess_frames(
        all_frames,
        n_frames if n_frames else len(all_frames),
        specified_frames=specified_frames,
        resize=(512, 512),
    )
    return frames


if __name__ == "__main__":
    # tasks = PI_DEMO_TASKS.keys()
    task = "Paper towel in holder"

    # provider = "gemini"
    # name = f"{provider}/gemini-1.5-flash"

    provider = "openai"
    name = f"{provider}/gpt-4o"
    # name = f"{provider}/gpt-4o-mini"
    # name = f"{provider}/gpt-4-turbo"

    # provider = "anthropic"
    # name = f"{provider}/claude-3-5-sonnet-20240620"

    vlm = VLM(provider=provider, name=name)

    # n_frames = [1, 5, 10, 20]
    n_frames = [3]
    success_flags = ["success"]  # , "fail"]
    # success_flags = ["fail"]
    prediction_tracker = []
    label_tracker = []

    model_kwargs = dict(response_format=RolloutDescription, n=5)

    # for task
    # for provider
    for n_frame in n_frames:
        for success_flag in success_flags:
            info_dict = {
                "instructions": instructions.format(
                    task=PI_DEMO_TASKS[task]["task"],
                    field_instructions=chr(10).join(field_instructions),
                ),
                "response_format": response_format_instructions,
            }
            try:
                frames = get_frames(task, success_flag, n_frames=n_frame)
            except OSError:
                print(f"skipping {n_frame}/{success_flag}/{task}")
                continue
            messages, res = vlm.ask(
                info_dict,
                prompt_filename="test_prompt.jinja2",
                images=frames,
                double_prompt=True,
                model_kwargs=model_kwargs,
            )
            print(f"got answer; cost {completion_cost(res)}")
            content = res.choices[0].message.content
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

            try:
                # Parse the JSON string from the code block
                # prediction = json.loads(code_str)
                # code_str = code_str.replace("'", '"')
                # prediction = ast.literal_eval(code_str)
                description = RolloutDescription(**json.loads(content))
                # RolloutDescription(**json.loads((res.choices[i].message.content))
                prediction_tracker.append(description.success_score)
            except Exception as e:
                print("Failed to parse JSON from response")
                print(traceback.format_exc())
                print(f"Original error: {e}")
                print(f"Code string that failed to parse: {content}")
                breakpoint()
                continue
            label_tracker.append(1 if success_flag == "success" else 0)
            breakpoint()
    breakpoint()
    prediction_tracker = np.array(prediction_tracker)
    label_tracker = np.array(label_tracker)
    print(prediction_tracker)
    print(label_tracker)
