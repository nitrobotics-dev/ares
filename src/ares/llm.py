import base64
import os
import typing as t

import numpy as np
import vertexai
from jinja2 import Environment, FileSystemLoader
from litellm import completion, completion_cost
from litellm.utils import ModelResponse
from PIL import Image
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerativeModel, Part

from ares.image_utils import (
    choose_and_preprocess_frames,
    encode_image,
    split_video_to_frames,
)


class LLM:
    def __init__(self, provider: str, llm_name: str, llm_kwargs: dict = {}):
        self.provider = provider
        self.llm_name = llm_name
        self.llm_kwargs = llm_kwargs
        # self.check_valid_key()

    def check_valid_key(self) -> bool:
        # note: don't use litellm util here, it uses 10 tokens!
        try:
            res = completion(
                model=self.llm_name,
                messages=[{"role": "user", "content": "!"}],
                max_tokens=1,
            )
        except Exception as e:
            print(f"Error checking valid key: {e}")
            return False
        return True

    def _get_prompt(self, prompt_filename: str, info: dict) -> str:
        # Set up Jinja environment to load templates from src/ares directory
        jinja_env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        template = jinja_env.get_template(prompt_filename)
        # Render template with information
        return template.render(**info)

    def _construct_messages(
        self,
        prompt_filename: str,
        info: dict,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        double_prompt: bool = False,
    ) -> list[dict[str, t.Any]]:
        content: list[dict[str, t.Any]] = []
        prompt = self._get_prompt(prompt_filename, info)
        content.append({"type": "text", "text": prompt})
        if images:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(images[i])}"
                    },
                }
                for i in range(len(images))
            ]
            content.extend(image_contents)
        if double_prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def ask(
        self,
        prompt_filename: str,
        info: dict,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
    ) -> t.Tuple[t.Any, ModelResponse]:
        if video_path:
            raise NotImplementedError("Video path not implemented for this LLM")
        messages = self._construct_messages(
            prompt_filename, info, images, double_prompt
        )
        return messages, completion(
            model=self.llm_name, messages=messages, **self.llm_kwargs
        )


class GeminiVideoLLM(LLM):
    def __init__(self, provider: str, llm_name: str, llm_kwargs: dict = {}):
        super().__init__(provider, llm_name, llm_kwargs)
        vertexai.init(
            project=os.environ["VERTEX_PROJECT"], location=os.environ["VERTEX_LOCATION"]
        )
        self.model = GenerativeModel(llm_name)
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

    def ask(
        self,
        prompt_filename: str,
        info: dict,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
    ) -> t.Tuple[t.Any, ModelResponse]:
        prompt = self._get_prompt(prompt_filename, info)
        if video_path:
            if "https://" in video_path:
                video = Part.from_uri(
                    mime_type="video/mp4",
                    uri=video_path,
                )
            else:
                encoded_video = base64.b64encode(open(video_path, "rb").read()).decode(
                    "utf-8"
                )
                video = Part.from_data(
                    data=base64.b64decode(encoded_video), mime_type="video/mp4"
                )
        messages = [prompt, video]
        if double_prompt:
            messages.append(prompt)
        responses = self.model.generate_content(
            messages,
            generation_config=self.generation_config,
        )
        return messages, responses


class RolloutDescription(BaseModel):
    robot_setup: t.Literal["one arm", "two arms"]
    environment: t.Literal["floor", "table", "other"]
    lighting_conditions: t.Literal["normal", "dim", "bright"]
    # task: str = Field(max_length=50, description="Short task description")
    description: str = Field(
        max_length=1000,
        description="A detailed description of the robot's actions over the course of the images.",
    )
    success_str: str = Field(
        max_length=1000,
        description="""
    A detailed description of whether or not the robot successfully completes the task. 
    Be very specific and critical about whether or not the robot has met the intended goal state of the task and include lots of details pertaining to partial success.
    In order to be successful, the robot must have completed the task in a way that is consistent with the task description. Any error or deviation from the task description is a failure.
    """.strip(),
    )
    success_score: float = Field(
        description="A float score between 0 and 1, representing the success of the task. A score of 0 means the task was not completed at all, and a score of 1 means the task was completed absolutely perfectly.",
    )

    @classmethod
    def to_field_instructions(cls) -> list[str]:
        field_instructions = []
        for field_name, field in cls.model_fields.items():
            field_instructions.append(f"    - {field_name}: {str(field)}")
        return field_instructions

    @classmethod
    def to_example_dict(cls) -> dict:
        example_dict = {}
        traj_items = list(cls.model_fields.items())
        for field_name, field in [traj_items[0], traj_items[-1]]:
            if hasattr(field.annotation, "__args__"):  # For Literal types
                example_dict[field_name] = field.annotation.__args__[0]
            else:
                example_dict[field_name] = "..."
        return example_dict


if __name__ == "__main__":
    import math

    from ares.task_utils import PI_DEMO_PATH, PI_DEMO_TASKS

    # os.environ["LITELLM_LOG"] = "DEBUG"
    # litellm.set_verbose=True
    # task = "Eggs in carton"
    # task = "Grocery Bagging"
    # task = "Toast out of toaster"
    # task = "Towel fold"
    # task = "Stack bowls"
    # task = "Tupperware in microwave"
    # task = "Items in drawer"
    # task = "Laundry fold (shirts)"
    # task = "Laundry fold (shorts)"
    # task = "Paper towel in holder"
    task = "Food in to go box"
    # success = "fail"
    success = "success"

    video_path = os.path.join(
        PI_DEMO_PATH, f"{PI_DEMO_TASKS[task]['filename_prefix']}_{success}.mp4"
    )
    n_frames = 10
    all_frames = split_video_to_frames(video_path)
    print(f"split video into {len(all_frames)} frames")
    specified_frames: list[int] | None = None
    frames = choose_and_preprocess_frames(
        all_frames, n_frames, specified_frames=specified_frames, resize=(512, 512)
    )

    # provider = "gemini"
    # llm_name = f"{provider}/gemini-1.5-flash"

    provider = "openai"
    llm_name = f"{provider}/gpt-4o"
    # llm_name = f"{provider}/gpt-4o-mini"
    # llm_name = f"{provider}/gpt-4-turbo"

    # vlm = GeminiVideoLLM("gemini", "gemini-1.5-flash", dict())
    llm = LLM(provider=provider, llm_name=llm_name)

    # Build instruction string dynamically from model fields
    field_instructions = RolloutDescription.to_field_instructions()

    # Build instructions string, will go into prompt jinja2 template
    instructions = f"""
    Look at the images provided and consider the following task description:
    TASK: {PI_DEMO_TASKS[task]}

    Create a response to the task by answering the following questions:
    {chr(10).join(field_instructions)}
    """.strip()

    # Build example response dict dynamically from model fields
    response_format = f"""
    For the response, first respond with about 500 words that describe the entire video, focusing on the robot's actions and the task.
    Then, respond with a python dict, e.g. {RolloutDescription.to_example_dict()} that fulfills the above specifications.
    """.strip()

    info_dict = {
        "instructions": instructions,
        "response_format": response_format,
    }
    breakpoint()

    messages, res = llm.ask(
        "test_prompt.jinja2",
        info_dict,
        images=frames,
        double_prompt=True,
    )

    breakpoint()
    print(res.choices[0].message.content, completion_cost(res))
