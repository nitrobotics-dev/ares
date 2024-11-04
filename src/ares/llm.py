import base64
import os
import litellm
from litellm import completion, acompletion, completion_cost
from litellm.utils import check_valid_key, get_valid_models
import asyncio
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from PIL import Image
import numpy as np
import typing as t
import io
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import cv2
from pydantic import BaseModel, Field


from ares.image_utils import encode_image, split_video_to_frames, choose_and_preprocess_frames

class LLM:
    def __init__(self, provider: str, llm: str, llm_kwargs: dict = {}):
        self.provider = provider
        self.llm = llm
        self.llm_kwargs = llm_kwargs
        # self.check_valid_key()

    def check_valid_key(self):
        # note: don't use litellm util here, it uses 10 tokens!
        try:
            res = completion(
                model=self.llm,
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
    ) -> list[dict]:
        content = []
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
        return [{"role": "user", "content": content}]

    def ask(
        self,
        prompt_filename: str,
        info: dict,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
    ) -> t.Tuple[t.Any, str]:
        messages = self._construct_messages(prompt_filename, info, images)
        return messages, completion(
            model=self.llm, messages=messages, **self.llm_kwargs
        )

class GeminiVideoLLM(LLM):
    def __init__(self, provider: str, llm: str, llm_kwargs: dict = {}):
        super().__init__(provider, llm, llm_kwargs)
        vertexai.init(
            project=os.environ["VERTEX_PROJECT"], location=os.environ["VERTEX_LOCATION"]
        )
        self.model = GenerativeModel(llm)
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

    def ask(
        self,
        prompt_filename: str,
        prompt: str,
        info: dict,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
    ) -> str:
        if "https://" in video_path:
            video1 = Part.from_uri(
                mime_type="video/mp4",
                uri=video_path,
            )
        else:    
            encoded_video = base64.b64encode(open(video_path, "rb").read()).decode("utf-8")
            video1 = Part.from_data(
                data=base64.b64decode(encoded_video), mime_type="video/mp4"
            )
        responses = self.model.generate_content(
            [prompt, video1],
            generation_config=self.generation_config,
        )
        breakpoint()

class TrajectoryDescription(BaseModel):
    robot_setup: t.Literal["one arm", "two arms"]
    environment: t.Literal["floor", "table", "other"]
    lighting_conditions: t.Literal["normal", "dim", "bright"]
    # task: str = Field(max_length=50, description="Short task description")
    description: str = Field(max_length=1000, description="A detailed description of the robot's actions over the course of the images.")  # Detailed task description
    success: str = Field(max_length=1000, description="""
    A detailed description of whether or not the robot successfully completes the task. 
    Be very specific and critical about whether or not the robot has met the intended goal state of the task and include lots of details pertaining to partial success.
    """.strip())

    @classmethod
    def to_field_instructions(cls):
        field_instructions = []
        for field_name, field in cls.model_fields.items():
            field_instructions.append(f"    - {field_name}: {str(field)}")
        return field_instructions

    @classmethod
    def to_example_dict(cls):
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

    # task = "Paper towel in holder"
    # task = "Eggs in carton"
    task = "Grocery Bagging"

    # video_path = "https://dnrjl01ydafck.cloudfront.net/v2/upload/processed_towel_fail.mp4"
    # video_path = os.path.join(PI_DEMO_PATH, "processed_towel_fail.mp4")   
    # video_path = os.path.join(PI_DEMO_PATH, "processed_eggs_fail.mp4")
    # video_path = os.path.join(PI_DEMO_PATH, "processed_eggs_success.mp4")
    # video_path = os.path.join(PI_DEMO_PATH, f"processed_grocery_bagging_fail.mp4")
    video_path = os.path.join(PI_DEMO_PATH, f"processed_grocery_bagging_success.mp4")



    n_frames = 10
    all_frames = split_video_to_frames(video_path)
    specified_frames = None
    frames = choose_and_preprocess_frames(all_frames, n_frames, specified_frames=specified_frames, resize=(224, 224))

    # provider = "gemini"
    # llm = f"{provider}/gemini-1.5-flash"

    provider = "openai"
    llm = f"{provider}/gpt-4o"
    # llm = f"{provider}/gpt-4o-mini"
    # llm = f"{provider}/gpt-4-turbo"

    # vlm = GeminiVideoLLM("gemini", "gemini-1.5-flash", dict())
    llm = LLM(provider=provider, llm=llm)


    # Build instruction string dynamically from model fields
    field_instructions = TrajectoryDescription.to_field_instructions()


    # Build instructions string, will go into prompt jinja2 template
    instructions = f"""
    Look at the images provided and consider the following task description:
    TASK: {PI_DEMO_TASKS[task]}

    Create a response to the task by answering the following questions:
    {chr(10).join(field_instructions)}
    """.strip()

    # Build example response dict dynamically from model fields
    response_format = f"""
    Respond with a python dict, e.g. {TrajectoryDescription.to_example_dict()}. Respond with JUST the necessary text.
    """.strip()

    info_dict = {
        "instructions": instructions,
        "response_format": response_format,
    }

    messages, res = llm.ask(
        "test_prompt.jinja2",
        info_dict,
        images=frames,
    )

    breakpoint()
    print(res.choices[0].message.content)
