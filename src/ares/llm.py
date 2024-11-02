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


def encode_image(image: t.Union[str, np.ndarray, Image.Image]) -> str:
    if isinstance(image, str):  # file path
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, (np.ndarray, Image.Image)):  # numpy array or PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise TypeError(
            "Unsupported image format. Use file path, numpy array, or PIL image."
        )


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

    # async def aask(
    #     self,
    #     prompt_filename: str,
    #     info: dict,
    #     images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
    # ) -> str:
    #     messages = self._construct_messages(prompt_filename, info, images)
    #     return await acompletion(model=self.llm, messages=messages, **self.llm_kwargs)


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

def split_video_to_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


if __name__ == "__main__":
    os.environ["LITELLM_LOG"] = "DEBUG"
    litellm.set_verbose=True

    # video_path = "https://dnrjl01ydafck.cloudfront.net/v2/upload/processed_towel_fail.mp4"
    video_path = "/workspaces/ares/data/pi_demos/processed_towel_fail.mp4"
    # prompt = """please describe the video. specifically, how well the robot succeeds in the task it is attempting -- does it actually achieve the task? if not, why not?"""
    
    # vlm = GeminiVideoLLM("gemini", "gemini-1.5-flash", dict())
    # out = vlm.ask(None, prompt, {}, None, video_path)

    frames = split_video_to_frames(video_path)[::10]
    if frames[0].shape[0] > 224 or frames[0].shape[1] > 224:
        # downsample each frame to (224,224)
        frames = [cv2.resize(frame, (224, 224)) for frame in frames]

    provider = "gemini"
    llm = f"{provider}/gemini-1.5-flash"
    llm = LLM(provider=provider, llm=llm)
    # if not llm.check_valid_key():
    #     print("Invalid key")
    #     exit(1)

    from pydantic import BaseModel, Field

    class TrajectoryDescription(BaseModel):
        robot_setup: t.Literal["one arm", "two arms"]
        environment: t.Literal["floor", "table", "other"]
        lighting_conditions: t.Literal["normal", "dim", "bright"]
        task: str = Field(max_length=50)  # Short task description
        description: str = Field(max_length=500)  # Detailed task description

    # Build instruction string dynamically from model fields
    field_instructions = []
    for field_name, field in TrajectoryDescription.model_fields.items():
        field_instructions.append(f"    - {field_name}: {str(field)}")

    instructions = f"""
    Consider all the features of the video. Respond with an answer for each of the features below:
    {chr(10).join(field_instructions)}
    """.strip()

    # Build example response dict dynamically from model fields
    example_dict = {}
    traj_items = list(TrajectoryDescription.model_fields.items())
    for field_name, field in [traj_items[0], traj_items[-1]]:
        if hasattr(field.annotation, "__args__"):  # For Literal types
            example_dict[field_name] = field.annotation.__args__[0]
        else:
            example_dict[field_name] = "..."

    response_format = f"""
    Respond with a python dict, e.g. {example_dict}

    Here is a link to the video: https://dnrjl01ydafck.cloudfront.net/v2/upload/processed_towel_fail.mp4
    """.strip()

    # im_path = "data/pi_eggs.png"
    # im_path = "https://dnrjl01ydafck.cloudfront.net/v2/upload/processed_towel_fail.mp4"

    info_dict = {
        "instructions": instructions,
        "response_format": response_format,
    }
    # images = [im_path]



    messages, res = llm.ask(
        "test_prompt.jinja2",
        info_dict,
        images=frames,
    )

    breakpoint()
    print(res.choices[0].message.content)
