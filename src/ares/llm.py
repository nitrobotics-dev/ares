import base64
import os
import typing as t

import numpy as np
import vertexai
from jinja2 import Environment, FileSystemLoader
from litellm import completion
from litellm.utils import ModelResponse
from PIL import Image
from vertexai.generative_models import GenerativeModel, Part

from ares.image_utils import encode_image


def structure_image_messages(
    image: t.Union[str, np.ndarray, Image.Image], provider: str
) -> dict:
    if "anthropic" in provider:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",  # matches the JPEG format above
                "data": encode_image(image),
            },
        }
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
    }


class LLM:
    def __init__(self, provider: str, llm_name: str):
        self.provider = provider
        self.llm_name = llm_name
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
            image_contents = []
            for i in range(len(images)):
                text_content = {"type": "text", "text": f"Image {i}:"}
                image_content = structure_image_messages(
                    images[i], provider=self.provider
                )
                image_contents.extend([text_content, image_content])
            # image_contents = [
            #     structure_image_messages(images[i], provider=self.provider)
            #     for i in range(len(images))
            # ]
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
        llm_kwargs: t.Dict = dict(),
    ) -> t.Tuple[list[dict[str, t.Any]], ModelResponse]:
        if video_path:
            raise NotImplementedError("Video path not implemented for this LLM")
        messages = self._construct_messages(
            prompt_filename, info, images, double_prompt
        )
        return messages, completion(
            model=self.llm_name, messages=messages, **llm_kwargs
        )


class GeminiVideoLLM(LLM):
    def __init__(self, provider: str, llm_name: str):
        super().__init__(provider, llm_name)
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
        llm_kwargs: t.Dict = dict(),
    ) -> t.Tuple[list[dict[str, t.Any]], ModelResponse]:
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
