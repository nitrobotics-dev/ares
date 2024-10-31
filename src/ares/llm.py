import base64
import os
from litellm import completion, acompletion
import asyncio
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from PIL import Image
import numpy as np
import typing as t
import io


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
    def __init__(self, llm: str, llm_kwargs: dict = {}):
        self.llm = llm
        self.llm_kwargs = llm_kwargs

    def _get_prompt(self, prompt_filename: str, info: dict) -> str:
        # Set up Jinja environment to load templates from src/ares directory
        jinja_env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        template = jinja_env.get_template(prompt_filename)
        # Render template with information
        return template.render(**info)

    def _construct_messages(self, prompt_filename: str, info: dict) -> list[dict]:
        content = []
        prompt = self._get_prompt(
            prompt_filename, {k: v for k, v in info.items() if "image" not in k}
        )
        content.append({"type": "text", "text": prompt})
        if "images" in info:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(info['images'][i])}"
                    },
                }
                for i in range(len(info["images"]))
            ]
            content.extend(image_contents)
        return [{"role": "user", "content": content}]

    def ask(self, prompt_filename: str, info: dict) -> str:
        messages = self._construct_messages(prompt_filename, info)
        return completion(model=self.llm, messages=messages, **self.llm_kwargs)

    async def aask(self, prompt_filename: str, info: dict) -> str:
        messages = self._construct_messages(prompt_filename, info)
        return await acompletion(model=self.llm, messages=messages, **self.llm_kwargs)
