import asyncio
import base64
import os
import typing as t
from asyncio import Semaphore

import numpy as np
import torch
import vertexai
from jinja2 import Environment, FileSystemLoader
from litellm import acompletion, completion
from litellm.utils import ModelResponse
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor
from vertexai.generative_models import GenerativeModel, Part

from ares.image_utils import encode_image

# TODO -- singleton, model specific rate limits
RATE_LIMITS = {
    "openai": 100,  # 5000 RPM
    "anthropic": 10,  # 50 RPM
    "gemini": 5,  # 100 RPM
}


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


class VLM:
    def __init__(self, provider: str, name: str):
        self.provider = provider
        self.name = name
        # Add rate limiting per provider
        self._setup_rate_limits()
        self.check_valid_key()

    def _setup_rate_limits(self, max_concurrent: int | None = None):
        max_concurrent = max_concurrent or RATE_LIMITS.get(self.provider, 10)
        self.semaphore = Semaphore(max_concurrent)

    def check_valid_key(self) -> bool:
        # note: don't use litellm util here, it uses 10 tokens! roll our own check with 1
        try:
            res = completion(
                model=self.name,
                messages=[{"role": "user", "content": "!"}],
                max_tokens=1,
            )
        except Exception as e:
            print(f"Error checking valid key: {e}")
            return False
        return True

    def _get_prompt(self, prompt_filename: str, info: dict) -> str:
        # Set up Jinja environment to load templates from src/ares/models/prompts directory
        jinja_env = Environment(
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "prompts"))
        )
        template = jinja_env.get_template(prompt_filename)
        # Render template with information
        return template.render(**info)

    def _construct_messages(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        double_prompt: bool = False,
    ) -> list[dict[str, t.Any]]:
        content: list[dict[str, t.Any]] = []
        if "prompt" not in info and prompt_filename is not None:
            prompt = self._get_prompt(prompt_filename, info)
        else:
            prompt = info["prompt"]
        content.append({"type": "text", "text": prompt})
        if images is not None and len(images) > 0:
            image_contents = []
            for i in range(len(images)):
                text_content = {"type": "text", "text": f"Image {i}:"}
                image_content = structure_image_messages(
                    images[i], provider=self.provider
                )
                image_contents.extend([text_content, image_content])
            content.extend(image_contents)
        if double_prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    async def ask_async(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
        model_kwargs: t.Dict | None = None,
    ) -> t.Tuple[list[dict[str, t.Any]], ModelResponse]:
        """Rate-limited async version of ask method"""
        async with self.semaphore:
            model_kwargs = model_kwargs or dict()
            if video_path:
                raise NotImplementedError("Video path not implemented for this VLM")
            messages = self._construct_messages(
                info, prompt_filename, images, double_prompt
            )
            print(
                f"submitting async request to {self.name}"
                + (f" with {len(images)} images" if images else "")
            )
            return messages, await acompletion(
                model=self.name, messages=messages, **model_kwargs
            )

    async def ask_batch_async(
        self,
        infos: list[dict],
        prompt_filename: str | None = None,
        images_list: (
            list[t.Sequence[t.Union[str, np.ndarray, Image.Image]]] | None
        ) = None,
        double_prompt: bool = False,
        model_kwargs: t.Dict | None = None,
    ) -> list[t.Tuple[list[dict[str, t.Any]], ModelResponse]]:
        """Process multiple requests with rate limiting"""
        if images_list is None:
            images_list = [None] * len(infos)

        tasks = [
            self.ask_async(
                info,
                prompt_filename=prompt_filename,
                images=images,
                double_prompt=double_prompt,
                model_kwargs=model_kwargs,
            )
            for info, images in zip(infos, images_list)
        ]

        return await asyncio.gather(*tasks)


class GeminiVideoVLM(VLM):
    def __init__(self, provider: str, name: str):
        super().__init__(provider, name)
        vertexai.init(
            project=os.environ["VERTEX_PROJECT"], location=os.environ["VERTEX_LOCATION"]
        )
        self.model = GenerativeModel(name)
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

    def ask(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
        model_kwargs: t.Dict | None = None,
    ) -> t.Tuple[list[dict[str, t.Any]], ModelResponse]:
        model_kwargs = model_kwargs or dict()
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


class Embedder:
    def __init__(self, provider: str, name: str):
        self.provider = provider
        self.name = name
        self._setup_model()

    def _setup_model(self) -> None:
        """Initialize model and processor. Should be implemented by child classes."""
        name = f"{self.provider}/{self.name}"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)

    def _process_image_input(
        self, inp: t.Union[Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various image input types to PIL Image."""
        if isinstance(inp, np.ndarray):
            return Image.fromarray(inp)
        elif isinstance(inp, str):
            return Image.open(inp)
        return inp

    def embed(self, inp: t.Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Embed text or image input."""
        with torch.inference_mode():
            if isinstance(inp, str):
                # Text embedding
                inputs = self.processor(text=[inp], padding=True, return_tensors="pt")
                outputs = self.model.get_text_features(**inputs)
                return outputs.detach().numpy()[0]
            else:
                # Image embedding
                image = self._process_image_input(inp)
                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.model.get_image_features(**inputs)
                return outputs.detach().numpy()[0]


class SentenceTransformerEmbedder:
    def __init__(self, provider: str, name: str):
        self.model = SentenceTransformer(f"{provider}/{name}", trust_remote_code=True)

    def embed(self, inp: str, prefix: str = "clustering: ") -> np.ndarray:
        return np.array(self.model.encode(prefix + inp))
