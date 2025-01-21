import asyncio
from typing import List

import numpy as np
from PIL import Image

from ares.configs.annotations import Annotation, binary_mask_to_rle
from ares.models.base import VLM


async def get_grounding_nouns_async(
    vlm: VLM,
    image: np.ndarray,
    task_instructions: str,
    prompt_filename: str = "grounding_description.jinja2",
) -> str:
    """Get object labels from VLM asynchronously."""
    if task_instructions is None:
        task_instructions = ""
    _, response = await vlm.ask_async(
        info=dict(task_instructions=task_instructions),
        prompt_filename=prompt_filename,
        images=[image],
    )
    label_str = response.choices[0].message.content
    label_str = label_str.replace("a ", "").replace("an ", "")
    return label_str


def get_grounding_nouns(
    vlm: VLM,
    image: np.ndarray,
    task_instructions: str,
    prompt_filename: str = "grounding_description.jinja2",
) -> str:
    return asyncio.run(
        get_grounding_nouns_async(vlm, image, task_instructions, prompt_filename)
    )


def convert_to_annotations(
    detection_results: list[list[dict]],
) -> list[list[Annotation]]:
    """Convert detection results from dictionaries to Annotation objects."""
    # convert masks to rle
    outputs = []
    for frame_anns in detection_results:
        frame_anns = []
        for ann in frame_anns:
            if ann.get("segmentation") is not None:
                ann["segmentation"] = binary_mask_to_rle(ann["segmentation"])
            frame_anns.append(ann)
        outputs.append(frame_anns)
    return outputs
