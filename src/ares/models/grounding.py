import asyncio
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForMaskGeneration,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from ares.configs.annotations import Annotation, binary_mask_to_rle
from ares.models.base import VLM


class GroundingAnnotator:
    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        segmenter_id: str | None = "facebook/sam-vit-base",
        detector_thresholds: dict[str, float] | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.detector_processor, self.detector_model = self.setup_detector(detector_id)
        self.segmentor_processor, self.segmentor_model = self.setup_segmenter(
            segmenter_id
        )
        self.detector_thresholds = detector_thresholds or {
            "box_threshold": 0.4,
            "text_threshold": 0.3,
        }
        print(
            f"Loaded detector {detector_id}"
            + (
                f"and segmenter {segmenter_id} on device {device}"
                if segmenter_id
                else ""
            )
        )

    def setup_detector(
        self, model_id: str
    ) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Downloading model {model_id}...")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        return processor, model

    def setup_segmenter(
        self, model_id: str | None
    ) -> Tuple[AutoProcessor, AutoModelForMaskGeneration]:
        if model_id is None:
            return None, None
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Downloading model {model_id}...")
        model = AutoModelForMaskGeneration.from_pretrained(
            model_id, token=os.environ.get("HUGGINGFACE_API_KEY")
        ).to(self.device)
        return processor, model

    def run_detector(
        self,
        images: list[Image.Image],
        labels_str: str,
    ) -> list[list[dict]]:
        # Process all images in a single batch
        inputs = self.detector_processor(
            images=images, text=[labels_str] * len(images), return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.detector_model(**inputs)

        target_sizes = [[img.size[1], img.size[0]] for img in images]  # [height, width]

        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.detector_thresholds["box_threshold"],
            text_threshold=self.detector_thresholds["text_threshold"],
            target_sizes=target_sizes,
        )

        all_annotations = []
        for image_idx, result in enumerate(results):
            frame_annotations = []
            if "boxes" in result:
                for box_idx in range(len(result["boxes"])):
                    ann_dict = {
                        "bbox": result["boxes"][box_idx].tolist(),
                        "category_name": result["labels"][box_idx],
                        "score": result["scores"][box_idx].item(),
                    }
                    frame_annotations.append(ann_dict)
            all_annotations.append(frame_annotations)

        return all_annotations

    def run_segmenter(
        self,
        images: list[Image.Image],
        annotations: list[list[dict]],
    ) -> list[list[dict]]:
        # Process each image's annotations
        all_points = []
        all_labels = []
        max_points = max(len(frame_anns) for frame_anns in annotations)

        for frame_anns in annotations:
            frame_points = [
                [
                    (box["bbox"][0] + box["bbox"][2]) / 2,
                    (box["bbox"][1] + box["bbox"][3]) / 2,
                ]
                for box in frame_anns
            ]
            # Pad points and labels to ensure consistent shape
            while len(frame_points) < max_points:
                frame_points.append([0.0, 0.0])  # Add dummy points

            frame_labels = [1] * len(frame_anns)
            frame_labels.extend([0] * (max_points - len(frame_anns)))  # Pad with zeros

            all_points.append(frame_points)
            all_labels.append(frame_labels)

        if not any(all_points):  # Handle case with no detections
            return annotations

        inputs = self.segmentor_processor(
            images=images,
            input_points=all_points,
            input_labels=all_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.segmentor_model(**inputs)

        scores = outputs["iou_scores"]
        masks = self.segmentor_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )

        # Process results for each frame
        for frame_idx, (frame_masks, frame_scores, frame_anns) in enumerate(
            zip(masks, scores, annotations)
        ):
            for obj_idx, (mask, score, ann) in enumerate(
                zip(frame_masks, frame_scores, frame_anns)
            ):
                best_mask = mask[score.argmax()]
                ann["segmentation"] = binary_mask_to_rle(best_mask.numpy())

        return annotations

    def process_batch(
        self,
        images: list[Image.Image],
        labels_str: str,
    ) -> list[list[dict]]:
        """Process a batch of images with detection and segmentation."""
        box_annotations = self.run_detector(images, labels_str)
        if not any(box_annotations):
            print("No detections found in any frame")
            return box_annotations

        if self.segmentor_model is not None:
            segment_annotations = self.run_segmenter(images, box_annotations)
            return segment_annotations
        else:
            return box_annotations

    def annotate_video(
        self,
        frames: list[np.ndarray],
        labels_str: str,
        batch_size: int = 2,
    ) -> list[list[dict]]:
        """Annotate video frames in batches."""
        all_annotations = []

        # Convert frames to PIL Images
        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        # Process in batches
        for i in range(0, len(pil_frames), batch_size):
            batch_frames = pil_frames[i : i + batch_size]
            batch_annotations = self.process_batch(batch_frames, labels_str)
            all_annotations.extend(batch_annotations)

        return all_annotations


async def get_grounding_nouns_async(
    vlm: VLM,
    frames: list[np.ndarray],
    task_instructions: str,
    prompt_filename: str = "grounding_description.jinja2",
) -> str:
    """Get object labels from VLM asynchronously."""
    if task_instructions is None:
        task_instructions = ""
    print(f"Task instructions: {task_instructions}")
    _, response = await vlm.ask_async(
        info=dict(task_instructions=task_instructions),
        prompt_filename=prompt_filename,
        images=frames,
    )
    label_str = response.choices[0].message.content
    label_str = label_str.replace("a ", "").replace("an ", "")
    return label_str


def get_grounding_nouns(
    vlm: VLM,
    frames: list[np.ndarray],
    task_instructions: str,
    prompt_filename: str = "grounding_description.jinja2",
) -> str:
    return asyncio.run(
        get_grounding_nouns_async(vlm, frames, task_instructions, prompt_filename)
    )


def convert_to_annotations(
    detection_results: list[list[dict]],
) -> list[list[Annotation]]:
    """Convert detection results from dictionaries to Annotation objects."""
    return [
        [Annotation(**ann_dict) for ann_dict in frame_anns]
        for frame_anns in detection_results
    ]
