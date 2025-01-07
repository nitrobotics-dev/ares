import os
import pickle
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import pymongo
import torch
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm
from transformers import (
    AutoModelForMaskGeneration,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from ares.configs.annotations import Annotation, binary_mask_to_rle
from ares.databases.annotation_database import (
    TEST_ANNOTATION_DB_PATH,
    AnnotationDatabase,
)
from ares.image_utils import load_video_frames
from ares.models.base import VLM
from ares.models.shortcuts import get_gemini_2_flash, get_gpt_4o


class GroundingAnnotator:
    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        segmenter_id: str | None = "facebook/sam-vit-base",
        detector_thresholds: dict[str, float] | None = None,
        device: str = "cpu",
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
        print(f"Loaded models for {detector_id} and {segmenter_id} on device {device}")

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
        images: List[Image.Image],
        labels_str: str,
    ) -> List[List[Annotation]]:
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
                    frame_annotations.append(Annotation(**ann_dict))
            all_annotations.append(frame_annotations)

        return all_annotations

    def run_segmenter(
        self,
        images: List[Image.Image],
        annotations: List[List[Annotation]],
    ) -> List[List[Annotation]]:
        # Process each image's annotations
        all_points = []
        all_labels = []
        max_points = max(len(frame_anns) for frame_anns in annotations)

        for frame_anns in annotations:
            frame_points = [
                [
                    (box.bbox_xyxy[0] + box.bbox_xyxy[2]) / 2,
                    (box.bbox_xyxy[1] + box.bbox_xyxy[3]) / 2,
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
                ann.segmentation = binary_mask_to_rle(best_mask.numpy())

        return annotations

    def process_batch(
        self,
        images: List[Image.Image],
        labels_str: str,
    ) -> List[List[Annotation]]:
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
        frames: List[np.ndarray],
        labels_str: str,
        batch_size: int = 2,
    ) -> List[List[Annotation]]:
        """Annotate video frames in batches."""
        all_annotations = []

        # Convert frames to PIL Images
        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        # Process in batches
        for i in tqdm(range(0, len(pil_frames), batch_size)):
            batch_frames = pil_frames[i : i + batch_size]
            batch_annotations = self.process_batch(batch_frames, labels_str)
            all_annotations.extend(batch_annotations)

        return all_annotations


def get_vlm_labels(
    vlm: VLM, frames: List[np.ndarray], prompt_filename: str, task_instructions: str
) -> str:
    """Get object labels from VLM."""
    if task_instructions is None:
        task_instructions = ""
    print(f"Task instructions: {task_instructions}")
    messages, response = vlm.ask(
        info=dict(task_instructions=task_instructions),
        prompt_filename=prompt_filename,
        images=frames,
    )
    label_str = response.choices[0].message.content
    print(f"Label string: {label_str} ")
    return label_str


if __name__ == "__main__":
    from ares.databases.structured_database import (
        TEST_ROBOT_DB_PATH,
        RolloutSQLModel,
        get_rollout_by_name,
        setup_database,
    )

    # Initialize components
    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    annotator = GroundingAnnotator(
        segmenter_id=None,
        detector_thresholds={"box_threshold": 0.3, "text_threshold": 0.2},
    )
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)

    # Process video
    # Process video
    # dataset_name = "cmu_play_fusion"
    # fname = "data/train/episode_208.mp4"
    formal_dataset_name = "UCSD Kitchen"
    dataset_name = "ucsd_kitchen_dataset_converted_externally_to_rlds"
    # fnames = ["data/train/episode_2.mp4", "data/train/episode_3.mp4"]
    fnames = [f"data/train/episode_{i}.mp4" for i in range(10, 20)]
    target_fps = 1

    # vlm = get_gemini_2_flash()
    vlm = get_gpt_4o()

    for fname in fnames:
        rollout = get_rollout_by_name(
            engine, formal_dataset_name, fname.replace("mp4", "npy")
        )
        frames, frame_indices = load_video_frames(dataset_name, fname, target_fps)
        label_str = get_vlm_labels(
            vlm,
            frames,
            "grounding_description.jinja2",
            rollout.task.language_instruction,
        )
        # label_str = "a robot. a robot gripper. a sink. a cabinet"
        label_str = label_str.replace("a ", "").replace("an ", "")

        video_annotations = annotator.annotate_video(frames, label_str)
        ann_db.delete_video_and_annotations(video_id=f"{dataset_name}/{fname}")
        video_id = ann_db.add_video_with_annotations(
            dataset_name=dataset_name,
            video_path=fname,
            frames=frames,
            frame_indices=frame_indices,
            annotations=video_annotations,
            label_str=label_str,
        )
        print(
            f"Added video {video_id} to database with {[len(x) for x in video_annotations]} frame annotations"
        )
    breakpoint()
