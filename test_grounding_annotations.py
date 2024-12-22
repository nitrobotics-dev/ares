import os
import tempfile

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForMaskGeneration,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from ares.databases.annotations import Annotation, binary_mask_to_rle
from ares.image_utils import (
    choose_and_preprocess_frames,
    get_frame_indices_for_fps,
    get_image_from_path,
    get_video_frames,
    get_video_from_path,
    split_video_to_frames,
)


def run_detector(
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    image: Image.Image,
    text: str,
    device: str,
) -> list[list[Annotation]]:
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    output_annotations = []
    for image, annotations in zip([image], results):
        these_annotations = []
        if "boxes" in annotations:
            for i in range(len(annotations["boxes"])):
                ann_dict = {
                    "bbox": annotations["boxes"][i].tolist(),
                    "category_name": annotations["labels"][i],
                    "score": annotations["scores"][i].item(),
                }
                these_annotations.append(Annotation(**ann_dict))
        output_annotations.append(these_annotations)
    return output_annotations


def run_segmenter(
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    image: Image.Image,
    device: str,
    annotations: list[list[Annotation]],
) -> list[list[Annotation]]:
    boxes = [
        [ann.bbox_xyxy for ann in frame_annotations]
        for frame_annotations in annotations
    ]

    # Reshape input points to match expected format - Fix the dimensionality
    input_points = [
        [
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
        ]  # Use center point of each box
        for box in boxes[0]
    ]

    input_labels = [1] * len(boxes[0])  # Simplified labels format

    inputs = processor(
        images=image,
        input_points=[input_points],  # Wrap in list for batch dimension
        input_labels=[input_labels],  # Wrap in list for batch dimension
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )
    # add to annotations
    for i, frame_masks in enumerate(masks):
        for j, mask in enumerate(frame_masks.squeeze(0)):
            annotations[i][j].segmentation = binary_mask_to_rle(mask.numpy())
    return annotations


def setup_models_for_grounding(
    detector_id: str = "IDEA-Research/grounding-dino-tiny",
    segmenter_id: str = "facebook/sam-vit-base",
    device: str = "cpu",
) -> tuple[
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    AutoModelForMaskGeneration,
]:
    detector_processor = AutoProcessor.from_pretrained(detector_id)
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_id
    ).to(device)

    segmentor_processor = AutoProcessor.from_pretrained(segmenter_id)
    segmentator_model = AutoModelForMaskGeneration.from_pretrained(
        segmenter_id, token=os.environ.get("HUGGINGFACE_API_KEY")
    ).to(device)
    return detector_processor, detector_model, segmentor_processor, segmentator_model


class GroundingAnnotator:
    def __init__(self, detector_id: str, segmenter_id: str, device: str = "cpu"):
        self.device = device
        (
            self.detector_processor,
            self.detector_model,
            self.segmentor_processor,
            self.segmentator_model,
        ) = setup_models_for_grounding(detector_id, segmenter_id, device)

    def annotate(self, image_url: str, text: str) -> list[list[Annotation]]:
        image = get_image_from_path(image_url)
        box_annotations = run_detector(
            self.detector_processor, self.detector_model, image, text, self.device
        )
        segment_annotations = run_segmenter(
            self.segmentor_processor,
            self.segmentator_model,
            image,
            self.device,
            box_annotations,
        )
        return segment_annotations

    def annotate_video(
        self, frames: list[np.ndarray], text: str
    ) -> list[list[Annotation]]:
        """Annotate provided video frames.

        Args:
            frames: List of frames to annotate
            text: Text prompt for object detection

        Returns:
            List of annotations for each frame
        """
        all_annotations = []
        for frame in frames:
            # Save frame temporarily and get its path
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                # Use existing annotate method
                frame_annotations = self.annotate(tmp.name, text)
                all_annotations.extend(frame_annotations)
                os.unlink(tmp.name)  # Clean up temp file

        return all_annotations


if __name__ == "__main__":
    # Example with single image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    text = "a cat. a remote control."
    annotator = GroundingAnnotator()
    annotations = annotator.annotate(image_url, text)

    # Example with video
    video_path = get_video_from_path("dataset_name", "video.mp4")
    frame_indices = get_frame_indices_for_fps(video_path, target_fps=1)
    all_frames = split_video_to_frames(video_path)
    frames_to_process = choose_and_preprocess_frames(
        all_frames, specified_frames=frame_indices
    )
    video_annotations = annotator.annotate_video(frames_to_process, text)
