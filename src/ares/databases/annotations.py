import json
import os
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pydantic import BaseModel, Field


def rle_to_binary_mask(rle: dict) -> np.ndarray:
    """Convert RLE format to binary mask."""
    rle = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
    return mask_utils.decode(rle)


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to RLE format."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"counts": rle["counts"].decode("utf-8"), "size": rle["size"]}


class Annotation(BaseModel):
    # Core detection attributes
    bbox: list[float]  # [x1, y1, x2, y2] / LTRBformat
    category_id: int | None = None
    category_name: str | None = None
    # denotes confidence of the detection if float else None if ground truth
    score: float | None = None

    # Segmentation attributes
    segmentation: Optional[Union[dict, list[list[float]]]] = (
        None  # RLE or polygon format
    )

    # Tracking and metadata
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Get bbox in xyxy format."""
        return self.bbox

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        """Get bbox in xywh format."""
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    # Add this validation method
    def model_post_init(self, __context) -> None:
        """Validate bbox format after initialization."""
        x1, y1, x2, y2 = self.bbox
        if x1 > x2:
            raise ValueError(f"Invalid bbox: x1 ({x1}) must be <= x2 ({x2})")
        if y1 > y2:
            raise ValueError(f"Invalid bbox: y1 ({y1}) must be <= y2 ({y2})")

    @property
    def mask(self) -> np.ndarray:
        """Convert RLE or polygon segmentation to binary mask."""
        if self.segmentation is None:
            return None

        if isinstance(self.segmentation, dict):  # RLE format
            return rle_to_binary_mask(self.segmentation)
        else:  # Polygon format
            mask = np.zeros(
                (
                    int(max(p[1] for p in self.segmentation[0])) + 1,
                    int(max(p[0] for p in self.segmentation[0])) + 1,
                ),
                dtype=np.uint8,
            )
            points = np.array(self.segmentation[0]).reshape((-1, 2))
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            return mask

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        bbox: list[float],
        category_id: int,
        category_name: str,
        score: float,
        **kwargs,
    ) -> "Annotation":
        """Create annotation from binary mask."""
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            score=score,
            segmentation=binary_mask_to_rle(mask),
            **kwargs,
        )

    def compute_iou(self, other: "Annotation") -> float:
        """Compute IoU between this annotation and another."""
        if self.mask is None or other.mask is None:
            # Fall back to bbox IoU if masks aren't available
            return self.compute_bbox_iou(other)

        intersection = np.logical_and(self.mask, other.mask).sum()
        union = np.logical_or(self.mask, other.mask).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    def compute_bbox_iou(self, other: "Annotation") -> float:
        """Compute IoU between bounding boxes."""
        # Extract coordinates
        x1, y1, x2, y2 = self.bbox
        x1_, y1_, x2_, y2_ = other.bbox

        # Compute intersection
        x_left = max(x1, x1_)
        y_top = max(y1, y1_)
        x_right = min(x2, x2_)
        y_bottom = min(y2, y2_)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Compute areas
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)

        # Compute IoU
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def transform(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> "Annotation":
        """Transform the annotation coordinates."""
        # Transform bbox
        x1, y1, x2, y2 = self.bbox
        if flip_horizontal:
            x1, x2 = 1 - x2, 1 - x1
        if flip_vertical:
            y1, y2 = 1 - y2, 1 - y1

        transformed_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

        # Transform segmentation if it exists
        transformed_segmentation = None
        if isinstance(self.segmentation, list):  # Polygon format
            transformed_segmentation = []
            for polygon in self.segmentation:
                transformed_polygon = []
                for i in range(0, len(polygon), 2):
                    x, y = polygon[i], polygon[i + 1]
                    if flip_horizontal:
                        x = 1 - x
                    if flip_vertical:
                        y = 1 - y
                    transformed_polygon.extend([x * scale_x, y * scale_y])
                transformed_segmentation.append(transformed_polygon)

        # Create new instance with transformed coordinates
        return Annotation(
            bbox=transformed_bbox,
            category_id=self.category_id,
            category_name=self.category_name,
            score=self.score,
            segmentation=transformed_segmentation or self.segmentation,
            track_id=self.track_id,
            attributes=self.attributes,
        )

    def visualize(
        self,
        image: Union[np.ndarray, Image.Image],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Visualize the annotation on an image."""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        vis_image = image.copy()

        # Draw bbox
        x1, y1, x2, y2 = [int(c) for c in self.bbox]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        # Draw mask if available
        if self.mask is not None:
            mask_overlay = vis_image.copy()
            mask_overlay[self.mask > 0] = color
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

        # Add text
        text = f"{self.category_name}: {self.score:.2f}"
        if self.track_id is not None:
            text += f" ID: {self.track_id}"

        cv2.putText(
            vis_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

        return vis_image

    @staticmethod
    def visualize_all(
        image: Union[np.ndarray, Image.Image],
        annotations: list["Annotation"],
        colors: Optional[list[tuple]] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw all annotations on a single image."""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        vis_image = image.copy()

        # Generate colors if not provided
        if colors is None:
            colors = [
                (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
                for _ in range(len(annotations))
            ]

        # Draw each annotation
        for ann, color in zip(annotations, colors):
            x1, y1, x2, y2 = [int(c) for c in ann.bbox]

            # Draw mask if available
            if ann.mask is not None:
                mask_overlay = vis_image.copy()
                mask_overlay[ann.mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

            # Draw bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # Add text
            text = f"{ann.category_name}: {ann.score:.2f}"
            if ann.track_id is not None:
                text += f" ID: {ann.track_id}"

            cv2.putText(
                vis_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return vis_image

    def to_dict(self) -> dict:
        """Convert annotation to dictionary format suitable for JSON serialization."""
        base_dict = self.model_dump(exclude_none=True)
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        """Create annotation from dictionary."""
        return cls(**data)

    def save_json(self, filepath: str) -> None:
        """Save annotation to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, filepath: str) -> "Annotation":
        """Load annotation from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    import requests
    import torch
    from PIL import Image
    from transformers import (
        AutoModelForMaskGeneration,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
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

    detector_id = None
    detector_id = (
        "IDEA-Research/grounding-dino-tiny" if detector_id is None else detector_id
    )
    segmenter_id = None
    segmenter_id = "facebook/sam-vit-base" if segmenter_id is None else segmenter_id
    device = "cpu"

    detector_processor = AutoProcessor.from_pretrained(detector_id)
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_id
    ).to(device)

    segmentor_processor = AutoProcessor.from_pretrained(segmenter_id)
    segmentator_model = AutoModelForMaskGeneration.from_pretrained(
        segmenter_id, token=os.environ.get("HUGGINGFACE_API_KEY")
    ).to(device)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Check for cats and remote controls
    text = "a cat. a remote control."

    box_annotations = run_detector(
        detector_processor, detector_model, image, text, device
    )
    segment_annotations = run_segmenter(
        segmentor_processor, segmentator_model, image, device, box_annotations
    )

    # code to visualize
    for i, frame_annotations in enumerate(segment_annotations):
        # Visualize all annotations in the frame at once
        # vis_image = Annotation.visualize_all(image, frame_annotations)
        # # Instead of cv2.imshow, save to file
        # cv2.imwrite(
        #     f"/tmp/viz_frame_{i}.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        # )

        # instead, do each annotation separately and save to file
        for j, ann in enumerate(frame_annotations):
            vis_image = ann.visualize(image)
            cv2.imwrite(
                f"/tmp/viz_frame_{i}_ann_{j}.png",
                cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR),
            )
    # breakpoint()
    # test saving to file and reloading
    ann.save_json("/tmp/test.json")
    ann2 = Annotation.load_json("/tmp/test.json")
    print(ann.segmentation["counts"][:5])
    print(ann2.segmentation["counts"][:5])
    breakpoint()
