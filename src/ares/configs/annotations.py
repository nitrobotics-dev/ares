import json
import typing as t

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pydantic import BaseModel, Field, model_validator


def rle_to_binary_mask(rle: dict) -> np.ndarray:
    """Convert RLE format to binary mask."""
    rle = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
    return mask_utils.decode(rle)


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to RLE format."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"counts": rle["counts"].decode("utf-8"), "size": rle["size"]}


class Annotation(BaseModel):
    """
    Base object to hold annotation data.
    """

    # Core detection attributes
    description: str | None = None
    bbox: list[float] | None = None  # [x1, y1, x2, y2] / LTRB format
    category_id: int | None = None
    category_name: str | None = None
    # denotes confidence of the detection if float else None if ground truth
    score: float | None = None

    # Segmentation attributes
    segmentation: t.Optional[t.Union[dict, list[list[float]]]] = (
        None  # RLE or polygon format
    )

    # Tracking and metadata
    track_id: t.Optional[int] = None
    attributes: dict[str, t.Any] = Field(default_factory=dict)
    annotation_type: str | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            np.ndarray: lambda x: x.tolist(),
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
        },
    }

    @model_validator(mode="after")
    def sanity_check(self) -> "Annotation":
        # some portion of the annotation must be present!
        if (
            self.description is None
            and self.bbox is None
            and self.segmentation is None
            and self.attributes is None
        ):
            raise ValueError(
                "Annotation must have at least one attribute; description, bbox, segmentation, or attributes"
            )
        return self

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
    def model_post_init(self, __context: t.Any) -> None:
        """Validate bbox format after initialization."""
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if x1 > x2:
                raise ValueError(f"Invalid bbox: x1 ({x1}) must be <= x2 ({x2})")
            if y1 > y2:
                raise ValueError(f"Invalid bbox: y1 ({y1}) must be <= y2 ({y2})")

    @property
    def mask(self) -> np.ndarray | None:
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
        t.Union = np.logical_or(self.mask, other.mask).sum()
        return float(intersection) / float(t.Union) if t.Union > 0 else 0.0

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
        t.Union = area1 + area2 - intersection
        return intersection / t.Union if t.Union > 0 else 0.0

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

    def __json__(self):
        """
        Helper method for JSON serialization
        Used as `json.dumps(..., default=lambda x: x.__json__() if hasattr(x, "__json__") else x)`
        """
        return self.model_dump()
