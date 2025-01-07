import hashlib
import random

import cv2
import numpy as np

from ares.configs.annotations import Annotation


# Generate a consistent color map for classes
def get_color_mapping(category_str: str) -> tuple[int, int, int]:
    # consistent color mapping based on category string
    hash_str = hashlib.sha256(category_str.encode()).hexdigest()[:6]
    # Convert pairs of hex digits to RGB values (0-255)
    r = int(hash_str[0:2], 16)
    g = int(hash_str[2:4], 16)
    b = int(hash_str[4:6], 16)
    return (r, g, b)


# Draw annotations
def draw_annotations(
    image: np.ndarray,
    annotations: list[Annotation],
    show_scores: bool = True,
    alpha: float = 0.5,  # transparency for segmentation masks
) -> np.ndarray:
    annotated_image = image.copy()
    overlay = image.copy()

    # Track unique categories for legend
    unique_categories = {}

    for annotation in annotations:
        label = annotation.category_name or "unknown"
        color = get_color_mapping(label)
        unique_categories[label] = color

        # Draw bounding box
        if annotation.bbox:
            x1, y1, x2, y2 = map(int, annotation.bbox)
            # Draw rectangle with transparency
            cv2.rectangle(
                overlay, (x1, y1), (x2, y2), color, -1
            )  # Filled box for overlay
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)  # Border

            # Prepare label text
            if show_scores and annotation.score is not None:
                label_text = f"{label} {annotation.score:.2f}"
            else:
                label_text = label

            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Adjust label position to ensure it's within image bounds
            text_x = min(max(x1, 0), image.shape[1] - text_w)
            text_y = max(y1 - 2, text_h + 4)  # Ensure there's room for text

            # Draw label background and text
            cv2.rectangle(
                annotated_image,
                (text_x, text_y - text_h - 4),
                (text_x + text_w, text_y),
                color,
                -1,
            )
            cv2.putText(
                annotated_image,
                label_text,
                (text_x, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw segmentation mask
        if annotation.segmentation:
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[annotation.mask == 1] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

    # Calculate legend dimensions
    legend_spacing = 25  # Vertical spacing between legend items
    legend_height = len(unique_categories) * legend_spacing + 10
    legend_padding = 20  # Padding around legend

    # Create extended canvas for image + legend
    canvas = np.full(
        (image.shape[0] + legend_height + legend_padding, image.shape[1], 3),
        255,  # White background
        dtype=np.uint8,
    )

    # Place the annotated image at the top
    canvas[: image.shape[0]] = cv2.addWeighted(
        overlay, alpha, annotated_image, 1 - alpha, 0
    )

    # Draw legend below the image
    legend_x = 10
    legend_y = image.shape[0] + legend_padding

    for idx, (category, color) in enumerate(unique_categories.items()):
        # Draw color box
        cv2.rectangle(
            canvas,
            (legend_x, legend_y + idx * legend_spacing - 15),
            (legend_x + 20, legend_y + idx * legend_spacing),
            color,
            -1,
        )
        # Draw category text
        cv2.putText(
            canvas,
            category,
            (legend_x + 30, legend_y + idx * legend_spacing - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas
