import os 
import cv2
import numpy as np

import typing as t
from PIL import Image
import io
import base64

    
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

def split_video_to_frames(video_path: str) -> list[np.ndarray]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def choose_and_preprocess_frames(all_frames: list[np.ndarray], n_frames: int = 10, specified_frames: list[int] | None = None, resize: tuple[int, int] | None = None) -> list[np.ndarray]:
    if specified_frames is None:
        total_frames = len(all_frames)
        indices = np.linspace(0, total_frames-1, n_frames, dtype=int, endpoint=True)
        frames = [all_frames[i] for i in indices]
    else:
        frames = [all_frames[i] for i in specified_frames]

    if resize:
        frames = [cv2.resize(frame, (224, 224)) for frame in frames]
    return frames