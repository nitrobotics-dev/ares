from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pymongo import MongoClient

TEST_ANNOTATION_DB_PATH = "mongodb://localhost:27017/"


class AnnotationDatabase:
    def __init__(self, connection_string: str = TEST_ANNOTATION_DB_PATH):
        self.client = MongoClient(connection_string)
        self.db = self.client.video_annotations

        # Just two collections - videos and annotations
        self.videos = self.db.videos
        self.annotations = self.db.annotations

        # Set up indexes
        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Create indexes for efficient querying."""
        self.annotations.create_index(
            [
                ("video_id", 1),
                ("type", 1),
                ("frame", 1),  # Will be null for whole-video annotations
            ]
        )
        self.annotations.create_index([("video_id", 1), ("key", 1)])

    def add_video(self, video_id: str, metadata: Dict[str, Any]) -> str:
        """Add or update video metadata."""
        self.videos.update_one(
            {"_id": video_id},
            {"$set": {"metadata": metadata, "updated_at": datetime.now()}},
            upsert=True,
        )
        return video_id

    def add_annotation(
        self,
        video_id: str,
        key: str,
        value: Any,
        annotation_type: str,
        frame: Optional[Union[int, List[int]]] = None,
    ) -> str:
        """
        Add a key-value annotation.

        Args:
            video_id: Identifier for the video
            key: Annotation key (e.g., "person_bbox", "scene_label")
            value: The annotation value
            annotation_type: Type of annotation (e.g., "bbox", "label", "mask")
            frame: Optional frame number or list of frames this applies to
        """
        annotation = {
            "video_id": video_id,
            "key": key,
            "value": value,
            "type": annotation_type,
            "frame": frame,
            "created_at": datetime.now(),
        }
        result = self.annotations.insert_one(annotation)
        return str(result.inserted_id)
