from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pymongo import MongoClient

from ares.configs.annotations import Annotation

TEST_ANNOTATION_DB_PATH = "mongodb://localhost:27017"


class AnnotationDatabase:
    def __init__(self):
        self.client = MongoClient(TEST_ANNOTATION_DB_PATH)
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

    def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video metadata."""
        video = self.videos.find_one({"_id": video_id})
        return video["metadata"] if video else None

    def get_annotations(
        self,
        video_id: str,
        annotation_type: Optional[str] = None,
        frame: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get annotations for a video, optionally filtered by type and frame."""
        query = {"video_id": video_id}
        if annotation_type:
            query["type"] = annotation_type
        if frame is not None:
            query["frame"] = frame
        return list(self.annotations.find(query))

    def add_frame_annotations(
        self, video_id: str, frame: int, annotations: List[Annotation]
    ) -> List[str]:
        """Add multiple annotations for a specific frame."""
        annotation_ids = []
        for ann in annotations:
            # Convert the Annotation object to a dict suitable for MongoDB
            ann_dict = ann.to_dict()
            ann_id = self.add_annotation(
                video_id=video_id,
                key=f"frame_{frame}_annotation",
                value=ann_dict,
                annotation_type="detection",
                frame=frame,
            )
            annotation_ids.append(ann_id)
        return annotation_ids

    def get_frame_annotations(self, video_id: str, frame: int) -> List["Annotation"]:
        """Get all annotations for a specific frame."""

        annotations = self.get_annotations(
            video_id=video_id, annotation_type="detection", frame=frame
        )
        return [Annotation.from_dict(ann["value"]) for ann in annotations]

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the current state of the database.

        Returns:
            Dict containing:
            - Total number of videos
            - Total number of annotations
            - Annotations per type
            - Number of frame-specific vs video-level annotations
        """
        stats = {
            "total_videos": self.videos.count_documents({}),
            "total_annotations": self.annotations.count_documents({}),
            "annotations_by_type": {},
            "frame_annotations": self.annotations.count_documents(
                {"frame": {"$ne": None}}
            ),
            "video_level_annotations": self.annotations.count_documents(
                {"frame": None}
            ),
        }

        # Get counts per annotation type
        pipeline = [{"$group": {"_id": "$type", "count": {"$sum": 1}}}]
        for type_stat in self.annotations.aggregate(pipeline):
            stats["annotations_by_type"][type_stat["_id"]] = type_stat["count"]

        return stats

    def delete_annotations(
        self, video_id: str, annotation_type: Optional[str] = None
    ) -> int:
        """Delete annotations for a video, optionally filtered by type.

        Returns:
            Number of annotations deleted
        """
        query = {"video_id": video_id}
        if annotation_type:
            query["type"] = annotation_type
        result = self.annotations.delete_many(query)
        return result.deleted_count

    def delete_video(self, video_id: str, delete_annotations: bool = True) -> bool:
        """Delete a video and optionally its annotations.

        Args:
            video_id: ID of the video to delete
            delete_annotations: If True, also delete all annotations for this video

        Returns:
            True if video was found and deleted
        """
        if delete_annotations:
            self.delete_annotations(video_id)

        result = self.videos.delete_one({"_id": video_id})
        return result.deleted_count > 0


# make test inputs
if __name__ == "__main__":
    db = AnnotationDatabase()
    # db.add_video("test_video", {"test": "test"})
    # db.add_annotation("test_video", "test_annotation", "test_value", "test_type", 1)
    # anns = db.get_annotations("test_video", "test_type", 1)
    # add more test samples to the database
    # for i in range(100):
    #     db.add_annotation(
    #         "test_video", f"test_annotation_{i}", f"test_value_{i}", "test_type", i
    #     )
    # # also add video laevel anns and frame level anns
    # db.add_annotation("test_video", "test_annotation", "test_value", "test_type", None)
    # db.add_annotation(
    #     "test_video", "test_annotation", "test_value", "test_type", [1, 2, 3]
    # )
    # stats = db.get_database_stats()
    # print(stats)

    # # Delete all annotations for "test_video"
    # deleted_count = db.delete_annotations("test_video")
    # print(f"Deleted {deleted_count} annotations")

    # # Delete the test video and its annotations
    # db.delete_video("test_video")

    # stats = db.get_database_stats()
    # print(stats)
    # breakpoint()
