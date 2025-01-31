import time
import typing as t
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from pymongo import MongoClient

from ares.configs.annotations import Annotation

ANNOTATION_DB_PATH = "mongodb://localhost:27017"


def get_video_id(dataset_filename: str, video_path: str) -> str:
    return (
        f"{dataset_filename}/{Path(video_path.removeprefix('/')).with_suffix('.mp4')}"
    )


class AnnotationDatabase:
    def __init__(self, connection_string: str) -> None:
        self.client: MongoClient = MongoClient(connection_string)
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

    def add_video(self, video_id: str, metadata: dict[str, t.Any]) -> str:
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
        value: Annotation,
        annotation_type: str,
        frame: t.Optional[t.Union[int, list[int]]] = None,
    ) -> str:
        """
        Add a key-value annotation.

        Args:
            video_id: Identifier for the video
            key: Annotation key (e.g., "person_bbox", "scene_label")
            value: The annotation value
            annotation_type: Type of annotation (e.g., "bbox", "label", "mask")
            frame: t.Optional frame number or list of frames this applies to
        """
        assert isinstance(value, Annotation)
        value_dict = value.to_dict()
        annotation = {
            "video_id": video_id,
            "key": key,
            "value": value_dict,
            "type": annotation_type,
            "frame": frame,
            "created_at": datetime.now(),
        }
        result = self.annotations.insert_one(annotation)
        return str(result.inserted_id)

    def get_video_metadata(self, video_id: str) -> t.Optional[dict[str, t.Any]]:
        """Get video metadata."""
        video = self.videos.find_one({"_id": video_id})
        # Handle case where video exists but has no metadata
        return video.get("metadata") if video else None

    def get_annotations(
        self,
        video_id: str,
        annotation_type: t.Optional[str] = None,
        frame: t.Optional[int] = None,
    ) -> list[dict[str, t.Any]] | None:
        """Get annotations for a video, optionally filtered by type and frame."""
        # First check if video exists
        if not self.videos.find_one({"_id": video_id}):
            return None

        query = {"video_id": video_id}
        if annotation_type:
            query["type"] = annotation_type
        if frame is not None:
            query["frame"] = frame
        anns = list(self.annotations.find(query))
        output = dict()

        # outputs tiered by type and then frame if exists
        for ann in anns:
            ann_dict = ann["value"]
            if "annotation_type" not in ann_dict:
                ann_dict["annotation_type"] = ann["type"]
            if ann["frame"] is not None:
                if ann["type"] not in output:
                    output[ann["type"]] = defaultdict(list)
                output[ann["type"]][ann["frame"]].append(Annotation.from_dict(ann_dict))
            else:
                if ann["type"] not in output:
                    output[ann["type"]] = list()
                output[ann["type"]].append(Annotation.from_dict(ann_dict))
        return output

    def add_frame_annotations(
        self, video_id: str, frame: int, annotations: list[Annotation]
    ) -> list[str]:
        """Add multiple annotations for a specific frame."""
        annotation_ids = []
        for ann in annotations:
            ann_id = self.add_annotation(
                video_id=video_id,
                key=f"frame_{frame}_annotation",
                value=ann,
                annotation_type="detection",
                frame=frame,
            )
            annotation_ids.append(ann_id)
        return annotation_ids

    def get_database_stats(self) -> dict[str, t.Any]:
        """Get statistics about the current state of the database.

        Returns:
            Dict containing:
            - Total number of videos
            - Total number of unique annotated frames (across all videos)
            - Total number of annotations
            - Annotations per type
            - Stats per dataset (videos, frames, annotations)
        """
        tic = time.time()
        pipeline = [
            {
                "$group": {
                    "_id": {"video_id": "$video_id", "frame": "$frame"},
                    "count": {"$sum": 1},
                }
            },
            {"$count": "total"},
        ]

        total_frames_result = list(self.annotations.aggregate(pipeline))
        total_frames = total_frames_result[0]["total"] if total_frames_result else 0

        stats: dict[str, t.Any] = {
            "total_videos": self.videos.count_documents({}),
            "total_annotated_frames": total_frames,
            "total_annotations": self.annotations.count_documents({}),
            "annotations_by_type": {},
            "per_dataset": {},
        }

        # Get counts per annotation type
        pipeline = [{"$group": {"_id": "$type", "count": {"$sum": 1}}}]
        for type_stat in self.annotations.aggregate(pipeline):
            stats["annotations_by_type"][type_stat["_id"]] = type_stat["count"]

        # Add per-dataset statistics
        pipeline = [
            {
                "$lookup": {
                    "from": "videos",
                    "localField": "video_id",
                    "foreignField": "_id",
                    "as": "video_info",
                }
            },
            {"$unwind": "$video_info"},
            {
                "$group": {
                    "_id": "$video_info.metadata.dataset_filename",
                    "total_annotations": {"$sum": 1},
                    "unique_videos": {"$addToSet": "$video_id"},
                    "unique_frames": {
                        "$addToSet": {"video": "$video_id", "frame": "$frame"}
                    },
                }
            },
        ]

        for dataset_stat in self.annotations.aggregate(pipeline):
            dataset_filename = dataset_stat["_id"]
            if dataset_filename:  # Skip if dataset_filename is None
                stats["per_dataset"][dataset_filename] = {
                    "total_videos": len(dataset_stat["unique_videos"]),
                    "total_frames": len(dataset_stat["unique_frames"]),
                    "total_annotations": dataset_stat["total_annotations"],
                }

        toc = time.time()
        print(f"Time to get annotation database stats: {toc - tic:.2f} seconds")
        return stats

    def delete_annotations(
        self, video_id: str, annotation_type: t.Optional[str] = None
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

    def add_video_with_annotations(
        self,
        dataset_filename: str,
        video_path: str,
        frames: list[np.ndarray],
        frame_indices: list[int],
        annotations: list[list[Annotation]],
        label_str: str,
    ) -> str:
        """Add a video and its frame annotations to the database.

        Args:
            dataset_filename: Name of the dataset
            video_path: Path to the video file
            frames: List of video frames
            frame_indices: List of frame indices
            annotations: List of annotations per frame
            label_str: String containing object labels used for detection

        Returns:
            video_id: Unique identifier for the video
        """
        # Create video metadata
        video_id = get_video_id(dataset_filename, video_path)
        metadata = {
            "dataset_filename": dataset_filename,
            "video_path": video_path,
            "num_frames": len(frames),
            "frame_indices": frame_indices,
        }

        # Add video metadata
        self.add_video(video_id, metadata)

        # Add grounding string annotation
        self.add_annotation(
            video_id=video_id,
            key="nouns",
            value=Annotation(description=label_str, annotation_type="grounding_string"),
            annotation_type="grounding_string",
            frame=None,
        )

        # Add annotations for each frame
        for frame_idx, frame_annotations in enumerate(annotations):
            self.add_frame_annotations(
                video_id=video_id,
                frame=frame_indices[frame_idx],
                annotations=frame_annotations,
            )

        return video_id

    def delete_video_and_annotations(self, video_id: str) -> None:
        """Delete a video and its annotations."""
        self.delete_video(video_id)
        self.delete_annotations(video_id)

    def peek_database(self, limit: int = 5) -> dict[str, t.Any]:
        """Preview entries from both videos and annotations collections.

        Args:
            limit: Number of entries to return from each collection

        Returns:
            Dictionary containing sample entries from both collections and examples per annotation type
        """
        # Get sample entries
        samples = {
            "videos": list(self.videos.find().limit(limit)),
            "annotations": list(self.annotations.find().limit(limit)),
        }

        # Get examples for each annotation type
        pipeline = [
            {"$group": {"_id": "$type", "examples": {"$push": "$$ROOT"}}},
            {"$project": {"examples": {"$slice": ["$examples", limit]}}},
        ]

        type_examples = list(self.annotations.aggregate(pipeline))
        samples["annotation_types"] = {
            stat["_id"]: stat["examples"] for stat in type_examples
        }

        return samples

    def get_video_ids(self) -> list[str]:
        return list(self.videos.find().distinct("_id"))

    def get_annotation_ids(self) -> list[str]:
        return list(self.annotations.find().distinct("video_id"))


if __name__ == "__main__":
    db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)

    # Preview database contents
    stats = db.get_database_stats()
    preview = db.peek_database(limit=10000)

    breakpoint()
