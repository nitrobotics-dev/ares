from tqdm import tqdm

from ares.configs.annotations import Annotation
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase


def migrate():
    db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)

    # Get all videos
    video_ids = db.get_video_ids()
    print(f"Found {len(video_ids)} videos to process")

    for video_id in tqdm(video_ids):
        # Get annotations for this video
        annotations = db.get_annotations(video_id, annotation_type="success_criteria")
        if not annotations or "success_criteria" not in annotations:
            print(f"Skipping {video_id} - no success_criteria annotations found")
            continue

        # Get the original string from the list of annotations
        string_value = None
        for ann in annotations["success_criteria"]:
            if isinstance(ann, str):
                string_value = ann
                break

        if string_value is None:
            print(f"Skipping {video_id} - no string annotation found")
            continue

        # Create new Annotation object with the original string
        annotation_obj = Annotation(
            description=string_value, annotation_type="success_criteria"
        )

        # Delete all existing success_criteria annotations for this video
        db.delete_annotations(video_id, annotation_type="success_criteria")

        # Add the new annotation
        db.add_annotation(
            video_id=video_id,
            key="string",
            value=annotation_obj,
            annotation_type="success_criteria",
            frame=None,
        )

        print(f"Processed {video_id}")

    print("Migration complete!")


if __name__ == "__main__":
    migrate()
