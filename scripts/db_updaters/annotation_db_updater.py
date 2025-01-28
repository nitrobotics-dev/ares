from tqdm import tqdm

from ares.configs.annotations import Annotation
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase


def migrate_nouns_to_annotation_objects():
    """Migrate 'nouns' annotations from plain strings to Annotation objects."""
    db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)

    # Get all videos
    video_ids = db.get_video_ids()
    print(f"Found {len(video_ids)} videos to process")

    for video_id in tqdm(video_ids):
        # Get annotations for this video
        annotations = db.get_annotations(video_id, annotation_type="grounding_string")
        if not annotations or "grounding_string" not in annotations:
            print(f"Skipping {video_id} - no grounding_string annotations found")
            continue

        # Get the original string from the list of annotations
        string_value = None
        for ann in annotations["grounding_string"]:
            if isinstance(ann, str):
                string_value = ann
                break

        if string_value is None:
            print(f"Skipping {video_id} - no string annotation found")
            continue

        # Create new Annotation object with the original string
        annotation_obj = Annotation(
            description=string_value, annotation_type="grounding_string"
        )

        # Delete all existing grounding_string annotations for this video
        db.delete_annotations(video_id, annotation_type="grounding_string")

        # Add the new annotation
        db.add_annotation(
            video_id=video_id,
            key="nouns",
            value=annotation_obj.to_dict(),
            annotation_type="grounding_string",
            frame=None,
        )

        print(f"Processed {video_id}")

    print("Migration complete!")


if __name__ == "__main__":
    migrate_nouns_to_annotation_objects()
