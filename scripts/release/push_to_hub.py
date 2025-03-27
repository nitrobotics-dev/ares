"""
Helper script to push relevant data and databases to the HF Hub for distribution.
All folders are uploaded as tar.gz files.
"""

import os
import tarfile
from huggingface_hub import HfApi, upload_file
from ares.constants import ARES_DATA_DIR
from ares.databases.structured_database import ROBOT_DB_NAME
from ares.databases.embedding_database import EMBEDDING_DB_NAME

# insert your repo name here!
HF_REPO = "jacobphillips99/ares-data"
ANNOTATION_DB_BACKUP_NAME = "annotation_mongodump"

UPLOAD_CONFIGS = [
    # {"type": "file", "source": ROBOT_DB_NAME, "dest": ROBOT_DB_NAME},
    {
        "type": "folder",
        "source": ANNOTATION_DB_BACKUP_NAME,
        "dest": ANNOTATION_DB_BACKUP_NAME,
    },
    {"type": "folder", "source": EMBEDDING_DB_NAME, "dest": EMBEDDING_DB_NAME},
    # {"type": "folder", "source": "videos", "dest": "videos"},
]


def backup_mongodb() -> None:
    """Create a MongoDB backup in the data directory."""
    backup_path = os.path.join(ARES_DATA_DIR, ANNOTATION_DB_BACKUP_NAME).replace(
        "/workspaces/", ""
    )
    print(
        f"Please run this command in your shell from root directory outside the container (or press c to skip): \n\n"
        f"mongodump --uri=mongodb://localhost:27017 --out={backup_path}\n\n"
    )
    breakpoint()
    print("MongoDB backup complete")


def create_tarfile(source_dir: str, output_filename: str) -> str:
    """Create a tar archive of the given directory."""
    output_path = os.path.join(ARES_DATA_DIR, output_filename)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(
            os.path.join(ARES_DATA_DIR, source_dir),
            arcname=os.path.basename(source_dir),
        )
    return output_path


def upload_to_hf(config: dict[str, str], token: str) -> None:
    source_path = os.path.join(ARES_DATA_DIR, config["source"])

    if config["type"] == "file":
        # Direct file upload
        upload_file(
            path_or_fileobj=source_path,
            path_in_repo=config["dest"],
            repo_id=HF_REPO,
            repo_type="dataset",
            token=token,
        )
    else:
        # For folders, create tar.gz and upload
        if config["source"] == "videos":
            # Handle videos directory specially - tar each subdirectory
            video_datasets = [
                d
                for d in os.listdir(source_path)
                if os.path.isdir(os.path.join(source_path, d))
            ]

            print(f"Found {len(video_datasets)} video datasets to upload")
            for idx, dataset in enumerate(video_datasets, 1):
                print(f"\nProcessing dataset {idx}/{len(video_datasets)}: {dataset}")
                dataset_path = os.path.join("videos", dataset)
                tar_filename = f"videos_{dataset}.tar.gz"
                tar_path = create_tarfile(dataset_path, tar_filename)

                upload_file(
                    path_or_fileobj=tar_path,
                    path_in_repo=f"videos/{tar_filename}",
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    token=token,
                )
                os.remove(tar_path)
        else:
            # For other folders, create single tar.gz
            tar_filename = f"{config['source']}.tar.gz"
            tar_path = create_tarfile(config["source"], tar_filename)

            upload_file(
                path_or_fileobj=tar_path,
                path_in_repo=tar_filename,
                repo_id=HF_REPO,
                repo_type="dataset",
                token=token,
            )
            os.remove(tar_path)

    print(f"Uploaded {config['source']}")


if __name__ == "__main__":
    token = os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("Please set HUGGINGFACE_API_KEY environment variable")

    api = HfApi(token=token)

    # Check if HF repo exists
    matching_datasets = api.list_datasets(search=HF_REPO)
    if not any(d.id == HF_REPO for d in matching_datasets):
        print(f"Creating repo {HF_REPO}")
        api.create_repo(repo_id=HF_REPO, repo_type="dataset")
    else:
        print(f"Repo {HF_REPO} already exists")

    # Create MongoDB backup
    # backup_mongodb()

    # Upload each item in the upload config
    for item in UPLOAD_CONFIGS:
        upload_to_hf(item, token)

    print("Upload complete.")
