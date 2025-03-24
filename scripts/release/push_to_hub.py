"""
Helper script to push relevant data and databases to the HF Hub for distribution.
"""

import os
import tarfile
from huggingface_hub import (
    HfApi,
    upload_folder,
    upload_file,
)
import subprocess
from ares.constants import ARES_DATA_DIR
from ares.databases.structured_database import ROBOT_DB_NAME
from ares.databases.embedding_database import EMBEDDING_DB_NAME

HF_REPO = "jacobphillips99/ares-data"
ANNOTATION_DB_BACKUP_NAME = "annotation_mongodump"

UPLOAD_CONFIGS = [
    # {"type": "file", "source": ROBOT_DB_NAME, "dest": ROBOT_DB_NAME, "size": "small"},
    # {
    #     "type": "folder",
    #     "source": ANNOTATION_DB_BACKUP_NAME,
    #     "dest": ANNOTATION_DB_BACKUP_NAME,
    #     "size": "small"
    # },
    # {"type": "folder", "source": EMBEDDING_DB_NAME, "dest": EMBEDDING_DB_NAME, "size": "small"},
    {"type": "folder", "source": "videos", "dest": "videos", "size": "large"},
]


def backup_mongodb() -> None:
    """Create a MongoDB backup in the correct data directory. Note: may have to change the path depending on your OS."""
    backup_path = os.path.join(ARES_DATA_DIR, ANNOTATION_DB_BACKUP_NAME).replace(
        "/workspaces/", ""
    )
    print(
        f"Please run this command in your shell from root directory outside the container (or press c to skip): \n\nmongodump --uri=mongodb://localhost:27017 --out={backup_path}\n\n"
    )
    breakpoint()
    print("MongoDB backup complete")


def create_tarfile(source_dir: str, output_filename: str) -> str:
    """Create a tar archive of the given directory.

    Args:
        source_dir: Path to directory to tar
        output_filename: Name of the output tar file

    Returns:
        str: Path to the created tar file
    """
    output_path = os.path.join(ARES_DATA_DIR, output_filename)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(os.path.join(ARES_DATA_DIR, source_dir), arcname=source_dir)
    return output_path


def upload_video_dataset(dataset: str, source_path: str, token: str) -> None:
    """Upload a single video dataset to HuggingFace.

    Args:
        dataset: Name of the dataset directory
        source_path: Base path to the videos directory
        token: HuggingFace API token
    """
    tar_filename = f"videos_{dataset}.tar.gz"
    dataset_path = os.path.join("videos", dataset)
    tar_path = create_tarfile(dataset_path, tar_filename)

    print(f"Uploading {tar_filename}...")
    upload_file(
        path_or_fileobj=tar_path,
        path_in_repo=f"videos/{tar_filename}",
        repo_id=HF_REPO,
        repo_type="dataset",
        token=token,
    )

    # Clean up tar file after upload
    os.remove(tar_path)
    print(f"Successfully uploaded {dataset}")


def upload_to_hf(config: dict[str, str], token: str) -> None:
    source_path = os.path.join(ARES_DATA_DIR, config["source"])

    # If it's the videos folder, handle each dataset separately
    if config["type"] == "folder" and config["source"] == "videos":
        # Get all subdirectories in the videos folder
        video_datasets = [
            d
            for d in os.listdir(source_path)
            if os.path.isdir(os.path.join(source_path, d))
        ]

        print(f"Found {len(video_datasets)} datasets to upload")
        for idx, dataset in enumerate(video_datasets, 1):
            print(f"\nProcessing dataset {idx}/{len(video_datasets)}: {dataset}")
            upload_video_dataset(dataset, source_path, token)
    else:
        if config["type"] == "file":
            upload_file(
                path_or_fileobj=source_path,
                path_in_repo=config["dest"],
                repo_id=HF_REPO,
                repo_type="dataset",
                token=token,
            )
        else:
            upload_folder(
                folder_path=source_path,
                path_in_repo=config["dest"],
                repo_id=HF_REPO,
                repo_type="dataset",
                token=token,
            )
    print(f"Uploaded {config['source']} to {config['dest']}")


if __name__ == "__main__":
    # create the dataset repo if it doesn't exist
    token = os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("Please set HUGGINGFACE_API_KEY environment variable")

    api = HfApi(token=token)

    # # check to see if our HF repo exists
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

    print("upload complete.")
