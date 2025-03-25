# Release Scripts

This directory contains scripts for managing ARES quick-start [data and databases on the Hugging Face Hub](https://huggingface.co/datasets/jacobphillips99/ares-data). The data includes:

- `robot_data.db`: SQLite database containing structured robot data
- `embedding_data`: IndexManager for the EmbeddingDatabase
- `annotation_mongodump`: MongoDB dump of the AnnotationDatabase
- `videos`: Collection of robot demonstration videos and frames

## Prerequisites

- Set the `HUGGINGFACE_API_KEY` environment variable with your Hugging Face access token
- MongoDB installed and running locally (for pull/restore operations)

## Scripts

### pull_from_hub.sh

Downloads and restores the ARES data and databases from the Hugging Face Hub (`jacobphillips99/ares-data`).

```bash
./scripts/release/pull_from_hub.sh [output_directory]
```

- Default output directory: `$HOME/ares/data`
- Downloads and extracts all databases and video datasets
- Automatically restores MongoDB dump to your local MongoDB instance

### push_to_hub.py

Uploads the ARES data and databases to the Hugging Face Hub.

```bash
python scripts/release/push_to_hub.py
```

- Creates tar archives for folder-based data
- Handles video datasets separately, creating individual archives per dataset
- Supports both direct file uploads and directory uploads
- Creates the Hugging Face repository if it doesn't exist