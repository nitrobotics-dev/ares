# Release Scripts

This directory contains scripts for managing ARES quick-start [data and databases on the Hugging Face Hub](https://huggingface.co/datasets/jacobphillips99/ares-data). The data includes the StructuredDatabase, AnnotationDatabase, EmbeddingDatabase, and videos spanning 5000 Open X-Embodiment demonstations, structured as follows:

- `robot_data.db`: StructuredDatabase SQLite database containing structured robot data
- `embedding_data`: IndexManager for the EmbeddingDatabase containing FAISS indexes
- `annotation_mongodump`: MongoDB dump of the AnnotationDatabase
- `videos`: Collection of robot demonstration videos and frames

## Prerequisites

- Set the `HUGGINGFACE_API_KEY` environment variable with your Hugging Face access token
- MongoDB installed and running locally (for pull/restore operations)
- MongoDB database tools (mongorestore/mongodump) installed on your system
  - **Ubuntu/Debian**: `sudo apt-get install mongodb-database-tools`
  - **macOS**: `brew install mongodb/brew/mongodb-database-tools`
  - **Windows**: Download and install from the [MongoDB Download Center](https://www.mongodb.com/try/download/database-tools)
  - **From source**: Follow instructions at [MongoDB GitHub repository](https://github.com/mongodb/mongo-tools)
- 

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