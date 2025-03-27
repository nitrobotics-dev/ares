#!/bin/bash
# Script to download data from Hugging Face hub, covering roughly 5000 rollouts from the Open X-Embodiment project. Includes:
#   - robot_data.db (StructuredDatabase SQLite database)
#   - embedding_data (EmbeddingDatabase IndexManager)
#   - annotation_mongodump (AnnotationDatabase MongoDB dump)
#   - videos (videos and frames)
# Usage (from root directory): ./scripts/release/pull_from_hub.sh [output_directory]
#
# Note: This script downloads both the databases and the videos, which are separated into different tar files per-dataset.
# You may need to install the `mongo-db-tools` to restore the MongoDB backup: https://www.mongodb.com/try/download/database-tools

set -euo pipefail

OUTDIR="${1:-$HOME/ares/data}"
HF_REPO="jacobphillips99/ares-data"
HF_BASE="https://huggingface.co/datasets/$HF_REPO"
HF_DOWNLOAD="$HF_BASE/resolve/main"

# Check for HF token
if [ -z "${HUGGINGFACE_API_KEY:-}" ]; then
    echo "Error: HUGGINGFACE_API_KEY environment variable is not set"
    exit 1
fi

mkdir -p "$OUTDIR"

# download and restore the StructuredDatabase
echo "downloading robot_data.db..."
curl -L -H "Authorization: Bearer $HUGGINGFACE_API_KEY" "$HF_DOWNLOAD/robot_data.db" -o "$OUTDIR/robot_data.db"

# download and restore the EmbeddingDatabase
echo "downloading embedding_data..."
curl -L -H "Authorization: Bearer $HUGGINGFACE_API_KEY" "$HF_DOWNLOAD/embedding_data.tar.gz" -o "$OUTDIR/embedding_data.tar.gz"
tar -xzf "$OUTDIR/embedding_data.tar.gz" -C "$OUTDIR"
rm "$OUTDIR/embedding_data.tar.gz"

# download MongoDB AnnotationDatabase
echo "downloading annotation_mongodump..."
curl -L -H "Authorization: Bearer $HUGGINGFACE_API_KEY" "$HF_DOWNLOAD/annotation_mongodump.tar.gz" -o "$OUTDIR/annotation_mongodump.tar.gz"
tar -xzf "$OUTDIR/annotation_mongodump.tar.gz" -C "$OUTDIR"
rm "$OUTDIR/annotation_mongodump.tar.gz"
# restore MongoDB backup
echo "restoring mongo backup..."
# check if mongo-db-tools is installed
if ! command -v mongorestore &> /dev/null; then
    echo "Error: mongorestore could not be found"
    exit 1
fi
mongorestore --uri="mongodb://localhost:27017" "$OUTDIR/annotation_mongodump"

# Get list of video dataset tars from HF hub and download, unpack, and remove each tar file
echo "downloading videos..."
mkdir -p "$OUTDIR/videos"
echo "fetching video datasets..."
API_URL="https://huggingface.co/api/datasets/$HF_REPO/tree/main/videos"
echo "Fetching from: $API_URL"
for tar_file in $(curl -s -H "Authorization: Bearer $HUGGINGFACE_API_KEY" "$API_URL" | grep -o '"path":"videos/[^"]*\.tar\.gz"' | cut -d'"' -f4 | cut -d'/' -f2); do
    echo "Found tar file: $tar_file"
    echo "downloading and extracting $tar_file..."
    curl -L -H "Authorization: Bearer $HUGGINGFACE_API_KEY" "$HF_DOWNLOAD/videos/$tar_file" -o "$OUTDIR/videos/$tar_file"
    tar -xzf "$OUTDIR/videos/$tar_file" -C "$OUTDIR"
    rm "$OUTDIR/videos/$tar_file"
done

echo "done."