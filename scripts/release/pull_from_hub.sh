#!/bin/bash
set -euo pipefail

OUTDIR="${1:-$HOME/ares/data}"
HF_REPO="jacobphillips99/ares-data"
HF_BASE="https://huggingface.co/datasets/$HF_REPO"
HF_DOWNLOAD="$HF_BASE/resolve/main"

mkdir -p "$OUTDIR"

echo "downloading robot_data.db..."
curl -L "$HF_DOWNLOAD/robot_data.db" -o "$OUTDIR/robot_data.db"

echo "downloading embedding_data..."
mkdir -p "$OUTDIR/embedding_data"
curl -s "$HF_DOWNLOAD/embedding_data/index.faiss" -o "$OUTDIR/embedding_data/index.faiss"
curl -s "$HF_DOWNLOAD/embedding_data/index.json" -o "$OUTDIR/embedding_data/index.json"
# add more as needed

echo "downloading annotation_mongodump..."
mkdir -p "$OUTDIR/annotation_mongodump"
curl -s "$HF_DOWNLOAD/annotation_mongodump/collection.bson" -o "$OUTDIR/annotation_mongodump/collection.bson"
curl -s "$HF_DOWNLOAD/annotation_mongodump/collection.metadata.json" -o "$OUTDIR/annotation_mongodump/collection.metadata.json"
# add more if multiple collections

echo "restoring mongo backup..."
mongorestore --uri="mongodb://localhost:27017" "$OUTDIR/annotation_mongodump"

echo "downloading videos..."
mkdir -p "$OUTDIR/videos"

# Get list of video dataset tars from HF hub
echo "fetching video datasets..."
for tar_file in $(curl -s "$HF_BASE/tree/main/videos" | grep -o 'videos_[^"]*\.tar\.gz'); do
    echo "downloading and extracting $tar_file..."
    curl -L "$HF_DOWNLOAD/videos/$tar_file" -o "$OUTDIR/videos/$tar_file"
    tar -xzf "$OUTDIR/videos/$tar_file" -C "$OUTDIR"
    rm "$OUTDIR/videos/$tar_file"  # Clean up tar file after extraction
done

echo "done."