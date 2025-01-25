# ARES Training Setup

This directory contains the (MOCK) training pipeline for using ARES. The process is split into two main steps:
Note: this is a mock pipeline demonstrating how the ARES platform could be used for training.
## 1. Data Preprocessing

The first step uses `preprocess.py` to:
- Load a dataframe of desired rollout IDs to train on
- Query the database engine to fetch full rollout data
- Preload annotations from annotations_db for the specified key
- Save everything as a parquet file for efficient loading

```bash
python preprocess.py --ids-path path/to/ids.csv --output-path data/processed.parquet --annotation-key detection
```

## 2. Training

The second step uses `train.py` which provides:
- Custom PyTorch Dataset that loads the parquet file
- Efficient DataLoader for batching
- Constructs Rollout objects and annotation dictionaries

```bash
python train.py --data-path data/processed.parquet
```

## Data Structure

The preprocessed parquet file contains:
- All fields from the original rollouts
- Preloaded annotations under the 'annotations' column
- Frame indices and metadata needed for training

The PyTorch Dataset returns tuples of:
- Rollout object
- Dictionary of annotations
- Additional metadata needed for training
