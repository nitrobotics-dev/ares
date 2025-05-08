---
license: apache-2.0
language:
- en
size_categories:
  - 1k<n<10k
task_categories:
  - other
pretty_name: ares trajectories
tags:
  - robotics
  - trajectories
  - ares
  - lfd
configs:
  - config_name: default
    data_files:
    - split: train
      path: robot_data.parquet
---

# Dataset Card for ARES Data

**Author**: Jacob Phillips

[ARES Github Repository](https://github.com/jacobphillips99/ares)

![ARES System Diagram](https://raw.githubusercontent.com/jacobphillips99/ares/main/assets/ares_system_diagram.png)


## Dataset Description

This dataset contains the data and databases for the ARES project. It is a collection of 5000 rollouts from the Open X-Embodiment dataset which have been ingested into the ARES platform. The data is stored in three databases: 
- SQL database: a SQLite table containing structured information like length, success rates, and esimates provided by a VLM (such as description, focus objects, environment, etc.). The table is stored as `robot_data.db` and also dumped as a parquet file `robot_data.parquet`.
- Embedding database: a collection of FAISS indexes holding state and action embeddings for each rollout, as well as embeddings for descriptions and task instructions. The database is stored as a tar file `embedding_data.tar.gz`.
- Annotation database: a MongoDB collection containing annotations for each rollout, including the rollout name, the rollout data, and the annotations. The database is stored as a tar file `annotation_mongodump.tar.gz`.

## Usage

The databases can be downloaded and restored using the [`release scripts`](https://github.com/jacobphillips99/ares/tree/main/scripts/release) in the ARES repository. First, follow the instructions in the [README](https://github.com/jacobphillips99/ares/tree/main/README.md) to set up the environment. Then, run the following command to download and restore the databases:

```bash
cd scripts/release
./pull_from_hub.sh
```

Alternatively, the parquet file can be used natively in `transformers` with:

```python
from datasets import load_dataset

dataset = load_dataset("jacobphillips99/ares-data")
```