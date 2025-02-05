"""
Mock simple script to preprocess rollouts and annotations into the train format. 
This is a mock to show how to construct a preprocessed artifact to conduct training using the ARES platform.
We construct a training artifact in order to derisk loading errors and avoid massive database queries during training.

See ares/train/README.md for more details and ares/train/train.py for how to use the preprocessed artifact.

Preprocess assumes a list of IDs have been selected via curation on the ARES platform.
Extra info cols could be "grounding_string", "detections", "embodied_cot", etc.

Example usage:
    python preprocess.py --ids_df_path path/to/ids.csv --extra_info_cols col1 --extra_info_cols col2 --output_path output.parquet
    
    # Or with shorter syntax:
    python preprocess.py --ids_df_path path/to/ids.csv -extra_info_cols col1 -extra_info_cols col2 --output_path output.parquet
"""

import json
import os
from collections import defaultdict

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from ares.databases.annotation_database import (
    ANNOTATION_DB_PATH,
    AnnotationDatabase,
    get_video_id,
)
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    get_partial_df,
    get_rollouts_by_ids,
    setup_database,
)


def setup_extra_info_col(
    df: pd.DataFrame, col: str, ann_db: AnnotationDatabase
) -> list[str | None]:
    raw_anns: list[str | None] = []
    for _, row in tqdm(df.iterrows(), desc=f"Collecting annotations for {col}"):
        video_id = get_video_id(row["dataset_filename"], row["filename"])
        anns = ann_db.get_annotations(video_id, annotation_type=col)
        if anns and col in anns:
            raw_anns.append(
                json.dumps(
                    anns[col],
                    default=lambda x: x.__json__() if hasattr(x, "__json__") else x,
                )
            )
        else:
            raw_anns.append(None)

    # Check if all annotations are None -- probably an error
    if all(ann is None for ann in raw_anns):
        raise ValueError(f"No annotations found for column {col}")
    return raw_anns


@click.command()
@click.option("--output-path", type=str, help="Path to save the preprocessed artifact")
@click.option(
    "--ids-df-path",
    type=str,
    help="Path to CSV file containing rollout IDs",
    required=False,
    default=None,
)
@click.option(
    "--extra-info-cols",
    type=str,
    multiple=True,
    help="Extra info columns to collect annotations for. Can be specified multiple times for different columns",
    required=False,
    default=None,
)
def preprocess(
    output_path: str, ids_df_path: str | None, extra_info_cols: list[str] | None
) -> None:
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)

    if ids_df_path:
        ids_df = pd.read_csv(ids_df_path)
    else:
        ids_df = get_partial_df(engine, ["id"])

    if extra_info_cols is None:
        extra_info_cols = []

    # collect rollouts
    rollout_df = get_rollouts_by_ids(engine, ids_df.id.tolist(), return_df=True)

    # collect annotations from annotation database via extra_info_cols
    if extra_info_cols:
        ann_db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)
        extra_info_cols_to_anns = defaultdict(list)
        for col in extra_info_cols:
            raw_anns = setup_extra_info_col(rollout_df, col, ann_db)
            extra_info_cols_to_anns[col] = raw_anns

    # construct train df
    train_df = rollout_df.copy()
    # parquet doesnt like uuids, so we convert to str
    train_df.id = train_df.id.astype(str)
    for col in extra_info_cols:
        train_df[col] = extra_info_cols_to_anns[col]

    # save to parquet
    print(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_df.to_parquet(output_path)


if __name__ == "__main__":
    preprocess()
