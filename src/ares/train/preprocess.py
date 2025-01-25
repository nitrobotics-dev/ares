"""
Mock simple script to preprocess rollouts and annotations into the train format. 
This is a mock to show how to construct a preprocessed artifact to conduct training using the ARES platform.
We construct a training artifact in order to derisk loading errors and avoid massive database queries during training.

See ares/train/README.md for more details and ares/train/train.py for how to use the preprocessed artifact.

Preprocess assums a list of IDs have been selected via curation on the ARES platform.

Example usage:
    python preprocess.py --ids_df_path path/to/ids.csv --extra_info_cols col1 --extra_info_cols col2 --output_path output.parquet
    
    # Or with shorter syntax:
    python preprocess.py --ids_df_path path/to/ids.csv -extra_info_cols col1 -extra_info_cols col2 --output_path output.parquet
"""

import click
from tqdm import tqdm

from ares.databases.annotation_database import ANNOTATION_DB_PATH
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    get_rollouts_by_ids,
    setup_database,
)


@click.command()
@click.option("--ids_df_path", type=str, help="Path to CSV file containing rollout IDs")
@click.option(
    "--extra_info_cols",
    type=str,
    multiple=True,
    help="Extra info columns to collect annotations for. Can be specified multiple times",
)
@click.option("--output_path", type=str, help="Path to save the preprocessed artifact")
def preprocess(ids_df_path: str, extra_info_cols: list[str], output_path: str):
    ids_df = pd.read_csv(ids_df_path)

    # collect rollouts
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    rollout_df = get_rollouts_by_ids(engine, ids_df.id.tolist(), return_df=True)

    # collect annotations from annotation database via extra_info_cols
    ann_db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)
    extra_info_cols_to_anns = defaultdict(list)
    for col in extra_info_cols:
        print(f"Collecting annotations for extra info col{col}")
        for i, row in tqdm(
            rollout_df.iterrows(), desc=f"Collecting annotations for {col}"
        ):
            video_path = str(Path(row["filename"]).with_suffix(".mp4"))
            video_id = f"{row['dataset_filename']}/{video_path}"
            anns = ann_db.get_annotations(video_id, annotation_type=col)
            extra_info_cols_to_anns[col].append(json.dumps(anns))

    # construct train df
    train_df = pd.DataFrame(rollout_df)
    for col in extra_info_cols:
        train_df[col] = extra_info_cols_to_anns[col]

    # save to parquet
    print(f"Saving to {output_path}")
    train_df.to_parquet(output_path)
