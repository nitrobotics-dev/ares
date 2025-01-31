"""
Helper function to orchestrate annotation over datasets in the structured database and send annotations into the annotation database.
See `ares.annotating.annotating_fn.py` for the AnnotatingFn object that gets fulfilled for different annotation methods.
"""

import os
import pickle
import time

from ares.annotating.annotating_base import (
    ErrorResult,
    ResultTracker,
    setup_rollouts_from_sources,
)
from ares.annotating.annotating_fn import AnnotatingFn
from ares.constants import ANNOTATION_OUTER_BATCH_SIZE
from ares.databases.annotation_database import AnnotationDatabase
from ares.databases.structured_database import RolloutSQLModel, setup_database


def orchestrate_annotating(
    engine_path: str,
    ann_db_path: str,
    annotating_fn: AnnotatingFn,
    dataset_filename: str | None = None,
    split: str | None = None,
    rollout_ids: list[str] | None = None,
    outer_batch_size: int = ANNOTATION_OUTER_BATCH_SIZE,  # RAM limits number of concurrent rollouts formatted into requests
    ids_path: str = None,  # Path to ids to load; may be failed IDs from previous run
    annotating_kwargs: dict | None = None,
    failures_path: str | None = None,
) -> tuple[ResultTracker, list[ErrorResult]]:
    """
    Main function to run annotations, whether local, API-driven, or on Modal.

    Args:
        engine_path (str): Path to the database engine.
        ann_db_path (str): Path to the annotation database.
        annotating_fn (Callable): Function to run annotation.
        dataset_filename (str, optional): Dataset filename.
        split (str, optional): Data split.
        rollout_ids (list[str], optional): Specific rollout IDs to process.
        outer_batch_size (int, optional): Batch size for processing rollouts.
        ids_path (str, optional): Path to ids to load.
        annotating_kwargs (dict, optional): Additional keyword arguments for annotating_fn.
    """
    assert (
        dataset_filename is not None or ids_path is not None or rollout_ids is not None
    ), f"Must provide either dataset_filename, ids_path, or rollout_ids. Received: dataset_filename={dataset_filename}, ids_path={ids_path}, rollout_ids={rollout_ids}"

    annotating_kwargs = annotating_kwargs or {}
    # Initialize databases
    ann_db = AnnotationDatabase(connection_string=ann_db_path)
    engine = setup_database(RolloutSQLModel, path=engine_path)
    rollouts = setup_rollouts_from_sources(
        engine, rollout_ids, ids_path, dataset_filename, split
    )
    print(f"\n\nFound {len(rollouts)} total rollouts\n\n")
    if not rollouts:
        print(
            f"No rollouts found for dataset filename {dataset_filename}, retry failed path {ids_path}"
        )
        return

    tic = time.time()
    overall_tracker, overall_failures = annotating_fn(
        rollouts, ann_db, outer_batch_size, **annotating_kwargs
    )
    print(f"\n\nFailures: {overall_failures}\n\n")

    # Write failures to file to retry
    if failures_path and len(overall_failures) > 0:
        os.makedirs(os.path.dirname(failures_path), exist_ok=True)
        print(f"Writing failures to {failures_path}")
        with open(failures_path, "wb") as f:
            pickle.dump(overall_failures, f)

    print("Time taken:", time.time() - tic)
    print(f"\n\n")
    overall_tracker.print_stats()
    print(f"\nNumber of failures: {len(overall_failures)}")
    return overall_tracker, overall_failures
