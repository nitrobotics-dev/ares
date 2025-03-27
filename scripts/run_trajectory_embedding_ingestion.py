"""
Helper script to ingest a rollout into the embedding database. We ingest the trajectory matrices (such a states and actions) as well as description and task_language_instruction embeddings.
This enables us to perform efficient nearest neighbor search on the embeddings in order to find similar rollouts in the physical or language space.
"""

import time
import typing as t
from collections import defaultdict

import click
import numpy as np
from tqdm import tqdm

from ares.configs.base import Rollout
from ares.databases.embedding_database import (
    EMBEDDING_DB_PATH,
    META_INDEX_NAMES,
    STANDARDIZED_TIME_STEPS,
    FaissIndex,
    IndexManager,
    rollout_to_embedding_pack,
)
from ares.databases.structured_database import (
    RolloutSQLModel,
    get_rollouts_by_ids,
    setup_database,
    setup_rollouts,
)
from ares.models.base import Embedder
from ares.models.shortcuts import get_nomic_embedder


def ingest_trajectory_matrices_from_rollouts_per_dataset(
    rollouts: list[Rollout], index_manager: IndexManager
) -> None:
    # collect all embedding packs for states, actions (pre-existing)
    embedding_packs = []
    for rollout in rollouts:
        embedding_packs.append(rollout_to_embedding_pack(rollout))
    if len(embedding_packs) == 0:
        return

    # collect all the embeddings and get normalizing constants
    for k in embedding_packs[0].keys():
        try:
            packs = [
                pack[k]
                for pack in embedding_packs
                if not all(x is None for x in pack[k])
            ]
            embeddings = np.concatenate(packs) if packs else None
        except Exception as e:
            raise ValueError(f"Error concatenating embeddings for {k}: {e}")

        # Check if embeddings array contains all None values
        if embeddings is None or all(x is None for x in embeddings.flatten()):
            print(f"Skipping {k} - embeddings array contains all None values")
            continue
        print(f"found {embeddings.shape} for {k}; (N,K)")

        # find normalizing constants
        try:
            means = np.mean(embeddings, axis=0)
            stds = np.std(embeddings, axis=0)
        except Exception as e:
            raise ValueError(f"Error finding normalizing constants for {k}: {e}")
        feature_dim = embeddings.shape[1]
        print(f"found means {means.shape} and stds {stds.shape}")
        # setup index if not already existing
        if k not in index_manager.indices.keys():
            index_manager.init_index(
                k,
                feature_dim,
                STANDARDIZED_TIME_STEPS,
                norm_means=means,  # normalize with dimension-specific means
                norm_stds=stds,  # normalize with dimension-specific stds
            )

        # add the embeddings to the index! these will be normalized
        for rollout, pack in tqdm(
            zip(rollouts, embedding_packs), desc=f"Ingesting {k} embeddings"
        ):
            # some datasets do not provide this information
            if isinstance(pack.get(k), np.ndarray) and not all(
                x is None for x in pack[k]
            ):
                index_manager.add_matrix(k, pack[k], str(rollout.id))


def ingest_language_embeddings_from_rollouts_per_dataset(
    rollouts: list[Rollout], index_manager: IndexManager, embedder: Embedder
) -> None:
    feature_dim = embedder.embed("test").shape[0]
    for name in META_INDEX_NAMES:
        for rollout in tqdm(rollouts, desc=f"Ingesting {name} embeddings"):
            if name not in index_manager.indices.keys():
                index_manager.init_index(
                    name,
                    feature_dim,
                    time_steps=1,  # lang embeddings dont get time dimension
                    norm_means=None,  # no need to normalize
                    norm_stds=None,  # no need to normalize
                    extra_metadata={"model": embedder.name},
                )

            inp = rollout.get_nested_attr(name)
            # some datasets do not provide this information
            if inp is None:
                continue
            embedding = embedder.embed(inp)
            index_manager.add_vector(name, embedding, str(rollout.id))


def run_embedding_database_ingestion_per_dataset(
    rollouts: list[Rollout], embedder: Embedder, index_path: str
) -> None:
    index_manager = IndexManager(index_path, index_class=FaissIndex)

    tic = time.time()
    # add the trajectory matrices to the index and get normalizing constants for this dataset
    # (states and actions)
    ingest_trajectory_matrices_from_rollouts_per_dataset(rollouts, index_manager)
    index_manager.save()

    # add task and description embeddings to the index
    # (task, description)
    ingest_language_embeddings_from_rollouts_per_dataset(
        rollouts, index_manager, embedder
    )
    index_manager.save()

    print(f"Embedding database new rollouts: {len(rollouts)}")
    total_time = time.time() - tic
    print(f"Embedding database time: {total_time}")
    print(f"Embedding database mean time: {total_time / len(rollouts)}")
    relevant_metadata = {
        k: v
        for k, v in index_manager.metadata.items()
        if rollouts[0].dataset_formalname in k or k in META_INDEX_NAMES
    }
    print(f"Metadata: {relevant_metadata}")


@click.command()
@click.option(
    "--engine-url",
    type=str,
    required=True,
    help="SQLAlchemy database URL",
)
@click.option(
    "--dataset-formalname",
    type=t.Union[str, None],
    required=False,
    help="Formal name of the dataset to process",
    default=None,
)
@click.option(
    "--from-id-file",
    type=t.Union[str, None],
    required=False,
    help="File containing rollout ids to ingest",
    default=None,
)
@click.option(
    "--index-path",
    type=str,
    required=False,
    help="Path to the index to ingest",
    default=EMBEDDING_DB_PATH,
)
def main(
    engine_url: str,
    dataset_formalname: t.Union[str, None],
    from_id_file: t.Union[str, None],
    index_path: str,
) -> None:
    """Run embedding database ingestion for trajectory data."""
    assert (
        dataset_formalname is not None or from_id_file is not None
    ), "Either dataset_formalname or from_id_file must be provided"
    engine = setup_database(RolloutSQLModel, path=engine_url)
    embedder = get_nomic_embedder()
    if from_id_file is not None:
        with open(from_id_file, "r") as f:
            rollout_ids = [line.strip() for line in f.readlines()]

        rollouts = get_rollouts_by_ids(engine, rollout_ids) if rollout_ids else []
    else:
        rollouts = setup_rollouts(engine, dataset_formalname)

    if len(rollouts) == 0:
        print(
            f"No rollouts found for dataset_formalname: {dataset_formalname}, from_id_file: {from_id_file}"
        )
        return

    dataset_to_rollouts = defaultdict(list)
    for rollout in rollouts:
        dataset_to_rollouts[rollout.dataset_formalname].append(rollout)
    for dataset_formalname, dataset_rollouts in dataset_to_rollouts.items():
        run_embedding_database_ingestion_per_dataset(
            dataset_rollouts, embedder, index_path
        )


if __name__ == "__main__":
    main()
