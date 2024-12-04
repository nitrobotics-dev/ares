import os

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from ares.clustering import cluster_embeddings
from ares.databases.embedding_database import (
    TEST_EMBEDDING_DB_PATH,
    FaissIndex,
    IndexManager,
)
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)


def load_cached_embeddings(
    tmp_dump_dir: str, index_name: str, stored_embeddings: np.ndarray
) -> tuple | None:
    """Try to load cached embeddings and clusters for given index"""
    embeddings_path = os.path.join(tmp_dump_dir, f"{index_name}_embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, f"{index_name}_clusters.npz")

    if not (os.path.exists(embeddings_path) and os.path.exists(clusters_path)):
        return None

    loaded_embeddings = np.load(embeddings_path)
    if not (
        len(loaded_embeddings) == len(stored_embeddings)
        and np.allclose(loaded_embeddings, stored_embeddings)
    ):
        import pdb

        pdb.set_trace()  # Break if cached data doesn't match
        return None

    # Valid cached data found - load everything
    clusters_data = np.load(clusters_path)
    return (
        loaded_embeddings,
        clusters_data["reduced"],
        clusters_data["labels"],
    )


def save_embeddings(
    tmp_dump_dir: str,
    index_name: str,
    embeddings: np.ndarray,
    reduced: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Save embeddings and clusters to disk"""
    embeddings_path = os.path.join(tmp_dump_dir, f"{index_name}_embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, f"{index_name}_clusters.npz")

    np.save(embeddings_path, embeddings)
    np.savez(clusters_path, reduced=reduced, labels=labels)


def store_in_session(
    index_name: str,
    embeddings: np.ndarray,
    reduced: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Store embeddings and clusters in session state"""
    st.session_state[f"{index_name}_embeddings"] = embeddings
    st.session_state[f"{index_name}_reduced"] = reduced
    st.session_state[f"{index_name}_labels"] = labels


def initialize_data(tmp_dump_dir: str) -> None:
    """Initialize database connection and load/create embeddings"""
    # Initialize database and session if needed
    if "ENGINE" not in st.session_state or "SESSION" not in st.session_state:
        engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
        sess = Session(engine)
        st.session_state.ENGINE = engine
        st.session_state.SESSION = sess

    # Initialize index manager
    index_manager = IndexManager(
        base_dir=TEST_EMBEDDING_DB_PATH, index_class=FaissIndex
    )
    st.session_state.INDEX_MANAGER = index_manager

    # Get all vectors and their IDs
    all_data = index_manager.get_all_matrices()
    st.session_state.all_vecs = {
        name: data["arrays"] for name, data in all_data.items()
    }
    st.session_state.all_ids = {name: data["ids"] for name, data in all_data.items()}

    # Create tmp directory if it doesn't exist
    os.makedirs(tmp_dump_dir, exist_ok=True)

    # Process each index type
    for index_name in ["task", "description"]:
        stored_embeddings = index_manager.indices[index_name].get_all_vectors()
        stored_ids = index_manager.indices[index_name].get_all_ids()  # Get IDs

        # Try loading from cache first
        cached_data = load_cached_embeddings(
            tmp_dump_dir, index_name, stored_embeddings
        )
        if cached_data is not None:
            embeddings, reduced, labels = cached_data
        else:
            # Create new embeddings and clusters
            embeddings = stored_embeddings
            reduced, labels, _ = cluster_embeddings(embeddings)
            save_embeddings(tmp_dump_dir, index_name, embeddings, reduced, labels)

        # Store in session state
        store_in_session(index_name, embeddings, reduced, labels)
