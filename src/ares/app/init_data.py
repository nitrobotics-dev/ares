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
from ares.models.llm import LLM


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
        # this means we have new embeddings
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
    st.session_state.models = dict()
    st.session_state.models["summarizer"] = LLM(
        provider="openai", llm_name="gpt-4o-mini"
    )


def display_state_info():
    # all in a dropdown
    with st.expander("Session State Data Overview"):
        """Display information about data stored in streamlit session state."""
        st.header("Session State Data Overview")

        # Database Info
        st.subheader("Database Connections")
        st.write(
            "- ENGINE:",
            "Connected" if "ENGINE" in st.session_state else "Not Connected",
        )
        st.write(
            "- SESSION:",
            "Connected" if "SESSION" in st.session_state else "Not Connected",
        )

        # Index Manager Info
        st.subheader("Index Manager")
        if "INDEX_MANAGER" in st.session_state:
            index_manager = st.session_state.INDEX_MANAGER
            st.write("Available indices:", list(index_manager.indices.keys()))
            for name, index in index_manager.indices.items():
                st.write(f"\n**{name} Index:**")
                st.write(f"- Feature dimension: {index.feature_dim}")
                st.write(f"- Time steps: {index.time_steps}")
                st.write(f"- Total entries: {index.n_entries}")

        # Vectors and IDs
        st.subheader("Stored Vectors and IDs")
        if "all_vecs" in st.session_state:
            for name, vecs in st.session_state.all_vecs.items():
                if vecs is not None:
                    st.write(f"\n**{name}:**")
                    st.write(f"- Vector shape: {vecs.shape}")
                    st.write(f"- Number of IDs: {len(st.session_state.all_ids[name])}")

        # Embeddings Info
        st.subheader("Embedding Data")
        for index_name in ["task", "description"]:
            st.write(f"\n**{index_name}:**")

            emb_key = f"{index_name}_embeddings"
            red_key = f"{index_name}_reduced"
            lab_key = f"{index_name}_labels"

            if emb_key in st.session_state:
                embeddings = st.session_state[emb_key]
                reduced = st.session_state[red_key]
                labels = st.session_state[lab_key]

                st.write(f"- Original embeddings shape: {embeddings.shape}")
                st.write(f"- Reduced embeddings shape: {reduced.shape}")
                st.write(f"- Number of labels: {len(labels)}")
                st.write(f"- Unique clusters: {len(np.unique(labels))}")
