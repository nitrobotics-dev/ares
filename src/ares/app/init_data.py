import os

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.embedding_database import (
    EMBEDDING_DB_PATH,
    META_INDEX_NAMES,
    FaissIndex,
    IndexManager,
)
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)
from ares.models.base import VLM
from ares.utils.clustering import cluster_embeddings


def load_cached_embeddings(
    tmp_dump_dir: str, index_name: str, stored_embeddings: np.ndarray
) -> tuple | None:
    """
    Settig up embedding visualizations can be expensive, so we locally cache some of the generated arrays.
    """
    embeddings_path = os.path.join(tmp_dump_dir, f"{index_name}_embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, f"{index_name}_clusters.npz")
    ids_path = os.path.join(tmp_dump_dir, f"{index_name}_ids.npy")  # New path for IDs

    if not (
        os.path.exists(embeddings_path)
        and os.path.exists(clusters_path)
        and os.path.exists(ids_path)
    ):
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
    loaded_ids = np.load(ids_path)  # Load IDs
    return (
        loaded_embeddings,
        clusters_data["reduced"],
        clusters_data["labels"],
        loaded_ids,  # Return IDs as well
    )


def save_embeddings(
    tmp_dump_dir: str,
    index_name: str,
    embeddings: np.ndarray,
    reduced: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,  # Add IDs parameter
) -> None:
    """
    Save reduced embeddings, clusters, and IDs to disk
    """
    embeddings_path = os.path.join(tmp_dump_dir, f"{index_name}_embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, f"{index_name}_clusters.npz")
    ids_path = os.path.join(tmp_dump_dir, f"{index_name}_ids.npy")  # New path for IDs

    np.save(embeddings_path, embeddings)
    np.savez(clusters_path, reduced=reduced, labels=labels)
    np.save(ids_path, ids)  # Save IDs


def store_in_session(
    index_name: str,
    embeddings: np.ndarray,
    reduced: np.ndarray,
    labels: np.ndarray,
    stored_ids: np.ndarray,
) -> None:
    """
    Store embeddings, clusters, other info in session state
    """
    st.session_state[f"{index_name}_embeddings"] = embeddings
    st.session_state[f"{index_name}_reduced"] = reduced
    st.session_state[f"{index_name}_labels"] = labels
    st.session_state[f"{index_name}_ids"] = stored_ids


def initialize_data(tmp_dump_dir: str) -> None:
    """
    Initialize database connection, load data and create embeddings with caching.
    """
    # Skip if already initialized
    if all(
        key in st.session_state for key in ["ENGINE", "SESSION", "df", "INDEX_MANAGER"]
    ):
        print("Data already initialized")
        return

    # Initialize database and session
    print("Initializing database and session")
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    sess = Session(engine)
    st.session_state.ENGINE = engine
    st.session_state.SESSION = sess

    # Load dataframe
    print("Loading dataframe")
    query = select(RolloutSQLModel)
    df = pd.read_sql(query, engine)
    # Filter out unnamed columns
    df = df[[c for c in df.columns if "unnamed" not in c.lower()]]
    st.session_state.df = df

    # Initialize index manager
    print("Initializing index manager")
    index_manager = IndexManager(base_dir=EMBEDDING_DB_PATH, index_class=FaissIndex)
    st.session_state.INDEX_MANAGER = index_manager

    # Get all vectors and their IDs
    print("Getting all vectors and their IDs")
    all_data = index_manager.get_all_matrices()
    st.session_state.all_vecs = {
        name: data["arrays"] for name, data in all_data.items()
    }
    st.session_state.all_ids = {name: data["ids"] for name, data in all_data.items()}

    # Create tmp directory if it doesn't exist
    os.makedirs(tmp_dump_dir, exist_ok=True)

    # Process each index type
    for index_name in META_INDEX_NAMES:
        print(f"Processing {index_name} index")
        stored_embeddings = index_manager.indices[index_name].get_all_vectors()
        stored_ids = index_manager.indices[index_name].get_all_ids()

        # Try loading from cache first
        cached_data = load_cached_embeddings(
            tmp_dump_dir, index_name, stored_embeddings
        )
        if cached_data is not None:
            embeddings, reduced, labels, ids = cached_data  # Unpack IDs from cache
        else:
            # Create new embeddings and clusters
            embeddings = stored_embeddings
            reduced, labels, _ = cluster_embeddings(embeddings)
            ids = stored_ids  # Use the IDs from the index
            save_embeddings(tmp_dump_dir, index_name, embeddings, reduced, labels, ids)

        # Store in session state
        store_in_session(index_name, embeddings, reduced, labels, stored_ids)

    print("Setting up models")
    st.session_state.models = dict()
    st.session_state.models["summarizer"] = VLM(provider="openai", name="gpt-4o-mini")
    print("Setting up annotations database")
    st.session_state.annotations_db = AnnotationDatabase(
        connection_string=ANNOTATION_DB_PATH
    )
    print("Getting annotations database stats")
    st.session_state.annotation_db_stats = (
        st.session_state.annotations_db.get_database_stats()
    )


def display_state_info() -> None:
    """
    Helpful debugging state info displayed as the first rendered item in the streamlit object
    """
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
        st.write(
            "- ANNOTATIONS DB",
            "Connected" if "annotations_db" in st.session_state else "Not Connected",
        )
        # Index Manager Info
        st.subheader("Index Manager")
        if "INDEX_MANAGER" in st.session_state:
            index_manager = st.session_state.INDEX_MANAGER
            st.write(
                "Available indices: " + ", ".join(list(index_manager.indices.keys()))
            )
            with st.popover("Index Details"):
                for name, index in index_manager.indices.items():
                    st.write(f"\n**{name} Index:**")
                    st.write(f"- Feature dimension: {index.feature_dim}")
                    st.write(f"- Time steps: {index.time_steps}")
                    st.write(f"- Total entries: {index.n_entries}")

        # Vectors and IDs
        st.subheader("Stored Vectors and IDs")
        if "all_vecs" in st.session_state:
            with st.popover("Vector Details"):
                for name, vecs in st.session_state.all_vecs.items():
                    if vecs is not None:
                        st.write(f"\n**{name}:**")
                        st.write(f"- Vector shape: {vecs.shape}")
                        st.write(
                            f"- Number of IDs: {len(st.session_state.all_ids[name])}"
                        )

        st.subheader("Annotations DB")
        if "annotations_db" in st.session_state:
            if "annotation_db_stats" not in st.session_state:
                st.session_state.annotation_db_stats = (
                    st.session_state.annotations_db.get_database_stats()
                )
            st.json(st.session_state.annotation_db_stats, expanded=False)

        # Embeddings Info
        st.subheader("Embedding Data")
        for index_name in META_INDEX_NAMES:
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
