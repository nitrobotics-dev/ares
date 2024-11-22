import os

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from ares.clustering import cluster_embeddings
from ares.databases.structured_database import (
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)


def initialize_data(tmp_dump_dir: str) -> None:
    """Initialize database connection and load/create embeddings"""
    if "ENGINE" not in st.session_state or "SESSION" not in st.session_state:
        engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
        sess = Session(engine)
        st.session_state.ENGINE = engine
        st.session_state.SESSION = sess

    # get len of available dataset
    available_len = len(pd.read_sql(select(RolloutSQLModel), st.session_state.ENGINE))

    # Create tmp directory if it doesn't exist
    os.makedirs(tmp_dump_dir, exist_ok=True)
    embeddings_path = os.path.join(tmp_dump_dir, "embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, "clusters.npz")

    # Initialize or load embeddings
    if "embeddings" not in st.session_state:
        if os.path.exists(embeddings_path):
            # Load from disk
            loaded_embeddings = np.load(embeddings_path)
            # Check if loaded data matches current dataset size
            if len(loaded_embeddings) == available_len:
                st.session_state.embeddings = loaded_embeddings
                clusters_data = np.load(clusters_path)
                st.session_state.reduced = clusters_data["reduced"]
                st.session_state.labels = clusters_data["labels"]
                st.session_state.probs = clusters_data["probs"]
            else:
                # Data size mismatch, create new embeddings
                embeddings = np.random.rand(available_len, 2)
                for i in range(3):
                    embeddings[i * 200 : (i + 1) * 200] += i

                reduced, labels, probs = cluster_embeddings(embeddings)

                # Save to disk
                np.save(embeddings_path, embeddings)
                np.savez(clusters_path, reduced=reduced, labels=labels, probs=probs)

                # Store in session state
                st.session_state.embeddings = embeddings
                st.session_state.reduced = reduced
                st.session_state.labels = labels
                st.session_state.probs = probs


def initialize_mock_data(tmp_dump_dir: str, video_paths: list[str]) -> None:
    """Initialize or load mock data"""
    mock_data_path = os.path.join(tmp_dump_dir, "mock_data.pkl")

    if "MOCK_DATA" not in st.session_state:
        if os.path.exists(mock_data_path):
            # Load from disk
            st.session_state.MOCK_DATA = pd.read_pickle(mock_data_path)
        else:
            # Create new random data
            base_dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
            np.random.seed(42)  # Set seed for reproducibility
            random_offsets = np.random.randint(0, 365, size=365)
            dates = [
                date - pd.Timedelta(days=int(offset))
                for date, offset in zip(base_dates, random_offsets)
            ]

            sampled_paths = np.random.choice(video_paths, size=365)

            mock_data = pd.DataFrame(
                {
                    "creation_time": dates,
                    "length": np.random.randint(1, 100, size=365),
                    "task_success": np.array(
                        [np.random.uniform(i / 365, 1) for i in range(365)]
                    ),
                    "id": [f"vid_{i}" for i in range(365)],
                    "task": [f"Robot Task {i}" for i in np.random.randint(0, 10, 365)],
                    "views": np.random.randint(100, 1000, size=365),
                    "video_path": [
                        f"/workspaces/ares/data/pi_demos/{path}"
                        for path in sampled_paths
                    ],
                }
            )

            # Save to disk
            mock_data.to_pickle(mock_data_path)

            # Store in session state
            st.session_state.MOCK_DATA = mock_data
