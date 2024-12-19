import io
import json
import os
import random
import uuid
from typing import Any, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import streamlit as st
from Levenshtein import distance as levenshtein_distance

from ares.app.data_analysis import (
    generate_automatic_visualizations,
    infer_visualization_type,
)
from ares.app.plot_primitives import (
    create_bar_plot,
    create_line_plot,
    create_robot_array_plot,
    display_video_card,
)
from ares.databases.embedding_database import IndexManager, rollout_to_index_name
from ares.image_utils import get_video_frames, get_video_mp4


def generate_success_rate_visualizations(df: pd.DataFrame) -> list[dict]:
    """Generate success rate visualizations for categorical columns."""
    visualizations = []
    categorical_cols = sorted(
        [
            col
            for col in df.columns
            if infer_visualization_type(col, df)["viz_type"] == "bar"
        ]
    )

    for col in categorical_cols:
        # Create new DataFrame with success rates by category
        success_rates = pd.DataFrame(
            {
                col: df[col].unique(),
                "success_rate": [
                    df[df[col] == val]["task_success"].mean()
                    for val in df[col].unique()
                ],
            }
        )

        col_title = col.replace("_", " ").replace("-", " ").title()

        visualizations.append(
            {
                "figure": create_bar_plot(
                    success_rates,
                    x=col,
                    y="success_rate",
                    color="#2ecc71",
                    title=f"Success Rate by {col_title}",
                    labels={col: col_title, "success_rate": "Success Rate"},
                ),
                "title": f"{col_title} Success Rate",
            }
        )

    return visualizations


def generate_time_series_visualizations(
    df: pd.DataFrame, time_column: str = "creation_time"
) -> list[dict]:
    """Generate time series visualizations for numeric columns."""
    visualizations: list[dict] = []
    numeric_cols = sorted(df.select_dtypes(include=["int64", "float64"]).columns)

    for col in numeric_cols:
        viz_info = infer_visualization_type(col, df)
        if col == time_column or viz_info["viz_type"] is None:
            continue

        col_title = col.replace("_", " ").replace("-", " ").title()
        visualizations.append(
            {
                "figure": create_line_plot(
                    df,
                    x=time_column,
                    y=[col],
                    colors=["#1f77b4"],
                    title=f"{col_title} Over Time",
                    labels={
                        col: col_title,
                        time_column: time_column.title(),
                        "value": col_title,
                    },
                    y_format=".0%" if "task_success" in col.lower() else None,
                ),
                "title": f"{col_title} Trends",
            }
        )

    return visualizations


def create_tabbed_visualizations(
    visualizations: list[dict], tab_names: list[str]
) -> None:
    """Create tabs for each visualization."""
    tabs = st.tabs(tab_names)
    for tab, viz in zip(tabs, visualizations):
        with tab:
            st.plotly_chart(viz["figure"], use_container_width=True)


def display_video_grid(
    filtered_df: pd.DataFrame, max_videos: int = 5, lazy_load: bool = False
) -> None:
    """Display a grid of video cards for the first N rows of the dataframe.

    Args:
        filtered_df: DataFrame containing rollout data
        max_videos: Maximum number of videos to display in the grid
    """
    st.header("Rollout Examples")
    n_videos = min(max_videos, len(filtered_df))
    video_cols = st.columns(n_videos)

    for i, (_, row) in enumerate(filtered_df.head(n_videos).iterrows()):
        with video_cols[i]:
            display_video_card(dict(row), lazy_load=lazy_load, key=f"video_card_{i}")


def create_embedding_similarity_visualization(
    row: pd.Series,
    name: str,
    index_manager: IndexManager,
    n_most_similar: int,
) -> dict:
    """Create visualization for similar trajectories based on embedding feature type."""
    # get the query matrix
    query_matrix = index_manager.get_matrix_by_id(name, str(row.id))
    if query_matrix is None:
        raise ValueError(f"No matrix found for id: {row.id}")
    distances, ids, matrices = index_manager.search_matrix(
        name,
        query_matrix,
        n_most_similar + 1,  # to avoid self
    )

    # check index of id_str to see if it matches row.id then remove that index
    idx = np.where(ids == str(row.id))
    if len(idx[0]) != 0:
        idx = idx[0][0]
        distances = np.delete(distances, idx)
        ids = np.delete(ids, idx)
        matrices = np.delete(matrices, idx, axis=0)
    # Return the data instead of creating the visualization
    return {
        "distances": distances[:n_most_similar],
        "ids": ids[:n_most_similar],
        "matrices": matrices[:n_most_similar] if matrices is not None else None,
    }


def create_similarity_tabs(
    visualizations: list[dict],
    tab_names: list[str],
    df: pd.DataFrame,
    max_cols: int = 5,
) -> None:
    """Create tabs specifically for similarity visualizations."""
    tabs = st.tabs(tab_names)
    for tab, viz_data in zip(tabs, visualizations):
        with tab:
            similar_cols = st.columns(min(max_cols, len(viz_data["ids"])))
            for i, (dist, id_str) in enumerate(
                zip(viz_data["distances"], viz_data["ids"])
            ):
                with similar_cols[i % max_cols]:
                    st.write(f"Distance: {dist:.3f}")
                    # Convert id_str to UUID only if it's not already a UUID
                    search_id = (
                        id_str if isinstance(id_str, uuid.UUID) else uuid.UUID(id_str)
                    )
                    found_rows = df[df["id"] == search_id]
                    if len(found_rows) == 0:
                        st.write(f"No row found for id: {id_str}")
                    else:
                        display_video_card(
                            found_rows.iloc[0],
                            lazy_load=True,
                            key=f"video_card_{i}_tab_{tab}",
                        )


def show_hero_display(
    df: pd.DataFrame,
    idx: int,
    all_vecs: dict,
    show_n: int,
    index_manager: IndexManager,
    n_most_similar: int = 5,
    lazy_load: bool = False,
) -> list[dict]:
    """
    Row 1: text
    Row 2: video col, detail + robot array plots
    Row 3: n tabs covering most similar based on state, action, video, text (embedding), text (metric)

    Returns:
        List of visualization figures to be included in export
    """
    row = df.iloc[idx]
    # video card
    col1, col2 = st.columns(2)
    with col1:
        dataset, fname = (
            row["dataset_name"].lower().replace(" ", "_"),
            os.path.splitext(row["path"])[0],
        )
        if lazy_load:
            frame = get_video_frames(dataset, fname, n_frames=1)[0]
            st.image(frame)
            if st.button("Load Video"):
                st.video(get_video_mp4(dataset, fname))
        else:
            st.video(get_video_mp4(dataset, fname))
    with col2:
        with st.expander("Row Details", expanded=False):
            for col, val in row.items():
                if len(str(val)) > 1000:
                    continue
                st.write(f"{col}: {val}")
        if st.button("Generate Robot Array Plots", key="robot_array_plots_button_hero"):
            array_figs = generate_robot_array_plot_visualizations(
                row, all_vecs, show_n, highlight_idx=idx
            )
        else:
            array_figs = []

    # Row 3: n tabs covering most similar based on state, action, text
    st.write(f"**Similarity Search**")
    st.write(f"Most similar examples to {row['id']}, based on:")

    text_distance_fn = "Levenshtein"
    text_data_key = "task_language_instruction"
    tab_names = [
        f"Text - {text_distance_fn}",
        f"Text - Embedding",
        "State",
        "Action",
    ]

    # Get the similarity data
    state_viz_data = create_embedding_similarity_visualization(
        row,
        name=rollout_to_index_name(row, "states"),
        index_manager=index_manager,
        n_most_similar=n_most_similar,
    )
    action_viz_data = create_embedding_similarity_visualization(
        row,
        name=rollout_to_index_name(row, "actions"),
        index_manager=index_manager,
        n_most_similar=n_most_similar,
    )
    text_embedding_viz_data = create_embedding_similarity_visualization(
        row,
        name="description",  # HACK
        index_manager=index_manager,
        n_most_similar=n_most_similar,
    )

    text_viz_data = create_text_similarity_visualization(
        row,
        df,
        n_most_similar,
        text_data_key,
        distance_fn_name=text_distance_fn.lower(),
    )

    similarity_viz = [
        text_viz_data,
        text_embedding_viz_data,
        state_viz_data,
        action_viz_data,
    ]

    # Create the tabs with the data
    create_similarity_tabs(
        similarity_viz,
        tab_names,
        df,
    )

    # Return all visualizations for export
    return array_figs + similarity_viz


def generate_robot_array_plot_visualizations(
    row: pd.Series,
    all_vecs: dict,
    show_n: int = 1000,
    keys: list[str] | None = None,
    highlight_idx: int | None = None,
    scores: dict[str, np.ndarray] | None = None,
) -> list[plotly.graph_objects.Figure]:
    if highlight_idx is not None and scores is not None:
        raise ValueError(
            f"Cannot provide both highlight_idx and scores. Received highlight_idx: {highlight_idx}, scores: {scores}"
        )
    keys = keys or ["states", "actions"]
    figs = []
    for key in keys:
        name_key = row.dataset_name + "-" + row.robot_embodiment + "-" + key
        these_vecs = all_vecs[name_key]
        these_scores = scores.get(name_key) if scores else None

        with st.expander(f"Trajectory {key.title()} Display", expanded=False):
            fig = create_robot_array_plot(
                these_vecs,
                title_base=f"Trajectory {key.title()} Display",
                highlight_idx=highlight_idx,
                show_n=show_n,
                scores=these_scores,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"robot_array_{name_key}_{show_n}_{row.name}",
            )
            figs.append(fig)
    return figs


def create_text_similarity_visualization(
    row: pd.Series,
    df: pd.DataFrame,
    n_most_similar: int,
    data_key: str,
    distance_fn_name: str = "levenshtein",
) -> dict:
    """Create visualization for similar trajectories based on text similarity of task instructions."""
    # Get the reference instruction
    reference = row[data_key]

    # Calculate distances for all instructions
    distance_fns = {"levenshtein": levenshtein_distance}
    distance_fn = distance_fns[distance_fn_name]
    distances = df[data_key].apply(lambda x: distance_fn(reference, x))

    distances = pd.Series(distances, index=df.index)
    distances = distances[distances.index != row.name]  # remove ROW
    top_indices = distances.nsmallest(n_most_similar).index

    return {
        "distances": distances[top_indices].values,
        "ids": df.iloc[top_indices]["id"].values,
        "matrices": None,
    }


def total_statistics(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader(f"Total rollouts:")
        st.write(str(len(df)))
    with col2:
        st.subheader(f"Total length:")
        st.write(str(df.length.sum()) + " steps")
    with col3:
        st.subheader(f"Total robots:")
        st.write(str(df["robot_embodiment"].nunique()))
    with col4:
        st.subheader(f"Total datasets:")
        st.write(str(df["dataset_name"].nunique()))
