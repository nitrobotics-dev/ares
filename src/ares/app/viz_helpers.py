import io
import json
import os
import random
import traceback
import uuid
from typing import Any, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import streamlit as st
from Levenshtein import distance as levenshtein_distance

from ares.app.annotation_helpers import draw_annotations
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
from ares.configs.annotations import Annotation
from ares.databases.annotation_database import AnnotationDatabase
from ares.databases.embedding_database import IndexManager, rollout_to_index_name
from ares.name_remapper import DATASET_NAMES
from ares.utils.image_utils import (
    choose_and_preprocess_frames,
    get_video_frames,
    get_video_mp4,
)


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
        # Use groupby and rename the column to match the expected name
        success_rates = (
            df.groupby(col)["task_success"]
            .mean()
            .reset_index()
            .rename(columns={"task_success": "success_rate"})
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
    # Pre-filter numeric columns and cache the result
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    viz_types = {
        col: infer_visualization_type(col, df)["viz_type"] for col in numeric_cols
    }

    visualizations = []
    for col in sorted(numeric_cols):
        if col == time_column or viz_types[col] is None:
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
    filter_zero_distance_matches: bool = False,
) -> dict:
    """Create visualization for similar trajectories based on embedding feature type."""
    # get the query matrix
    query_matrix = index_manager.get_matrix_by_id(name, str(row.id))
    if query_matrix is None:
        raise ValueError(f"No matrix found for id: {row.id}")

    if filter_zero_distance_matches:
        # Try increasingly larger k values until we get enough non-zero distances
        multipliers = [1, 10, 100]  # Will try k, 10k, 100k
        for mult in multipliers:
            k = (n_most_similar + 1) * mult  # +1 to account for self-match
            distances, ids, matrices = index_manager.search_matrix(
                name, query_matrix, k
            )

            # Filter out exact matches (using small epsilon to handle floating point)
            non_zero_mask = distances > 1e-10
            filtered_distances = distances[non_zero_mask]
            filtered_ids = ids[non_zero_mask]
            filtered_matrices = (
                matrices[non_zero_mask] if matrices is not None else None
            )

            if len(filtered_distances) >= n_most_similar:
                # We found enough non-zero matches
                distances = filtered_distances[:n_most_similar]
                ids = filtered_ids[:n_most_similar]
                matrices = (
                    filtered_matrices[:n_most_similar]
                    if filtered_matrices is not None
                    else None
                )
                break
    else:
        # Original behavior
        distances, ids, matrices = index_manager.search_matrix(
            name,
            query_matrix,
            n_most_similar + 1,  # to avoid self match
        )

    # check index of id_str to see if it matches row.id then remove that index
    idx = np.where(ids == str(row.id))
    if len(idx[0]) != 0:
        idx = idx[0][0]
        distances = np.delete(distances, idx)
        ids = np.delete(ids, idx)
        matrices = np.delete(matrices, idx, axis=0)

    return {
        "distances": distances[:n_most_similar],
        "ids": ids[:n_most_similar],
        "matrices": matrices[:n_most_similar] if matrices is not None else None,
    }


def create_similarity_tabs(
    visualizations: list[dict],
    tab_names: list[str],
    df: pd.DataFrame,
    max_cols_in_tab: int = 5,
) -> None:
    """Create tabs specifically for similarity visualizations."""
    tabs = st.tabs(tab_names)
    for tab, viz_data in zip(tabs, visualizations):
        with tab:
            similar_cols = st.columns(min(max_cols_in_tab, len(viz_data["ids"])))
            for i, (dist, id_str) in enumerate(
                zip(viz_data["distances"], viz_data["ids"])
            ):
                with similar_cols[i % max_cols_in_tab]:
                    st.write(f"Distance: {dist:.3f}")
                    # Convert id_str to UUID only if it's not already a UUID
                    search_id = (
                        id_str if isinstance(id_str, uuid.UUID) else uuid.UUID(id_str)
                    )
                    found_rows = df[df["id"] == search_id]
                    if len(found_rows) == 0:
                        breakpoint()
                        st.write(f"No row found for id: {id_str}")
                    else:
                        display_video_card(
                            found_rows.iloc[0],
                            lazy_load=True,
                            key=f"video_card_{i}_tab_{tab}",
                        )


def get_video_annotation_data(video_id: str) -> dict | None:
    """Retrieve video metadata and annotations from the annotation database.

    Args:
        video_id: dataset/path of row
        db: Annotation database instance

    Returns:
        Dictionary containing video metadata and annotations
    """
    video_data = st.session_state.annotations_db.get_video_metadata(video_id)
    if not video_data:
        return None
    # Get all annotations for this video
    annotations: dict[int, list[Annotation]] = (
        st.session_state.annotations_db.get_annotations(
            video_id, annotation_type="detection", frame=None
        )
    )
    return {"video_data": video_data, "annotations": annotations}


def show_hero_display(
    df: pd.DataFrame,
    row: pd.Series,
    all_vecs: dict,
    index_manager: IndexManager,
    traj_array_show_n: int = 100,
    retrieve_n_most_similar: int = 5,
    lazy_load: bool = False,
    max_cols: int = 5,
) -> list[dict]:
    """
    Row 1: text
    Row 2: video col, detail + robot array plots
    Row 3: n tabs covering most similar based on state, action, video, text (embedding), text (metric)

    Returns:
        List of visualization figures to be included in export
    """
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
        if row.task_language_instruction:
            st.write(f"**Task:** {row.task_language_instruction}")
        with st.expander("Row Details", expanded=False):
            st.json(row.to_dict())

        if st.button("Generate Robot Array Plots", key="robot_array_plots_button_hero"):
            array_figs = generate_robot_array_plot_visualizations(
                row, all_vecs, traj_array_show_n, highlight_row=True
            )
        else:
            array_figs = []

    # Add annotation data retrieval button
    if st.button("Retrieve Annotation Data"):
        try:
            dataset_name = row["dataset_name"]
            if dataset_name in DATASET_NAMES:
                # HACK FOR NOW
                dataset_name = DATASET_NAMES[dataset_name]["file_name"]

            video_id = f"{dataset_name}/{row['path']}".replace("npy", "mp4")
            db_data = get_video_annotation_data(video_id)
            if db_data is not None:
                annotation_data = db_data.get("annotations")
                if not annotation_data:
                    st.warning(
                        f"No annotation data found for this video for {video_id}"
                    )
                else:
                    frame_inds = list(annotation_data.keys())
                    all_frame_paths = get_video_frames(
                        dataset, fname, n_frames=None, just_path=True
                    )
                    selected_frames = choose_and_preprocess_frames(
                        all_frame_paths,
                        specified_frames=frame_inds,
                    )
                    annotated_frames = [
                        draw_annotations(frame, anns)
                        for frame, anns in zip(
                            selected_frames, annotation_data.values()
                        )
                    ]
                    with st.expander("Annotated Frames", expanded=False):
                        max_cols = 3
                        cols = st.columns(max_cols)
                        for i, (frame_ind, frame) in enumerate(
                            zip(frame_inds, annotated_frames)
                        ):
                            with cols[i % max_cols]:
                                st.write(f"Frame {frame_ind}")
                                st.image(frame)

                    with st.expander("Raw Annotation Data (as JSON)", expanded=False):
                        st.write("Video Data:")
                        st.json(db_data["video_data"])
                        st.write("Annotations:")
                        st.json(db_data["annotations"])

            else:
                st.warning(f"No video or annotation data found for {video_id}")
        except Exception as e:
            st.error(f"Error retrieving annotation data: {str(e)}")
            st.error(traceback.format_exc())
            breakpoint()

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

    # some robotics datasets have lots of overlap, e.g. the same task instruction.
    # we may want to filter out zero-distance matches, even if they aren't the same ID
    # and will need to persist selection in state
    zero_distance_filter_key = "filter_zero_distance_matches"
    if zero_distance_filter_key not in st.session_state:
        st.session_state[zero_distance_filter_key] = False

    if st.checkbox(
        "Filter zero-distance matches", value=st.session_state[zero_distance_filter_key]
    ):
        st.session_state[zero_distance_filter_key] = True
    else:
        st.session_state[zero_distance_filter_key] = False

    # Get the similarity data
    state_viz_data = create_embedding_similarity_visualization(
        row,
        name=rollout_to_index_name(row, "states"),
        index_manager=index_manager,
        n_most_similar=retrieve_n_most_similar,
        filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
    )
    action_viz_data = create_embedding_similarity_visualization(
        row,
        name=rollout_to_index_name(row, "actions"),
        index_manager=index_manager,
        n_most_similar=retrieve_n_most_similar,
        filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
    )
    text_embedding_viz_data = create_embedding_similarity_visualization(
        row,
        name="description",  # HACK: fix naming convention
        index_manager=index_manager,
        n_most_similar=retrieve_n_most_similar,
        filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
    )

    text_viz_data = create_text_similarity_visualization(
        row,
        df,
        retrieve_n_most_similar,
        text_data_key,
        distance_fn_name=text_distance_fn.lower(),
        filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
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
        max_cols_in_tab=max_cols,
    )

    # Return all visualizations for export
    return array_figs + similarity_viz


def generate_robot_array_plot_visualizations(
    row: pd.Series,
    all_vecs: dict,
    show_n: int = 1000,
    keys: list[str] | None = None,
    highlight_row: bool = False,
    scores: dict[str, np.ndarray] | None = None,
) -> list[plotly.graph_objects.Figure]:
    if highlight_row and scores is not None:
        raise ValueError(
            f"Cannot provide both highlight_row and scores. Received highlight_row: {highlight_row}, scores: {scores}"
        )
    keys = keys or ["states", "actions"]
    figs = []
    for key in keys:
        name_key = row.dataset_name + "-" + row.robot_embodiment + "-" + key
        these_vecs = all_vecs[name_key]
        these_scores = scores.get(name_key) if scores else None

        with st.expander(f"Trajectory {key.title()} Display", expanded=False):
            try:
                highlight_idx = np.where(
                    st.session_state.all_ids[name_key] == str(row.id)
                )[0][0]
            except:
                breakpoint()
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
    filter_zero_distance_matches: bool = False,
) -> dict:
    """Create visualization for similar trajectories based on text similarity of task instructions."""
    reference = row[data_key]
    distance_fns = {"levenshtein": levenshtein_distance}
    distance_fn = distance_fns[distance_fn_name]
    distances = pd.Series(
        {idx: distance_fn(reference, text) for idx, text in df[data_key].items()}
    )

    # Remove the reference row if it exists in this filtered df
    current_row_idx = df[df["id"] == row["id"]].index
    if len(current_row_idx) > 0:
        distances = distances.drop(current_row_idx)

    if filter_zero_distance_matches:
        # Filter out exact matches (distance = 0)
        distances = distances[distances > 0]

    # Get the top N indices, making sure we don't request more than available
    n_available = min(n_most_similar, len(distances))
    similar_pairs = distances.nsmallest(n_available)

    return {
        "distances": similar_pairs.values,
        "ids": df.loc[similar_pairs.index]["id"].values,
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


def annotation_statistics(ann_db: AnnotationDatabase) -> None:
    stats = ann_db.get_database_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader(f"Total videos:")
        st.write(str(stats["total_videos"]))
    with col2:
        st.subheader(f"Total annotated frames:")
        st.write(str(stats["total_annotated_frames"]))
    with col3:
        st.subheader(f"Total annotations:")
        st.write(str(stats["total_annotations"]))
    with col4:
        st.subheader(f"Annotations by type:")
        for k, v in stats["annotations_by_type"].items():
            st.write(f"{k}: {v}")
