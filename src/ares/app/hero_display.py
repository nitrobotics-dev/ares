import traceback

import numpy as np
import pandas as pd
import streamlit as st

from ares.app.annotation_viz_helpers import draw_annotations, draw_detection_data
from ares.app.viz_helpers import (
    create_embedding_similarity_visualization,
    create_similarity_tabs,
    create_text_similarity_visualization,
    generate_robot_array_plot_visualizations,
    get_video_annotation_data,
)
from ares.databases.annotation_database import get_video_id
from ares.databases.embedding_database import (
    META_INDEX_NAMES,
    TRAJECTORY_INDEX_NAMES,
    IndexManager,
    rollout_to_index_name,
)
from ares.utils.image_utils import (
    choose_and_preprocess_frames,
    get_video_frames,
    get_video_mp4,
)


def setup_zero_distance_checkbox_with_state() -> str:
    zero_distance_filter_key = "filter_zero_distance_matches"
    if zero_distance_filter_key not in st.session_state:
        st.session_state[zero_distance_filter_key] = False

    if st.checkbox(
        "Filter out zero-distance matches",
        value=st.session_state[zero_distance_filter_key],
    ):
        st.session_state[zero_distance_filter_key] = True
    else:
        st.session_state[zero_distance_filter_key] = False
    return zero_distance_filter_key


def display_hero_annotations(
    db_data: dict, video_id: str, dataset: str, fname: str
) -> None:
    """
    Create a nice display of all the annotations in the database.
    This include grounding annotations (e.g. detections) as well as composed datasets like embodied-chain-of-thought.
    """
    # check for annotation data
    annotation_data = db_data.get("annotations")
    if not annotation_data:
        st.warning(f"No annotation data found for this video for {video_id}")
    else:
        detection_data = annotation_data.get("detection")
        if detection_data:
            draw_detection_data(detection_data, dataset, fname)

        # show other top-level annotations (not frame-based)
        other_keys = [k for k in annotation_data.keys() if k != "detection"]
        with st.expander("Annotation Description Data", expanded=False):
            for key in other_keys:
                if isinstance(annotation_data[key], list):
                    this_data = [
                        ann.description
                        for ann in annotation_data[key]
                        if ann.description
                    ]
                    if this_data:
                        st.write(f"**{key.replace('_', ' ').title()}**")
                        st.write(this_data)

        # display all other annotation data, eg pseduo-ECoT, in-context-learning datasets, etc
        with st.expander("Raw Annotation Data (as JSON)", expanded=False):
            if "video_data" in db_data:
                st.write("Video Data:")
                st.json(db_data["video_data"], expanded=False)
            if "annotations" in db_data:
                st.write("Annotations:")
                st.json(db_data["annotations"], expanded=False)


def create_similarity_viz_objects(
    row: pd.Series,
    df: pd.DataFrame,
    index_manager: IndexManager,
    retrieve_n_most_similar: int,
) -> tuple[list[str], list[dict]]:
    """
    Use our embedding indices to retrieve similar examples based on meta indices (task, description) or trajectories (states, actions) per dataset.
    Right now we construct a pure-text comparison for only task as descriptions are too long for efficient calculation.
    """
    # some robotics datasets have lots of overlap, e.g. the same task instruction.
    # we may want to filter out zero-distance matches, even if they aren't the same ID
    # and will need to persist selection in state
    zero_distance_filter_key = setup_zero_distance_checkbox_with_state()

    # text comparison of levenshtein distance over task
    text_distance_fn = "levenshtein"
    text_distance_data_key = "task_language_instruction"
    text_viz_data = create_text_similarity_visualization(
        row,
        df,
        n_most_similar=retrieve_n_most_similar,
        data_key=text_distance_data_key,
        distance_fn_name=text_distance_fn,
        filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
    )

    # embedding retrieval over META_INDEX_NAMES
    text_embedding_viz_data = [
        create_embedding_similarity_visualization(
            row,
            name=key,
            index_manager=index_manager,
            n_most_similar=retrieve_n_most_similar,
            filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
        )
        for key in META_INDEX_NAMES
    ]

    # embedding retrieval over TRAJECTORY_INDEX_NAMES
    trajectory_viz_data = [
        create_embedding_similarity_visualization(
            row,
            name=rollout_to_index_name(row, k),
            index_manager=index_manager,
            n_most_similar=retrieve_n_most_similar,
            filter_zero_distance_matches=st.session_state[zero_distance_filter_key],
        )
        for k in TRAJECTORY_INDEX_NAMES
    ]

    # organize tab names and visualizations for tabs
    tab_names = [
        f"{text_distance_data_key.replace('_',' ').title()} - {text_distance_fn.title()}",
        *[f"{t.replace('_', ' ')} - Embedding".title() for t in META_INDEX_NAMES],
        *[t.title() for t in TRAJECTORY_INDEX_NAMES],
    ]

    similarity_viz = [
        text_viz_data,
        *text_embedding_viz_data,
        *trajectory_viz_data,
    ]
    return tab_names, similarity_viz


def show_hero_display(
    df: pd.DataFrame,
    row: pd.Series,
    all_vecs: dict,
    index_manager: IndexManager,
    traj_array_show_n: int = 100,
    retrieve_n_most_similar: int = 5,
    lazy_load: bool = False,
    max_cols: int = 5,
) -> None:
    """
    Row 1: text
    Row 2: video col, detail + robot array plots
    Row 3: n tabs covering most similar based on state, action, video, text (embedding), text (metric)
    Returns: visualization figures to be included in export
    """

    dataset, fname = (
        row["dataset_filename"],
        row["filename"],
    )

    col1, col2 = st.columns(2)
    with col1:
        # display the video
        if lazy_load:
            frame = get_video_frames(dataset, fname, n_frames=1)[0]
            st.image(frame)
            if st.button("Load Video"):
                st.video(get_video_mp4(dataset, fname))
        else:
            st.video(get_video_mp4(dataset, fname))
    with col2:
        # display a few key pieces of information, e.g. task and success
        if row.task_language_instruction:
            st.write(f"**Task:** {row.task_language_instruction}")
        if not np.isnan(row.task_success) and row.task_success:
            st.write(f"**Success:** {row.task_success:.2f}")
        if isinstance(row.trajectory_reward_step, str):
            if int(row.trajectory_reward_step) >= 0:
                st.write(
                    f"**Reward Step:** {row.trajectory_reward_step} ({100*int(row.trajectory_reward_step) / int(row.length):.2f}% through rollout)"
                )
            else:
                st.write(f"Failure episode!")
        # optionally display the rest of the row details as a truncated json
        with st.expander("Row Details", expanded=False):
            json_repr = {
                k: (v if len(str(v)) < 1000 else str(v)[:1000] + "...")
                for k, v in sorted(row.to_dict().items(), key=lambda x: x[0])
            }
            st.json(json_repr, expanded=False)

        if st.button("Generate Robot Array Plots", key="robot_array_plots_button_hero"):
            array_figs = generate_robot_array_plot_visualizations(
                row, all_vecs, traj_array_show_n, highlight_row=True
            )
        else:
            array_figs = []

    # Add annotation data retrieval button
    if st.button("Retrieve Annotation Data"):
        try:
            video_id = get_video_id(dataset, fname)
            db_data = get_video_annotation_data(video_id)
            if db_data is not None:
                display_hero_annotations(db_data, video_id, dataset, fname)
            else:
                st.warning(f"No video or annotation data found for {video_id}")
        except Exception as e:
            st.error(f"Error retrieving annotation data: {str(e)}")
            st.error(traceback.format_exc())
            breakpoint()

    # Row 3: n tabs covering most similar based on state, action, text
    st.write(f"**Similar Examples**")
    st.write(f"Most similar examples to {row['id']}, based on:")

    tab_names, similarity_viz = create_similarity_viz_objects(
        row, df, index_manager, retrieve_n_most_similar
    )

    # Create the tabs with the data
    create_similarity_tabs(
        similarity_viz,
        tab_names,
        df,
        max_cols_in_tab=max_cols,
    )
