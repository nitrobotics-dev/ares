import json
import os
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from ares.app.data_analysis import generate_automatic_visualizations
from ares.app.export_data import export_options
from ares.app.filter_helpers import (
    embedding_data_filters_display,
    select_row_from_df_user,
    structured_data_filters_display,
)
from ares.app.init_data import display_state_info, initialize_data
from ares.app.plot_primitives import show_dataframe
from ares.app.viz_helpers import (
    annotation_statistics,
    create_tabbed_visualizations,
    display_video_grid,
    generate_robot_array_plot_visualizations,
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    show_hero_display,
    total_statistics,
)
from ares.constants import ARES_DATA_DIR
from ares.databases.structured_database import RolloutSQLModel

print(f"finished imports")

# top level variables
title = "ARES Dashboard"
tmp_dump_dir = os.path.join(ARES_DATA_DIR, "tmp2")
section_times: dict[str, float] = defaultdict(float)


@contextmanager
def filter_error_context(section_name: str) -> Any:
    """Context manager for handling filter operation errors."""
    try:
        yield
    except Exception as e:
        st.error(f"Error in {section_name}: {str(e)}\n{traceback.format_exc()}")
        st.write("Stopping execution")
        st.stop()
        return None


@contextmanager
def timer_context(section_name: str) -> Any:
    """Context manager for timing sections."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        section_times[section_name] += elapsed_time


def load_data() -> pd.DataFrame:
    """Load data from session state or initialize if needed."""
    initialize_data(tmp_dump_dir)
    return st.session_state.df


# Streamlit app
def main() -> None:
    # Define section names
    section_loading = "loading data"
    print(section_loading)
    with filter_error_context(section_loading), timer_context(section_loading):
        print("\n" + "=" * 100 + "\n")
        st.set_page_config(page_title=title, page_icon="📊", layout="wide")
        st.title(title)
        df = load_data()

    section_state_info = "state info"
    print(section_state_info)
    with filter_error_context(section_state_info), timer_context(section_state_info):
        display_state_info()
        total_statistics(df)
        annotation_statistics(st.session_state.annotations_db)
        st.divider()

    section_filters = "structured data filters"
    print(section_filters)
    with filter_error_context(section_filters), timer_context(section_filters):
        # Structured data filters
        st.header(f"Data Filters")
        value_filtered_df = structured_data_filters_display(df, debug=True)
        kept_ids = value_filtered_df["id"].apply(str).tolist()
        st.write(
            f"Selected {len(value_filtered_df)} rows out of {len(df)} total via structured data filters"
        )
        filtered_df = None

        if len(kept_ids) == 0:
            breakpoint()

    # section_embedding_filters = "embedding data filters"
    # print(section_embedding_filters)
    # with filter_error_context(section_embedding_filters), timer_context(section_embedding_filters):
    #     # Embedding data filters
    #     id_key = "id"
    #     state_key = "task_language_instruction"
    #     raw_data_key = "task_language_instruction"

    #     # TODO: present description, task language instruction filter options
    #     reduced_embeddings = st.session_state[f"{state_key}_reduced"]
    #     labels = st.session_state[f"{state_key}_labels"]
    #     state_ids = st.session_state[f"{state_key}_ids"]
    #     try:
    #         desired_ids = df[id_key].apply(str)
    #         id_to_idx_state = {_id: idx for idx, _id in enumerate(state_ids)}
    #         desired_state_indices = [id_to_idx_state[_id] for _id in desired_ids]
    #         reduced_embeddings = reduced_embeddings[desired_state_indices]
    #         labels = labels[desired_state_indices]

    #         filtered_df, cluster_fig, selection, selection_flag = (
    #             embedding_data_filters_display(
    #                 df=df,
    #                 reduced=reduced_embeddings,
    #                 labels=labels,
    #                 raw_data_key=raw_data_key,
    #                 id_key=id_key,
    #                 keep_mask=kept_ids,
    #             )
    #         )
    #     except Exception as e:
    #         print(e)
    #         print(traceback.format_exc())
    #         breakpoint()

    #     if filtered_df.empty:
    #         st.warning(
    #             "No data available for the selected points! Try adjusting your selection to receive analytics."
    #         )
    #         return

    if filtered_df is None:
        filtered_df = value_filtered_df

    # Add a button to refresh the sample
    st.button(
        "Get New Random Sample"
    )  # Button press triggers streamlit rerun, triggers new random sample
    show_dataframe(filtered_df.sample(min(5, len(filtered_df))), title="Data Sample")
    # st.divider()

    section_display = "data distributions"
    with filter_error_context(section_display), timer_context(section_display):
        max_x_bar_options = 100
        # Create overview of all data
        st.header("Distribution Analytics")
        general_visualizations = generate_automatic_visualizations(
            filtered_df,
            time_column="ingestion_time",
            max_x_bar_options=max_x_bar_options,
        )
        general_visualizations = sorted(
            general_visualizations, key=lambda x: x["title"]
        )
        create_tabbed_visualizations(
            general_visualizations, [viz["title"] for viz in general_visualizations]
        )

    section_success_rate = "success estimate analytics"
    with (
        filter_error_context(section_success_rate),
        timer_context(section_success_rate),
    ):
        st.header("Success Estimate Analytics")
        success_visualizations = generate_success_rate_visualizations(filtered_df)
        create_tabbed_visualizations(
            success_visualizations, [viz["title"] for viz in success_visualizations]
        )
    st.divider()

    section_time_series = "time series trends"
    with filter_error_context(section_time_series), timer_context(section_time_series):
        st.header("Time Series Trends")
        time_series_visualizations = generate_time_series_visualizations(
            filtered_df, time_column="ingestion_time"
        )
        create_tabbed_visualizations(
            time_series_visualizations,
            [viz["title"] for viz in time_series_visualizations],
        )

    section_video_grid = "video grid"
    with filter_error_context(section_video_grid), timer_context(section_video_grid):
        # show video cards of first 5 rows in a horizontal layout
        st.header("Rollout Examples")
        n_videos = 5
        display_rows = pd.concat(
            {k: v.head(1) for k, v in filtered_df.groupby("dataset_name")}
        )
        if len(display_rows) < n_videos:
            # get enough videos to fill n_videos that arent already in display_rows
            extra_rows = filtered_df.head(n_videos)
            # remove rows that are already in display_rows
            extra_rows = extra_rows[~extra_rows.id.isin(display_rows.id)]
            display_rows = pd.concat([display_rows, extra_rows])
        display_video_grid(display_rows, lazy_load=True)
    st.divider()

    section_plot_hero = "plot hero display"
    with filter_error_context(section_plot_hero), timer_context(section_plot_hero):
        st.header("Rollout Display")
        # initialize or persist selected row in state
        select_row_from_df_user(filtered_df)
        selected_row = st.session_state.get("selected_row")

        if selected_row is not None:
            show_dataframe(pd.DataFrame([selected_row]), title="Selected Row")
            st.write(f"Selected row ID: {selected_row.id}")
            hero_visualizations = show_hero_display(
                df,  # compare selected row from filtered_df to all rows in df
                selected_row,
                st.session_state.all_vecs,
                index_manager=st.session_state.INDEX_MANAGER,
                lazy_load=False,
                retrieve_n_most_similar=10,
            )
        else:
            st.info("Please select a row to display details")
    st.divider()

    section_plot_robots = "plot robot arrays"
    with filter_error_context(section_plot_robots), timer_context(section_plot_robots):
        if st.button("Generate Robot Array Plots", key="robot_array_plots_button"):
            st.header("Robot Array Display")
            # Number of trajectories to display in plots
            robot_array_visualizations = generate_robot_array_plot_visualizations(
                selected_row,  # need row to select dataset/robot embodiment of trajectories
                st.session_state.all_vecs,
                show_n=1000,
            )
        else:
            st.write("No robot array plots generated")
            robot_array_visualizations = []
    st.divider()

    section_export = "exporting data"
    with filter_error_context(section_export), timer_context(section_export):
        # Export controls
        # Collect all visualizations
        # TODO: add structured data filters to export
        all_visualizations = [
            *general_visualizations,
            *success_visualizations,
            *time_series_visualizations,
            *robot_array_visualizations,
            *hero_visualizations,  # Add hero visualizations to export
        ]
        export_options(filtered_df, all_visualizations, title, cluster_fig=cluster_fig)

    # Print timing report at the end
    print("\n=== Timing Report ===")
    print(f"Total time: {sum(section_times.values()):.2f} seconds")
    for section, elapsed_time in section_times.items():
        print(f"{section}: {elapsed_time:.2f} seconds")
    print("==================\n")


if __name__ == "__main__":
    main()
