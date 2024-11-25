import json
import os
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from ares.app.export_data import export_options
from ares.app.filter_helpers import (
    embedding_data_filters_display,
    structured_data_filters_display,
)
from ares.app.init_data import initialize_data
from ares.app.viz_helpers import (
    create_tabbed_visualizations,
    display_video_grid,
    generate_automatic_visualizations,
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    plot_robot_array,
    show_dataframe,
    show_one_row,
)
from ares.clustering import visualize_clusters
from ares.databases.embedding_database import (
    TEST_EMBEDDING_DB_PATH,
    FaissIndex,
    IndexManager,
)
from ares.databases.structured_database import RolloutSQLModel
from ares.task_utils import PI_DEMO_PATH

title = "ARES Dashboard"
video_paths = list(os.listdir(PI_DEMO_PATH))

tmp_dump_dir = "/tmp/ares_dump"


@contextmanager
def filter_error_context(section_name: str) -> Any:
    """Context manager for handling filter operation errors."""
    try:
        yield
    except Exception as e:
        st.error(f"Error in {section_name}: {str(e)}\n{traceback.format_exc()}")
        # communicate stop to user
        st.write("Stopping execution")
        st.stop()
        return None


def load_data() -> pd.DataFrame:
    # Initialize mock data at the start of the app
    # initialize_mock_data(tmp_dump_dir)
    initialize_data(tmp_dump_dir)
    # Initial dataframe
    df = pd.read_sql(select(RolloutSQLModel), st.session_state.ENGINE)
    df = df[[c for c in df.columns if "unnamed" not in c.lower()]]
    return df


# Streamlit app
def main() -> None:
    # setup page and load data
    with filter_error_context("loading data"):
        print("\n" + "=" * 100 + "\n")
        st.set_page_config(page_title=title, page_icon="📊", layout="wide")
        st.title(title)

        df = load_data()

    # with filter_error_context("show example row"):
    #     show_one_row(df.sample(1).iloc[0])

    with filter_error_context("plot robot arrays"):
        row = df.iloc[0]
        index_manager = IndexManager(TEST_EMBEDDING_DB_PATH, FaissIndex)
        all_vecs = index_manager.get_all_matrices()
        vecs = all_vecs[row.dataset_name + "-" + row.robot_embodiment + "-states"]
        first_vecs = vecs[: min(50, len(vecs))]
        st.write(f"Showing first {len(first_vecs)} trajectories' states")
        plot_robot_array(first_vecs, title_base="Robot State Display", highlight_idx=0)

        # same for actions
        actions = all_vecs[row.dataset_name + "-" + row.robot_embodiment + "-actions"]
        first_actions = actions[: min(50, len(actions))]
        st.write(f"Showing first {len(first_actions)} trajectories' actions")
        plot_robot_array(
            first_actions, title_base="Robot Action Display", highlight_idx=1
        )

    # with filter_error_context("data filters"):
    #     # Structured data filters
    #     st.header(f"Data Filters")
    #     value_filtered_df = structured_data_filters_display(df)
    #     st.write(
    #         f"Selected {len(value_filtered_df)} rows out of {len(df)} total via structured data filters"
    #     )

    #     # Embedding data filters
    #     filtered_df, cluster_fig, selection, selection_flag = (
    #         embedding_data_filters_display(value_filtered_df)
    #     )
    #     st.write(
    #         f"Selection found! Using '{'box' if selection['box'] else 'lasso' if selection['lasso'] else 'points'}' as bounds"
    #         if selection_flag
    #         else "No selection found, using all points"
    #     )
    #     st.write(
    #         f"Selected {len(filtered_df)} rows out of {len(value_filtered_df)} available via embedding filters"
    #     )

    #     if filtered_df.empty:
    #         st.warning(
    #             "No data available for the selected points! Try adjusting your selection to receive analytics."
    #         )
    #         return

    # with filter_error_context("displaying data"):
    #     # show first 5 rows of dataframe
    #     show_dataframe(
    #         filtered_df.sample(min(5, len(filtered_df))), title="Sampled 5 Rows"
    #     )

    #     st.divider()  # Add horizontal line

    #     # Create overview of all data
    #     st.header("Distribution Analytics")
    #     general_visualizations = generate_automatic_visualizations(
    #         filtered_df, time_column="ingestion_time"
    #     )
    #     create_tabbed_visualizations(
    #         general_visualizations, [viz["title"] for viz in general_visualizations]
    #     )

    #     st.header("Success Rate Analytics")
    #     success_visualizations = generate_success_rate_visualizations(filtered_df)
    #     create_tabbed_visualizations(
    #         success_visualizations, [viz["title"] for viz in success_visualizations]
    #     )

    #     st.header("Time Series Trends")
    #     time_series_visualizations = generate_time_series_visualizations(
    #         filtered_df, time_column="ingestion_time"
    #     )
    #     create_tabbed_visualizations(
    #         time_series_visualizations,
    #         [viz["title"] for viz in time_series_visualizations],
    #     )

    #     # show video cards of first 5 rows in a horizontal layout
    #     display_video_grid(filtered_df)

    #     st.divider()  # Add horizontal line

    # with filter_error_context("exporting data"):
    #     # Export controls
    #     # Collect all visualizations
    #     # TODO: add structured data filters to export
    #     all_visualizations = [
    #         *general_visualizations,
    #         *success_visualizations,
    #         *time_series_visualizations,
    #     ]
    #     export_options(filtered_df, all_visualizations, title, cluster_fig=cluster_fig)


if __name__ == "__main__":
    main()
