import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from ares.app.export_data import export_options
from ares.app.filter_helpers import embedding_data_filters, structured_data_filters
from ares.app.init_data import initialize_data
from ares.app.viz_helpers import (
    create_tabbed_visualizations,
    display_video_grid,
    generate_automatic_visualizations,
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    show_dataframe,
)
from ares.clustering import visualize_clusters
from ares.databases.structured_database import RolloutSQLModel
from ares.task_utils import PI_DEMO_PATH

title = "ARES Dashboard"
video_paths = list(os.listdir(PI_DEMO_PATH))

tmp_dump_dir = "/tmp/ares_dump"


# Streamlit app
def main() -> None:
    try:
        print("\n" + "=" * 100 + "\n")
        st.set_page_config(page_title=title, page_icon="📊", layout="wide")
        st.title(title)

        # Initialize mock data at the start of the app
        # initialize_mock_data(tmp_dump_dir)
        initialize_data(tmp_dump_dir)
        # Initial dataframe
        df = pd.read_sql(select(RolloutSQLModel), st.session_state.ENGINE)
        df = df[[c for c in df.columns if "unnamed" not in c.lower()]]

        value_filtered_df = structured_data_filters(df)
        st.write(f"Selected {len(value_filtered_df)} rows out of {len(df)} total")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")

    try:
        # Create the visualization using state data
        cluster_fig, cluster_df, cluster_to_trace = visualize_clusters(
            st.session_state.reduced,
            st.session_state.labels,
            keep_mask=value_filtered_df.index.tolist(),
        )
        selection_flag, indices, selection = embedding_data_filters(
            value_filtered_df, cluster_fig, cluster_to_trace
        )
        n_pts = len(indices)
        clusters = st.session_state.labels[indices] if n_pts > 0 else []
        n_clusters = len(np.unique(clusters))
        filtered_df = value_filtered_df.iloc[
            indices
        ]  # Second stage of filtering (by cluster selection)
        ######### end hack #########
        st.write(
            f"Selection found! Using '{'box' if selection['box'] else 'lasso' if selection['lasso'] else 'points'}' as bounds"
            if selection_flag
            else "No selection found, using all points"
        )
        st.write(
            f"Found {n_pts} point{'' if n_pts == 1 else 's'} from {n_clusters} cluster{'' if n_clusters == 1 else 's'}"
        )

        if filtered_df.empty:
            st.warning(
                "No data available for the selected points! Try adjusting your selection to receive analytics."
            )
            return

        st.header("Selection Controls")
        st.write("Double click to clear selection")
        if st.button("Summarize Selection"):
            st.write(f"selected {len(selection['points'])} points")
            st.write(f"ex: {selection['points'][:5]}")
            st.write(f"filtered df: {filtered_df.sample(5)}")

        # show first 5 rows of dataframe
        show_dataframe(filtered_df.sample(5), title="Sampled 5 Rows")

        st.divider()  # Add horizontal line

        # Create overview of all data
        st.header("Distribution Analytics")
        general_visualizations = generate_automatic_visualizations(
            filtered_df, time_column="ingestion_time"
        )
        create_tabbed_visualizations(
            general_visualizations, [viz["title"] for viz in general_visualizations]
        )

        st.header("Success Rate Analytics")
        success_visualizations = generate_success_rate_visualizations(filtered_df)
        create_tabbed_visualizations(
            success_visualizations, [viz["title"] for viz in success_visualizations]
        )

        st.header("Time Series Trends")
        time_series_visualizations = generate_time_series_visualizations(
            filtered_df, time_column="ingestion_time"
        )
        create_tabbed_visualizations(
            time_series_visualizations,
            [viz["title"] for viz in time_series_visualizations],
        )

        # show video cards of first 5 rows in a horizontal layout
        display_video_grid(filtered_df)

        st.divider()  # Add horizontal line

        # Export controls
        # Collect all visualizations
        all_visualizations = [
            *general_visualizations,
            *success_visualizations,
            *time_series_visualizations,
        ]
        export_options(filtered_df, all_visualizations, title, cluster_fig=cluster_fig)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
