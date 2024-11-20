import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pdfkit
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select
from sqlmodel import SQLModel

from ares.app.export_data import export_options
from ares.app.init_data import initialize_data
from ares.app.viz_helpers import (
    create_bar_plot,
    create_data_filters,
    create_histogram,
    create_line_plot,
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
    print("\n" + "=" * 100 + "\n")
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    st.title(title)

    # Initialize mock data at the start of the app
    # initialize_mock_data(tmp_dump_dir)
    initialize_data(tmp_dump_dir)

    # Initial dataframe
    df = pd.read_sql(select(RolloutSQLModel), st.session_state.ENGINE)
    df = df[[c for c in df.columns if "unnamed" not in c.lower()]]

    # Add filters section
    st.header("Filters")
    col1, col2 = st.columns([6, 1])  # Create columns for layout
    with col1:
        st.subheader("Data Filters")
    with col2:
        if st.button("Reset Filters", type="primary"):
            # Clear all filter-related session state variables
            for key in list(st.session_state.keys()):
                if key.startswith("filter_"):
                    del st.session_state[key]
            st.rerun()  # Rerun the app to reset the filters

    value_filtered_df, skipped_cols = create_data_filters(df)
    if skipped_cols:
        st.warning(
            f"Skipped columns: {skipped_cols} due to high cardinality or lack of unique values"
        )

    st.write(f"Selected {len(value_filtered_df)} rows out of {len(df)} total")
    # st.write(f"Selected indices: {value_filtered_df.index.tolist()}")
    # st.write(f"Keep Mask: {value_filtered_df.index.tolist()}")

    try:
        # Create the visualization using state data
        fig, cluster_df, cluster_to_trace = visualize_clusters(
            st.session_state.reduced,
            st.session_state.labels,
            keep_mask=value_filtered_df.index.tolist(),
        )

        # Create columns for controls and info
        col1, col2 = st.columns([3, 1])

        with col1:
            # Display the plot and capture selected points from the event data
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="cluster_plot",
                selection_mode=["points", "box", "lasso"],
                # on_select=handle_select,  # Pass the function reference
                on_select="rerun",
            )
            selection = event.selection

        with col2:
            st.write("Selection Controls")
            st.write("Double click to clear selection")
            if st.button("Summarize Selection"):
                st.write(f"selected {len(selection['points'])} points")
                st.write(f"ex: {selection['points'][:5]}")

        ##### start hack #####
        # # HACK - fit selected points to df, only for this data
        available_dataset_pts = len(df)
        st.write(f"hack: available_dataset_pts: {available_dataset_pts}")
        # determine if selection was made OR if just no points selected
        selection_flag = any(len(selection[k]) > 0 for k in ["box", "lasso", "points"])
        if selection_flag:
            # selection made!
            indices = []
            try:
                valid_traces = [
                    v for k, v in cluster_to_trace.items() if "cluster" in k
                ]
                # Create a mapping of filtered indices to original indices
                filtered_to_original = value_filtered_df.index.tolist()

                for point in selection["points"]:
                    if point["curve_number"] in valid_traces:
                        # Get the point's position within its trace
                        point_number = point["point_number"]
                        # Get the trace data
                        trace_data = fig.data[point["curve_number"]]
                        # Get the original index from custom_data
                        original_idx = trace_data.customdata[point_number][0]
                        indices.append(original_idx)

                st.write(f"Debug info:")
                st.write(f"- Valid traces: {valid_traces}")
                st.write(f"- Number of selected points: {len(selection['points'])}")
                st.write(f"- First few points: {selection['points'][:2]}")
                st.write(f"- Resulting indices: {indices[:5]}")
            except Exception as e:
                st.error(
                    f"Error processing selection: {str(e)}\n{traceback.format_exc()}"
                )
                indices = []
        else:
            # no selection made, use all filtered points
            indices = value_filtered_df.index.tolist()
        # breakpoint()

        n_pts = len(indices)
        # breakpoint()
        clusters = st.session_state.labels[indices] if n_pts > 0 else []
        st.write(
            f"Selection found! Using {'box' if selection['box'] else 'lasso' if selection['lasso'] else 'points'} as bounds"
            if selection_flag
            else "No selection found, using all points"
        )
        n_clusters = len(np.unique(clusters))
        st.write(
            f"Selected {n_pts} point{'' if n_pts == 1 else 's'} from {n_clusters} cluster{'' if n_clusters == 1 else 's'}"
        )
        filtered_df = value_filtered_df.iloc[
            indices
        ]  # Second stage of filtering (by cluster selection)
        ######### end hack #########

        if filtered_df.empty:
            st.warning(
                "No data available for the selected points! Try adjusting your selection to receive analytics."
            )
            return

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

        export_options(filtered_df, all_visualizations, title, cluster_fig=fig)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
