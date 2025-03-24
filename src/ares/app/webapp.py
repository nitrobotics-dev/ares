"""
Main file for displaying the Streamlit app. This file contains the main function that defines the order of the sections in the app as well as
state management, error handling, timing, and data export functionality.
"""

import os
import time
import traceback
import typing as t
from collections import defaultdict
from contextlib import contextmanager

import streamlit as st

from ares.app.export_data import export_options
from ares.app.plot_primitives import show_dataframe
from ares.app.sections import (
    data_distributions_section,
    embedding_data_filters_section,
    loading_data_section,
    plot_hero_section,
    robot_array_section,
    state_info_section,
    structured_data_filters_section,
    success_rate_analytics_section,
    time_series_analytics_section,
    video_grid_section,
)
from ares.constants import ARES_DATA_DIR

# top level global variables
title = "ARES Dashboard"
tmp_dump_dir = os.path.join(ARES_DATA_DIR, "webapp_tmp")
section_times: dict[str, float] = defaultdict(float)


######################################################################
# Context managers for error handling and timing
# - `error_context` is used to catch errors in computation and render the error in the app
# - `timer_context` is used to time the execution of a section and print the timing to the console
######################################################################
@contextmanager
def error_context(section_name: str) -> t.Any:
    """
    Context manager for gracefully handling errors in computation of a section.
    Catch the error and render it in the app for easy debugging and readability.
    """
    print(section_name)
    try:
        yield
    except Exception as e:
        st.error(f"Error in {section_name}: {str(e)}\n{traceback.format_exc()}")
        st.write("Stopping execution")
        st.stop()


@contextmanager
def timer_context(section_name: str) -> t.Any:
    """
    Context manager for timing sections, helpful for debugging and performance analysis.
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        section_times[section_name] += elapsed_time


# Main function defining the order of the streamlit subsections
# Note: streamlit displays standalone-strings like `"""..."""` as markdown!
# Use `#` for comments in the streamlit context.
def main() -> None:
    print("Starting main")
    ######################################################################
    # Load data and setup state info
    ######################################################################
    section_loading = "loading data"
    with error_context(section_loading), timer_context(section_loading):
        df = loading_data_section(title, tmp_dump_dir)

    # simple expander for st.session_state information; helpful for debugging
    section_state_info = "state info"
    with error_context(section_state_info), timer_context(section_state_info):
        state_info_section(df)
    st.divider()

    ######################################################################
    # Filter data using structured (selected via buttons, dropdowns, etc.) and embedding (selected via pointer, boxes) filters
    ######################################################################
    section_filters = "structured data filters"
    with error_context(section_filters), timer_context(section_filters):
        structured_filtered_df, active_structured_filters = (
            structured_data_filters_section(df)
        )

    section_embedding_filters = "embedding data filters"
    with (
        error_context(section_embedding_filters),
        timer_context(section_embedding_filters),
    ):
        filtered_df, embedding_figs = embedding_data_filters_section(
            df, structured_filtered_df
        )
        if filtered_df.empty:
            st.warning(
                "No data available for the selected points! Try adjusting your selection to receive analytics."
            )
            return
    st.divider()

    ######################################################################
    # Display a section of the data and the distributions of the data, covering:
    # - general data distribution
    # - success rate
    # - time series trends
    # - video grid of examples
    ######################################################################
    section_data_sample = "data sample"
    with error_context(section_data_sample), timer_context(section_data_sample):
        show_dataframe(
            filtered_df.sample(min(5, len(filtered_df))), title="Data Sample"
        )
    st.divider()

    section_display = "data distributions"
    with error_context(section_display), timer_context(section_display):
        data_distributation_visualizations = data_distributions_section(filtered_df)

    section_success_rate = "success estimate analytics"
    with (
        error_context(section_success_rate),
        timer_context(section_success_rate),
    ):
        success_rate_visualizations = success_rate_analytics_section(filtered_df)
    st.divider()

    section_time_series = "time series trends"
    with error_context(section_time_series), timer_context(section_time_series):
        time_series_visualizations = time_series_analytics_section(filtered_df)
    st.divider()

    section_video_grid = "video grid"
    with error_context(section_video_grid), timer_context(section_video_grid):
        video_grid_section(filtered_df)
    st.divider()

    ######################################################################
    # Create a centralized focus on a single row of data with a 'hero' display
    # - Show the video, annotations, and other relevant data
    # - Create a tabbed interface for different views of the data
    # - Retrieve similar examples based on different metrics
    ######################################################################
    section_plot_hero = "plot hero display"
    with error_context(section_plot_hero), timer_context(section_plot_hero):
        selected_row = plot_hero_section(df, filtered_df)
    st.divider()

    ######################################################################
    # Plot robot arrays showing the distribution of robot actions and states relative to the rest
    # of the dataset. Useful for finding outliers and other interesting patterns.
    ######################################################################
    section_plot_robots = "plot robot arrays"
    with error_context(section_plot_robots), timer_context(section_plot_robots):
        robot_array_visualizations = robot_array_section(selected_row)
    st.divider()

    ######################################################################
    # Export the data and all visualizations to a file or training format.
    # Note: we don't export video grids due to file size.
    ######################################################################
    section_export = "exporting data"
    with error_context(section_export), timer_context(section_export):
        all_visualizations = [
            *data_distributation_visualizations,
            *success_rate_visualizations,
            *time_series_visualizations,
            *robot_array_visualizations,
        ]
        export_options(
            filtered_df,
            active_structured_filters,
            all_visualizations,
            title,
            go_figs=embedding_figs,
        )

    ######################################################################
    # Display the timing report found by the timer context manager
    ######################################################################
    print("\n=== Timing Report ===")
    print(f"Total time: {sum(section_times.values()):.2f} seconds")
    for section, elapsed_time in section_times.items():
        print(f"{section}: {elapsed_time:.2f} seconds")
    print("==================\n")


if __name__ == "__main__":
    main()
