import os
import time
import traceback
import typing as t
from collections import defaultdict
from contextlib import contextmanager

import streamlit as st

from ares.app.export_data import export_options
from ares.app.plot_primitives import show_dataframe
from ares.app.webapp_sections import (
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

# top level variables
title = "ARES Dashboard"
tmp_dump_dir = os.path.join(ARES_DATA_DIR, "tmp2")
section_times: dict[str, float] = defaultdict(float)


@contextmanager
def filter_error_context(section_name: str) -> t.Any:
    """Context manager for handling filter operation errors."""
    try:
        yield
    except Exception as e:
        st.error(f"Error in {section_name}: {str(e)}\n{traceback.format_exc()}")
        st.write("Stopping execution")
        st.stop()
        return None


@contextmanager
def timer_context(section_name: str) -> t.Any:
    """Context manager for timing sections."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        section_times[section_name] += elapsed_time


# Main function defining the order of the streamlit subsections
# Note: streamlit displays standalone-strings like `"""..."""` as markdown! Use `#` for comments.
def main() -> None:
    ######################################################################
    # Load data and setup state info
    ######################################################################
    # Define section names
    section_loading = "loading data"
    with filter_error_context(section_loading), timer_context(section_loading):
        df = loading_data_section(title, tmp_dump_dir)

    section_state_info = "state info"
    with filter_error_context(section_state_info), timer_context(section_state_info):
        state_info_section(df)
    st.divider()

    ######################################################################
    # Filter data using structured (selected via buttons, dropdowns, etc.) and embedding (selected via pointer, boxes) filters
    ######################################################################
    section_filters = "structured data filters"
    with filter_error_context(section_filters), timer_context(section_filters):
        structured_filtered_df = structured_data_filters_section(df)

    section_embedding_filters = "embedding data filters"
    with (
        filter_error_context(section_embedding_filters),
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
    with filter_error_context(section_data_sample), timer_context(section_data_sample):
        show_dataframe(
            filtered_df.sample(min(5, len(filtered_df))), title="Data Sample"
        )
    st.divider()

    section_display = "data distributions"
    with filter_error_context(section_display), timer_context(section_display):
        data_distributation_visualizations = data_distributions_section(filtered_df)

    section_success_rate = "success estimate analytics"
    with (
        filter_error_context(section_success_rate),
        timer_context(section_success_rate),
    ):
        success_rate_visualizations = success_rate_analytics_section(filtered_df)
    st.divider()

    section_time_series = "time series trends"
    with filter_error_context(section_time_series), timer_context(section_time_series):
        time_series_visualizations = time_series_analytics_section(filtered_df)
    st.divider()

    section_video_grid = "video grid"
    with filter_error_context(section_video_grid), timer_context(section_video_grid):
        video_grid_section(filtered_df)
    st.divider()

    ######################################################################
    # Create a centralized focus on a single row of data with a 'hero' display
    # - Show the video, annotations, and other relevant data
    # - Create a tabbed interface for different views of the data
    # - Retrieve similar examples based on different metrics
    ######################################################################
    section_plot_hero = "plot hero display"
    with filter_error_context(section_plot_hero), timer_context(section_plot_hero):
        hero_visualizations, selected_row = plot_hero_section(df, filtered_df)
    st.divider()

    ######################################################################
    # Plot robot arrays showing the distribution of robot actions and states relative to the rest
    # of the dataset. Useful for finding outliers and other interesting patterns.
    ######################################################################
    section_plot_robots = "plot robot arrays"
    with filter_error_context(section_plot_robots), timer_context(section_plot_robots):
        robot_array_visualizations = robot_array_section(filtered_df, selected_row)
    st.divider()

    ######################################################################
    # Export the data and all visualizations to a file or training format.
    ######################################################################
    section_export = "exporting data"
    with filter_error_context(section_export), timer_context(section_export):
        # TODO: add structured data filters to export
        all_visualizations = [
            *data_distributation_visualizations,
            *success_rate_visualizations,
            *time_series_visualizations,
            *hero_visualizations,
            *robot_array_visualizations,
        ]
        export_options(
            filtered_df,
            all_visualizations,
            title,
            cluster_fig=embedding_figs.get(next(iter(embedding_figs))),
        )

    ######################################################################
    # Display the timing report found by the context managers
    ######################################################################
    print("\n=== Timing Report ===")
    print(f"Total time: {sum(section_times.values()):.2f} seconds")
    for section, elapsed_time in section_times.items():
        print(f"{section}: {elapsed_time:.2f} seconds")
    print("==================\n")


if __name__ == "__main__":
    main()
