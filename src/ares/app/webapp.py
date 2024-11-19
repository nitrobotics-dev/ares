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
    create_histogram,
    create_line_plot,
    display_video_card,
    show_dataframe,
)
from ares.clustering import visualize_clusters
from ares.databases.structured_database import RolloutSQLModel
from ares.task_utils import PI_DEMO_PATH

title = "ARES Dashboard"
video_paths = list(os.listdir(PI_DEMO_PATH))

tmp_dump_dir = "/tmp/ares_dump"


def filter_data_controls() -> pd.DataFrame:
    """Display filter controls and return filtered data.

    Returns:
        pd.DataFrame: Filtered dataframe based on selected date range
    """
    st.header("Filters")
    col1, _ = st.columns([2, 2])
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(pd.Timestamp("2000-01-01"), pd.Timestamp.now()),
        )

    if not date_range or len(date_range) != 2:
        st.error("Please select a valid date range")
        return pd.DataFrame()

    # Filter the data based on date range
    filtered_df = st.session_state.MOCK_DATA[
        (st.session_state.MOCK_DATA["creation_time"] >= pd.Timestamp(date_range[0]))
        & (st.session_state.MOCK_DATA["creation_time"] <= pd.Timestamp(date_range[1]))
    ].copy()

    if filtered_df.empty:
        st.warning("No data available for the selected date range")
        return pd.DataFrame()

    return filtered_df


def infer_visualization_type(
    column_name: str,
    data: pd.DataFrame,
    skip_columns: list | None = None,
    max_str_length: int = 500,
) -> str | None:
    """Infer the appropriate visualization type based on column characteristics.

    Args:
        column_name: Name of the column to analyze
        data: DataFrame containing the data
        skip_columns: List of column names to skip
        max_str_length: Maximum string length allowed for categorical plots. Longer strings will be skipped.

    Returns:
        str | None: Type of visualization ('line', 'bar', 'histogram', etc.) or None if the column should be skipped
    """
    # Skip certain columns that shouldn't be plotted
    skip_columns = skip_columns or ["path", "id"]
    if column_name.lower() in skip_columns:
        return None

    # Get column data type
    dtype = data[column_name].dtype
    nunique = data[column_name].nunique()

    # Skip columns with long string values
    if pd.api.types.is_string_dtype(dtype):
        if data[column_name].str.len().max() > max_str_length:
            return None

    # Time series data (look for time/date columns)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return None  # We'll use this as an x-axis instead

    # Numeric continuous data
    if pd.api.types.is_numeric_dtype(dtype):
        if nunique > 20:  # Arbitrary threshold for continuous vs discrete
            return "histogram"
        else:
            return "bar"

    # Categorical data
    if pd.api.types.is_string_dtype(dtype) or nunique < 20:
        return "bar"

    print(
        f"Skipping column {column_name} with dtype {dtype} and nunique {nunique} -- didn't fit into any graph types"
    )
    print(data[column_name].value_counts())
    return None


def generate_success_rate_visualizations(df: pd.DataFrame) -> list[dict]:
    """Generate success rate visualizations for categorical columns."""
    visualizations = []
    categorical_cols = sorted(
        [col for col in df.columns if infer_visualization_type(col, df) == "bar"]
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


def generate_automatic_visualizations(
    df: pd.DataFrame, time_column: str = "creation_time"
) -> list[dict]:
    """Generate visualizations automatically based on data types."""
    visualizations = []

    # Sort all column names
    columns = sorted(df.columns)

    for col in columns:
        if col == time_column or infer_visualization_type(col, df) is None:
            continue

        col_title = col.replace("_", " ").replace("-", " ").title()
        viz_type = infer_visualization_type(col, df)

        if viz_type == "histogram":
            visualizations.append(
                {
                    "figure": create_histogram(
                        df,
                        x=col,
                        color="#1f77b4",
                        title=f"Distribution of {col_title}",
                        labels={col: col_title, "count": "Count"},
                    ),
                    "title": f"{col_title} Distribution",
                }
            )
        elif viz_type == "bar":
            agg_data = df.groupby(col).agg({time_column: "count"}).reset_index()
            visualizations.append(
                {
                    "figure": create_bar_plot(
                        agg_data,
                        x=col,
                        y=time_column,
                        color="#1f77b4",
                        title=f"Count by {col_title}",
                        labels={col: col_title, time_column: "Count"},
                    ),
                    "title": f"{col_title} Distribution",
                }
            )

    return visualizations


def generate_time_series_visualizations(
    df: pd.DataFrame, time_column: str = "creation_time"
) -> list[dict]:
    """Generate time series visualizations for numeric columns."""
    visualizations = []
    numeric_cols = sorted(df.select_dtypes(include=["int64", "float64"]).columns)

    for col in numeric_cols:
        if col == time_column or infer_visualization_type(col, df) is None:
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


def display_video_grid(filtered_df: pd.DataFrame, max_videos: int = 5) -> None:
    """Display a grid of video cards for the first N rows of the dataframe.

    Args:
        filtered_df: DataFrame containing rollout data
        max_videos: Maximum number of videos to display in the grid
    """
    st.header("Rollout Examples")
    n_videos = min(max_videos, len(filtered_df))
    video_cols = st.columns(n_videos)

    video_paths = [
        os.path.join(PI_DEMO_PATH, path)
        for path in os.listdir(PI_DEMO_PATH)
        if "ds_store" not in path.lower()
    ]

    for i, (_, row) in enumerate(filtered_df.head(n_videos).iterrows()):
        with video_cols[i]:
            video_path = video_paths[i % len(video_paths)]
            inp = {**row.to_dict(), "video_path": video_path}
            display_video_card(inp)


# Streamlit app
def main() -> None:
    print("\n" + "=" * 100 + "\n")
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    st.title(title)

    # Initialize mock data at the start of the app
    # initialize_mock_data(tmp_dump_dir)
    initialize_data(tmp_dump_dir)

    # Get filtered dataframe
    # filtered_df = filter_data_controls()
    # if filtered_df.empty:
    #     return

    # just get df from sess, engine in state, drop "unnamed" columns
    df = pd.read_sql(select(RolloutSQLModel), st.session_state.ENGINE)
    df = df[[c for c in df.columns if "unnamed" not in c.lower()]]

    try:
        # Create the visualization using state data
        fig, cluster_df = visualize_clusters(
            st.session_state.reduced, st.session_state.labels, st.session_state.probs
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
            for i in range(len(selection["points"])):
                if selection["points"][i]["curve_number"] == 0:
                    indices.append(selection["point_indices"][i])
        else:
            # no selection made, use all points
            indices = np.arange(len(st.session_state.reduced))
        indices = np.unique(np.array(indices) % available_dataset_pts)
        n_pts = len(indices)
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
        filtered_df = df.iloc[indices]
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
