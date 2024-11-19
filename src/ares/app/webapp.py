import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow_datasets as tfds
from sqlalchemy import Engine, case, func, text
from sqlalchemy.orm import Session
from sqlmodel import SQLModel, select
from tqdm import tqdm

from ares.app.viz_helpers import (
    create_bar_plot,
    create_histogram,
    create_line_plot,
    display_video_card,
    show_dataframe,
)
from ares.configs.base import Rollout
from ares.databases.structured_database import (
    SQLITE_PREFIX,
    TEST_ROBOT_DB_PATH,
    RolloutSQLModel,
    add_rollout,
    setup_database,
)
from ares.task_utils import PI_DEMO_PATH

title = "Video Analytics Dashboard"
video_paths = list(os.listdir(PI_DEMO_PATH))


def initialize_data() -> None:
    """Initialize database connection if it doesn't exist in session state"""
    if "ENGINE" not in st.session_state or "SESSION" not in st.session_state:
        engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
        sess = Session(engine)
        st.session_state.ENGINE = engine
        st.session_state.SESSION = sess


def initialize_mock_data() -> None:
    """Initialize mock data if it doesn't exist in session state"""
    if "MOCK_DATA" not in st.session_state:
        # Create date range first
        base_dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
        # Apply random offsets one at a time
        random_offsets = np.random.randint(0, 365, size=365)
        dates = [
            date - pd.Timedelta(days=int(offset))
            for date, offset in zip(base_dates, random_offsets)
        ]

        # Sample video paths randomly
        sampled_paths = np.random.choice(video_paths, size=365)

        st.session_state.MOCK_DATA = pd.DataFrame(
            {
                "creation_time": dates,
                "length": np.random.randint(1, 100, size=365),
                "success": np.array(
                    [np.random.uniform(i / 365, 1) for i in range(365)]
                ),
                "video_id": [f"vid_{i}" for i in range(365)],
                "task": [f"Robot Task {i}" for i in np.random.randint(0, 10, 365)],
                "views": np.random.randint(100, 1000, size=365),
                "video_path": [
                    f"/workspaces/ares/data/pi_demos/{path}" for path in sampled_paths
                ],
            }
        )


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


def export_dataframe(df: pd.DataFrame, base_path: str) -> str:
    """Export dataframe to CSV file with timestamp.

    Args:
        df: DataFrame containing all data
        base_path: Base directory path where file should be saved

    Returns:
        Path where file was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(base_path, "exports")

    # Create exports directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Save dataframe
    export_path = os.path.join(export_dir, f"video_analytics_{timestamp}.csv")
    df.to_csv(export_path, index=False)

    return export_path


def export_controls(filtered_df: pd.DataFrame) -> None:
    """Display and handle export controls for the dataframe.

    Args:
        filtered_df: DataFrame to be exported
    """
    export_col1, export_col2, _ = st.columns([1, 1, 2])
    with export_col1:
        export_path = st.text_input(
            "Export Directory",
            value="/tmp",
            help="Directory where exported files will be saved",
        )
    with export_col2:
        if st.button("Export Data"):
            try:
                export_path = export_dataframe(filtered_df, export_path)
                st.success(f"Data exported successfully to: {export_path}")
            except Exception as e:
                st.error(f"Failed to export data: {str(e)}\n{traceback.format_exc()}")


def infer_visualization_type(
    column_name: str, data: pd.DataFrame, skip_columns: list | None = None
) -> str | None:
    """Infer the appropriate visualization type based on column characteristics.

    Args:
        column_name: Name of the column to analyze
        data: DataFrame containing the data

    Returns:
        str | None: Type of visualization ('line', 'bar', 'histogram', etc.) or None if the column should be skipped
    """
    # Skip certain columns that shouldn't be plotted
    skip_columns = skip_columns or ["video_path", "video_id", "id"]
    if column_name.lower() in skip_columns:
        return None

    # Get column data type
    dtype = data[column_name].dtype
    nunique = data[column_name].nunique()

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


def generate_automatic_visualizations(
    df: pd.DataFrame, time_column: str = "creation_time"
) -> list[dict]:
    """Generate visualizations automatically based on data types.

    Args:
        df: DataFrame to visualize
        time_column: Name of the column to use for time series plots

    Returns:
        list[dict]: List of visualization configurations
    """
    visualizations = []

    # First, handle time series for numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if col == time_column or infer_visualization_type(col, df) is None:
            continue

        # Time series plot
        visualizations.append(
            {
                "figure": create_line_plot(
                    df,
                    x=time_column,
                    y=[col],
                    colors=["#1f77b4"],
                    title=f"{col.title()} Over Time",
                    labels={
                        col: col.title(),
                        time_column: time_column.title(),
                        "value": col.title(),
                    },
                    y_format=".0%" if "success" in col.lower() else None,
                ),
                "title": f"{col.title()} Trends",
            }
        )

        # Distribution plot
        if infer_visualization_type(col, df) == "histogram":
            visualizations.append(
                {
                    "figure": create_histogram(
                        df,
                        x=col,
                        color="#1f77b4",
                        title=f"Distribution of {col.title()}",
                        labels={col: col.title(), "count": "Count"},
                    ),
                    "title": f"{col.title()} Distribution",
                }
            )

    # Handle categorical columns
    categorical_cols = [
        col for col in df.columns if infer_visualization_type(col, df) == "bar"
    ]

    for col in categorical_cols:
        # Aggregate by category
        agg_data = (
            df.groupby(col).agg({"success": "mean", time_column: "count"}).reset_index()
        )

        # Count by category
        visualizations.append(
            {
                "figure": create_bar_plot(
                    agg_data,
                    x=col,
                    y=time_column,
                    color="#1f77b4",
                    title=f"Count by {col.title()}",
                    labels={col: col.title(), time_column: "Count"},
                ),
                "title": f"{col.title()} Distribution",
            }
        )

        # Success rate by category (if applicable)
        if "success" in df.columns:
            visualizations.append(
                {
                    "figure": create_bar_plot(
                        agg_data,
                        x=col,
                        y="success",
                        color="#1f77b4",
                        title=f"Success Rate by {col.title()}",
                        labels={col: col.title(), "success": "Success Rate"},
                    ),
                    "title": f"{col.title()} Success Rate",
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


# Streamlit app
def main() -> None:
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    st.title(title)

    # Initialize mock data at the start of the app
    # initialize_mock_data()
    initialize_data()

    # Get filtered dataframe
    # filtered_df = filter_data_controls()
    # if filtered_df.empty:
    #     return

    # just get df from sess, engine in state, drop "unnamed" columns
    df = st.session_state.SESSION.exec(select(RolloutSQLModel)).all()
    filtered_df = df[[c for c in df.columns if "unnamed" not in c.lower()]]

    try:
        # Export controls
        export_controls(filtered_df)

        # Generate automatic visualizations
        visualizations = generate_automatic_visualizations(filtered_df)

        # Create tabs
        tab_names = [viz["title"] for viz in visualizations]
        create_tabbed_visualizations(visualizations, tab_names)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
