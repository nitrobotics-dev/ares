import os
import time
import traceback
from collections import defaultdict
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
from ares.clustering import cluster_embeddings, visualize_clusters
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

tmp_dump_dir = "/tmp/ares_dump"


def initialize_data() -> None:
    """Initialize database connection and load/create embeddings"""
    if "ENGINE" not in st.session_state or "SESSION" not in st.session_state:
        engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
        sess = Session(engine)
        st.session_state.ENGINE = engine
        st.session_state.SESSION = sess

    # Create tmp directory if it doesn't exist
    os.makedirs(tmp_dump_dir, exist_ok=True)
    embeddings_path = os.path.join(tmp_dump_dir, "embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, "clusters.npz")

    # Initialize or load embeddings
    if "embeddings" not in st.session_state:
        if os.path.exists(embeddings_path):
            # Load from disk
            st.session_state.embeddings = np.load(embeddings_path)
            clusters_data = np.load(clusters_path)
            st.session_state.reduced = clusters_data["reduced"]
            st.session_state.labels = clusters_data["labels"]
            st.session_state.probs = clusters_data["probs"]
        else:
            # Create new random data and save to disk
            embeddings = np.random.rand(1000, 2)
            for i in range(3):
                embeddings[i * 200 : (i + 1) * 200] += i

            reduced, labels, probs = cluster_embeddings(embeddings)

            # Save to disk
            np.save(embeddings_path, embeddings)
            np.savez(clusters_path, reduced=reduced, labels=labels, probs=probs)

            # Store in session state
            st.session_state.embeddings = embeddings
            st.session_state.reduced = reduced
            st.session_state.labels = labels
            st.session_state.probs = probs


def initialize_mock_data() -> None:
    """Initialize or load mock data"""
    mock_data_path = os.path.join(tmp_dump_dir, "mock_data.pkl")

    if "MOCK_DATA" not in st.session_state:
        if os.path.exists(mock_data_path):
            # Load from disk
            st.session_state.MOCK_DATA = pd.read_pickle(mock_data_path)
        else:
            # Create new random data
            base_dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
            np.random.seed(42)  # Set seed for reproducibility
            random_offsets = np.random.randint(0, 365, size=365)
            dates = [
                date - pd.Timedelta(days=int(offset))
                for date, offset in zip(base_dates, random_offsets)
            ]

            sampled_paths = np.random.choice(video_paths, size=365)

            mock_data = pd.DataFrame(
                {
                    "creation_time": dates,
                    "length": np.random.randint(1, 100, size=365),
                    "task_success": np.array(
                        [np.random.uniform(i / 365, 1) for i in range(365)]
                    ),
                    "id": [f"vid_{i}" for i in range(365)],
                    "task": [f"Robot Task {i}" for i in np.random.randint(0, 10, 365)],
                    "views": np.random.randint(100, 1000, size=365),
                    "video_path": [
                        f"/workspaces/ares/data/pi_demos/{path}"
                        for path in sampled_paths
                    ],
                }
            )

            # Save to disk
            mock_data.to_pickle(mock_data_path)

            # Store in session state
            st.session_state.MOCK_DATA = mock_data


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


def export_dashboard(
    df: pd.DataFrame, visualizations: list[dict], base_path: str, format: str = "html"
) -> str:
    """Export dashboard including data, visualizations, and analytics.

    Args:
        df: DataFrame containing all data
        visualizations: List of visualization dictionaries with figures and titles
        base_path: Base directory path where file should be saved
        format: Export format ("html", "pdf", "csv", "xlsx")

    Returns:
        Path where file was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(base_path, "exports")
    os.makedirs(export_dir, exist_ok=True)

    base_filename = f"dashboard_export_{timestamp}"
    export_path = os.path.join(export_dir, base_filename)

    if format in ["csv", "xlsx"]:
        # Simple data-only export
        full_path = f"{export_path}.{format}"
        if format == "csv":
            df.to_csv(full_path, index=False)
        else:
            df.to_excel(full_path, index=False)
    else:
        # Full dashboard export
        full_path = f"{export_path}.{format}"
        img_dir = f"{export_path}_files"
        os.makedirs(img_dir, exist_ok=True)

        # Generate HTML content
        html_content = [
            "<html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            ".plot-container { margin: 20px 0; }",
            ".stats-container { margin: 20px 0; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "</style>",
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # Add cluster visualization if available
        if "reduced" in st.session_state:
            cluster_fig, _ = visualize_clusters(
                st.session_state.reduced,
                st.session_state.labels,
                st.session_state.probs,
            )
            cluster_path = os.path.join(img_dir, "clusters.png")
            cluster_fig.write_image(cluster_path)
            html_content.extend(
                [
                    "<h2>Cluster Analysis</h2>",
                    f'<img src="{os.path.basename(img_dir)}/clusters.png" style="max-width:100%">',
                ]
            )

        # Add all visualizations
        html_content.append("<h2>Analytics Visualizations</h2>")
        for i, viz in enumerate(visualizations):
            img_path = os.path.join(img_dir, f"plot_{i}.png")
            viz["figure"].write_image(img_path)
            html_content.extend(
                [
                    f"<div class='plot-container'>",
                    f"<h3>{viz['title']}</h3>",
                    f'<img src="{os.path.basename(img_dir)}/plot_{i}.png" style="max-width:100%">',
                    "</div>",
                ]
            )

        # Add summary statistics
        html_content.extend(
            [
                "<h2>Summary Statistics</h2>",
                "<div class='stats-container'>",
                df.describe().to_html(),
                "</div>",
            ]
        )

        # Add data table
        html_content.extend(
            [
                "<h2>Data Sample</h2>",
                "<div class='data-container'>",
                df.head(100).to_html(),  # First 100 rows
                "</div>",
                "</body></html>",
            ]
        )

        html_content = "\n".join(html_content)

        if format == "html":
            with open(full_path, "w") as f:
                f.write(html_content)
        else:  # PDF
            try:
                import pdfkit

                html_path = f"{export_path}_temp.html"
                with open(html_path, "w") as f:
                    f.write(html_content)
                pdfkit.from_file(html_path, full_path)
                os.remove(html_path)  # Clean up temp file
            except ImportError:
                raise ImportError("Please install pdfkit: pip install pdfkit")

    return full_path


def export_options(filtered_df: pd.DataFrame, visualizations: list[dict]) -> None:
    """Display and handle export controls for the dashboard.

    Args:
        filtered_df: DataFrame to be exported
        visualizations: List of visualization dictionaries
    """
    st.header("Export Options")
    export_col1, export_col2, export_col3, _ = st.columns([1, 1, 1, 1])

    with export_col1:
        export_path = st.text_input(
            "Export Directory",
            value="/tmp",
            help="Directory where exported files will be saved",
        )

    with export_col2:
        export_format = st.selectbox(
            "Export Format",
            options=["html", "pdf", "csv", "xlsx"],
            help="Choose the format for your export. HTML/PDF include visualizations.",
        )

    with export_col3:
        if st.button("Export Dashboard"):
            try:
                with st.spinner(f"Exporting dashboard as {export_format}..."):
                    export_path = export_dashboard(
                        filtered_df, visualizations, export_path, export_format
                    )
                    st.success(f"Dashboard exported successfully to: {export_path}")
            except Exception as e:
                st.error(
                    f"Failed to export dashboard: {str(e)}\n{traceback.format_exc()}"
                )


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


# Streamlit app
def main() -> None:
    print("\n" + "=" * 100 + "\n")
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
        st.header("General Analytics")
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

        st.header("Time Series Analytics")
        time_series_visualizations = generate_time_series_visualizations(
            filtered_df, time_column="ingestion_time"
        )
        create_tabbed_visualizations(
            time_series_visualizations,
            [viz["title"] for viz in time_series_visualizations],
        )

        # show video cards of first 5 rows in a horizontal layout
        st.header("Video Analytics")
        n_videos = min(5, len(filtered_df))  # Take minimum of 5 or available rows
        video_cols = st.columns(n_videos)
        video_paths = [
            os.path.join(PI_DEMO_PATH, path)
            for path in os.listdir(PI_DEMO_PATH)
            if "ds_store" not in path.lower()
        ]

        # Use enumerate to safely iterate through both columns and rows
        for i, (_, row) in enumerate(filtered_df.head(n_videos).iterrows()):
            with video_cols[i]:
                # Safely index into video_paths
                video_path = video_paths[
                    i % len(video_paths)
                ]  # Use modulo to avoid index errors
                inp = {**row.to_dict(), "video_path": video_path}
                display_video_card(inp)

        st.divider()  # Add horizontal line
        # Export controls
        # Collect all visualizations
        all_visualizations = []
        all_visualizations.extend(general_visualizations)
        all_visualizations.extend(success_visualizations)
        all_visualizations.extend(time_series_visualizations)

        export_options(filtered_df, all_visualizations)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
