import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

from ares.viz_helpers import (
    create_bar_plot,
    create_histogram,
    create_line_plot,
    display_video_card,
    show_dataframe,
)

title = "Video Analytics Dashboard"


def initialize_mock_data() -> None:
    """Initialize mock data if it doesn't exist in session state"""
    if "MOCK_METRICS" not in st.session_state:
        # Create date range first
        base_dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
        # Apply random offsets one at a time
        random_offsets = np.random.randint(0, 365, size=365)
        dates = [
            date - pd.Timedelta(days=int(offset))
            for date, offset in zip(base_dates, random_offsets)
        ]

        st.session_state.MOCK_METRICS = pd.DataFrame(
            {
                "date": dates,
                "length": np.random.randint(1, 100, size=365),
                "success": np.array(
                    [np.random.uniform(i / 365, 1) for i in range(365)]
                ),
            }
        )

    if "MOCK_VIDEOS" not in st.session_state:
        st.session_state.MOCK_VIDEOS = pd.DataFrame(
            {
                "video_id": [f"vid_{i}" for i in range(5)],
                "title": [
                    "Robot Task 1",
                    "Robot Task 2",
                    "Robot Task 3",
                    "Robot Task 4",
                    "Robot Task 5",
                ],
                "upload_date": pd.date_range(end=pd.Timestamp.now(), periods=5),
                "views": np.random.randint(100, 1000, size=5),
                "video_path": [
                    "/workspaces/ares/data/pi_demos/processed_toast_fail.mp4",
                    "/workspaces/ares/data/pi_demos/processed_stack_success.mp4",
                    "/workspaces/ares/data/pi_demos/processed_togo_fail.mp4",
                    "/workspaces/ares/data/pi_demos/processed_towel_fail.mp4",
                    "/workspaces/ares/data/pi_demos/processed_towel_success.mp4",
                ],
            }
        )


def filter_metrics(date_range: tuple[datetime.date, datetime.date]) -> pd.DataFrame:
    metrics_df = st.session_state.MOCK_METRICS[
        (st.session_state.MOCK_METRICS["date"] >= pd.Timestamp(date_range[0]))
        & (st.session_state.MOCK_METRICS["date"] <= pd.Timestamp(date_range[1]))
    ].copy()

    if metrics_df.empty:
        st.warning("No data available for the selected date range")
        return pd.DataFrame()

    return metrics_df


def filter_videos(date_range: tuple[datetime.date, datetime.date]) -> pd.DataFrame:
    videos_df = st.session_state.MOCK_VIDEOS.copy()

    videos_df = videos_df[
        (videos_df["upload_date"] >= pd.Timestamp(date_range[0]))
        & (videos_df["upload_date"] <= pd.Timestamp(date_range[1]))
    ]
    return videos_df


# Streamlit app
def main() -> None:
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    st.title(title)

    # Initialize mock data at the start of the app
    initialize_mock_data()

    # Filters
    st.header("Filters")
    col1, _ = st.columns([2, 2])
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(pd.Timestamp("2000-01-01"), pd.Timestamp.now()),
        )

    if not date_range or len(date_range) != 2:
        st.error("Please select a valid date range")
        return

    metrics_df = filter_metrics(date_range)
    if metrics_df.empty:
        return

    videos_df = filter_videos(date_range)

    if videos_df.empty:
        st.warning("No videos found for the selected date range")
        return

    try:
        # Add export controls
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
                    metrics_path, videos_path = export_dataframes(
                        metrics_df, videos_df, export_path
                    )
                    st.success(
                        f"""Data exported successfully:
                    - Metrics: {metrics_path}
                    - Videos: {videos_path}"""
                    )
                except Exception as e:
                    st.error(f"Failed to export data: {str(e)}")

        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_len = metrics_df["length"].mean()
            st.metric("Avg Len", f"{avg_len:.4f}")

        with col2:
            success_rate = metrics_df["success"].mean()
            st.metric("Avg Success", f"{100*success_rate:.4f}%")

        # Trending metrics chart
        st.subheader("Trending Metrics")
        # Group by date and calculate mean length and success rate
        # hack to get count
        metrics_df["count"] = 1
        daily_metrics = (
            metrics_df.groupby("date")
            .agg(
                {
                    "length": "mean",
                    "success": "mean",
                    "count": "sum",  # Count of records per day
                }
            )
            .reset_index()
        )

        # Add rolling averages
        col1 = st.columns(4)[0]
        with col1:
            window_size = st.number_input(
                "Moving Average Window Size",
                min_value=1,
                value=7,
                help="Number of days to use for the moving average calculation",
            )

        daily_metrics["length_ma"] = (
            daily_metrics["length"].rolling(window=window_size).mean()
        )
        daily_metrics["success_ma"] = (
            daily_metrics["success"].rolling(window=window_size).mean()
        )

        # Create two columns for the plots
        plot_col1, plot_col2 = st.columns(2)

        # Length plot
        with plot_col1:
            fig_length = create_line_plot(
                daily_metrics,
                x="date",
                y=["length", "length_ma"],
                title="Daily Length",
                labels={
                    "length": "Length",
                    "length_ma": f"{window_size}-day Moving Average",
                    "date": "Date",
                    "value": "Length",
                },
                colors=["#1f77b4", "#17becf"],
            )
            st.plotly_chart(fig_length, use_container_width=True)

        # Success rate plot
        with plot_col2:
            fig_success = create_line_plot(
                daily_metrics,
                x="date",
                y=["success", "success_ma"],
                title="Daily Success Rate",
                labels={
                    "success": "Success Rate",
                    "success_ma": f"{window_size}-day Moving Average",
                    "date": "Date",
                    "value": "Success Rate",
                },
                colors=["#ff7f0e", "#d62728"],
                y_format=".0%",
            )
            st.plotly_chart(fig_success, use_container_width=True)

        # Length histogram
        fig_hist = create_histogram(
            daily_metrics,
            x="length",
            title="Distribution of Video Lengths",
            labels={"length": "Length", "count": "Number of Videos"},
            color="#2ca02c",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # bar plot of count of rows per day
        fig_count = create_bar_plot(
            daily_metrics,
            x="date",
            y="count",
            title="Daily Video Count",
            labels={"date": "Date", "count": "Number of Videos"},
            color="#9467bd",
        )
        st.plotly_chart(fig_count, use_container_width=True)

        # Recent videos table and display
        st.subheader("Recent Videos")

        # Display videos in columns
        video_cols = st.columns(3)
        for idx, (_, video) in enumerate(videos_df.iterrows()):
            with video_cols[idx % 3]:
                display_video_card(video)

        # Show full table
        show_dataframe(videos_df, "Video Details", hide_columns=["video_path"])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


def export_dataframes(
    metrics_df: pd.DataFrame, videos_df: pd.DataFrame, base_path: str
) -> tuple[str, str]:
    """Export dataframes to CSV files with timestamps.

    Args:
        metrics_df: DataFrame containing metrics data
        videos_df: DataFrame containing videos data
        base_path: Base directory path where files should be saved

    Returns:
        Tuple of (metrics_path, videos_path) where files were saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(base_path, "exports")

    # Create exports directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Save dataframes
    metrics_path = os.path.join(export_dir, f"metrics_{timestamp}.csv")
    videos_path = os.path.join(export_dir, f"videos_{timestamp}.csv")

    metrics_df.to_csv(metrics_path, index=False)
    videos_df.to_csv(videos_path, index=False)

    return metrics_path, videos_path


if __name__ == "__main__":
    main()
