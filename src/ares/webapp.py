import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

title = "Video Analytics Dashboard"


# Streamlit app
def main() -> None:
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    st.title(title)

    # Sidebar filters
    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(pd.Timestamp.now() - pd.Timedelta(days=30), pd.Timestamp.now()),
    )

    if not date_range or len(date_range) != 2:
        st.error("Please select a valid date range")
        return

    # Create mock data
    try:
        metrics_df = pd.DataFrame(
            {
                "date": pd.date_range(start=date_range[0], end=date_range[1], freq="D"),
                "views": np.random.randint(
                    100, 1000, size=(date_range[1] - date_range[0]).days + 1
                ),
                "likes": np.random.randint(
                    10, 100, size=(date_range[1] - date_range[0]).days + 1
                ),
                "comments": np.random.randint(
                    1, 20, size=(date_range[1] - date_range[0]).days + 1
                ),
            }
        )
    except Exception as e:
        st.error(f"Error creating metrics data: {str(e)}")
        return

    if metrics_df.empty:
        st.warning("No data available for the selected date range")
        return

    videos_df = pd.DataFrame(
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
                "/workspaces/ares/data/pi_demos/.DS_Store",
                "/workspaces/ares/data/pi_demos/processed_towel_success.mp4",
            ],
        }
    )

    # Filter videos based on date range
    videos_df = videos_df[
        (videos_df["upload_date"] >= pd.Timestamp(date_range[0]))
        & (videos_df["upload_date"] <= pd.Timestamp(date_range[1]))
    ]

    if videos_df.empty:
        st.warning("No videos found for the selected date range")
        return

    try:
        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            total_views = metrics_df["views"].sum()
            st.metric("Total Views", f"{total_views:,}")

        with col2:
            total_likes = metrics_df["likes"].sum()
            st.metric("Total Likes", f"{total_likes:,}")

        with col3:
            engagement_rate = (
                (total_likes / total_views * 100) if total_views > 0 else 0
            )
            st.metric("Engagement Rate", f"{engagement_rate:.2f}%")

        # Trending metrics chart
        st.subheader("Trending Metrics")
        fig = px.line(
            metrics_df,
            x="date",
            y=["views", "likes", "comments"],
            title="Metrics Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent videos table and display
        st.subheader("Recent Videos")

        # Display videos in columns
        video_cols = st.columns(3)
        for idx, (_, video) in enumerate(videos_df.iterrows()):
            with video_cols[idx % 3]:
                if not pd.isna(video["video_path"]) and video["video_path"].endswith(
                    (".mp4", ".avi", ".mov")
                ):
                    st.video(video["video_path"])
                    st.write(f"**{video['title']}**")
                    st.write(f"Views: {video['views']:,}")
                    st.write(
                        f"Upload Date: {video['upload_date'].strftime('%Y-%m-%d')}"
                    )
                else:
                    st.warning(f"Invalid video path for {video['title']}")

        # Show full table
        st.subheader("Video Details")
        st.dataframe(
            videos_df.drop("video_path", axis=1),
            column_config={
                "video_id": st.column_config.TextColumn("Video ID"),
                "title": st.column_config.TextColumn("Title"),
                "upload_date": st.column_config.DateColumn("Upload Date"),
                "views": st.column_config.NumberColumn("Views", format="%d"),
            },
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()
