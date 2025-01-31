import pandas as pd
import streamlit as st

from ares.app.data_analysis import generate_automatic_visualizations
from ares.app.filter_helpers import (
    create_embedding_data_filter_display,
    select_row_from_df_user,
    structured_data_filters_display,
)
from ares.app.hero_display import show_hero_display
from ares.app.init_data import display_state_info, initialize_data
from ares.app.plot_primitives import show_dataframe
from ares.app.viz_helpers import (
    annotation_statistics,
    create_tabbed_visualizations,
    display_video_grid,
    generate_robot_array_plot_visualizations,
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    total_statistics,
)
from ares.databases.embedding_database import META_INDEX_NAMES


def load_data(tmp_dump_dir: str) -> pd.DataFrame:
    """Load data from session state or initialize if needed."""
    initialize_data(tmp_dump_dir)
    return st.session_state.df


def loading_data_section(title: str, tmp_dump_dir: str) -> pd.DataFrame:
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    st.title(title)
    return load_data(tmp_dump_dir)


def state_info_section(df: pd.DataFrame) -> None:
    display_state_info()
    total_statistics(df)
    annotation_statistics(st.session_state.annotations_db)


def structured_data_filters_section(df: pd.DataFrame) -> pd.DataFrame:
    # Structured data filters
    st.header(f"Data Filters")
    structured_filtered_df = structured_data_filters_display(df, debug=False)
    st.write(
        f"Selected {len(structured_filtered_df)} rows out of {len(df)} total via structured data filters"
    )
    if len(structured_filtered_df) == 0:
        st.warning("No data matches the structured filters!")
    return structured_filtered_df


def embedding_data_filters_section(
    df: pd.DataFrame,
    structured_filtered_df: pd.DataFrame,
) -> pd.DataFrame:
    st.subheader(f"Embedding Filters")
    embedding_figs = dict()
    embedding_filtered_dfs = []

    # Get filtered dataframes for each embedding
    for raw_data_key in META_INDEX_NAMES:
        st.write(f"**Filtering on {raw_data_key}**")
        filtered_df, cluster_fig = create_embedding_data_filter_display(
            df=df,  # Pass original df each time
            id_key="id",
            raw_data_key=raw_data_key,
            kept_ids=structured_filtered_df["id"].apply(str).tolist(),
        )
        embedding_filtered_dfs.append(filtered_df)
        embedding_figs[raw_data_key] = cluster_fig

    # Combine all filtered dataframes (AND operation)
    if embedding_filtered_dfs:
        all_filtered_ids = set(embedding_filtered_dfs[0]["id"])
        for filtered_df in embedding_filtered_dfs[1:]:
            all_filtered_ids &= set(filtered_df["id"])

        # Final filtered dataframe combines structured and embedding filters
        filtered_df = structured_filtered_df[
            structured_filtered_df["id"].isin(all_filtered_ids)
        ]
    else:
        filtered_df = structured_filtered_df
    return filtered_df, embedding_figs


def data_distributions_section(filtered_df: pd.DataFrame) -> list[dict]:
    max_x_bar_options = 100
    # Create overview of all data
    st.header("Distribution Analytics")
    general_visualizations = generate_automatic_visualizations(
        filtered_df,
        time_column="ingestion_time",
        max_x_bar_options=max_x_bar_options,
    )
    general_visualizations = sorted(general_visualizations, key=lambda x: x["title"])
    create_tabbed_visualizations(
        general_visualizations, [viz["title"] for viz in general_visualizations]
    )
    return general_visualizations


def success_rate_analytics_section(filtered_df: pd.DataFrame) -> list[dict]:
    st.header("Success Estimate Analytics")
    success_visualizations = generate_success_rate_visualizations(filtered_df)
    create_tabbed_visualizations(
        success_visualizations, [viz["title"] for viz in success_visualizations]
    )
    return success_visualizations


def time_series_analytics_section(filtered_df: pd.DataFrame) -> list[dict]:
    st.header("Time Series Trends")
    time_series_visualizations = generate_time_series_visualizations(
        filtered_df, time_column="ingestion_time"
    )
    create_tabbed_visualizations(
        time_series_visualizations,
        [viz["title"] for viz in time_series_visualizations],
    )
    return time_series_visualizations


def video_grid_section(filtered_df: pd.DataFrame) -> None:
    # show video cards of first 5 rows in a horizontal layout
    st.header("Rollout Examples")
    n_videos = 5
    display_rows = pd.concat(
        {k: v.head(1) for k, v in filtered_df.groupby("dataset_name")}
    )
    if len(display_rows) < n_videos:
        # get enough videos to fill n_videos that arent already in display_rows
        extra_rows = filtered_df.head(n_videos)
        # remove rows that are already in display_rows
        extra_rows = extra_rows[~extra_rows.id.isin(display_rows.id)]
        display_rows = pd.concat([display_rows, extra_rows])
    display_video_grid(display_rows, lazy_load=True)


def plot_hero_section(
    df: pd.DataFrame, filtered_df: pd.DataFrame
) -> tuple[list[dict], pd.Series]:
    st.header("Rollout Display")
    # initialize or persist selected row in state
    select_row_from_df_user(filtered_df)
    selected_row = st.session_state.get("selected_row")

    if selected_row is not None:
        show_dataframe(pd.DataFrame([selected_row]), title="Selected Row")
        st.write(f"Selected row ID: {selected_row.id}")
        hero_visualizations = show_hero_display(
            df,  # compare selected row from filtered_df to all rows in df
            selected_row,
            st.session_state.all_vecs,
            index_manager=st.session_state.INDEX_MANAGER,
            lazy_load=False,
            retrieve_n_most_similar=10,
        )
    else:
        st.info("Please select a row to display details")
    return hero_visualizations, selected_row


def robot_array_section(
    filtered_df: pd.DataFrame, selected_row: pd.Series
) -> list[dict]:
    if st.button("Generate Robot Array Plots", key="robot_array_plots_button"):
        st.header("Robot Array Display")
        # Number of trajectories to display in plots
        robot_array_visualizations = generate_robot_array_plot_visualizations(
            selected_row,  # need row to select dataset/robot embodiment of trajectories
            st.session_state.all_vecs,
            show_n=1000,
        )
    else:
        st.write("No robot array plots generated")
        robot_array_visualizations = []
    return robot_array_visualizations
