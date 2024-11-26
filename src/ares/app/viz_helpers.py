import io
import json
import os
import random
import uuid
from typing import Any, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import streamlit as st
from Levenshtein import distance as levenshtein_distance

from ares.databases.embedding_database import IndexManager, rollout_to_index_name
from ares.task_utils import PI_DEMO_PATH  # hack


def create_line_plot(
    df: pd.DataFrame,
    x: str,
    y: list[str],
    title: str,
    labels: dict[str, str],
    colors: list[str],
    y_format: str | None = None,
) -> plotly.graph_objects.Figure:
    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        labels=labels,
        color_discrete_sequence=colors,
    )
    layout_args = {
        "yaxis_title": labels.get("value", "Value"),
        "showlegend": True,
        "legend_title_text": "",
    }
    if y_format:
        layout_args["yaxis_tickformat"] = y_format
    fig.update_layout(**layout_args)
    return fig


def create_histogram(
    df: pd.DataFrame,
    x: str,
    title: str,
    labels: dict[str, str],
    color: str,
    nbins: int = 30,
) -> plotly.graph_objects.Figure:
    fig = px.histogram(
        df,
        x=x,
        nbins=nbins,
        title=title,
        labels=labels,
        color_discrete_sequence=[color],
        barmode="overlay",
        marginal="box",
    )
    fig.update_layout(
        xaxis_title=labels.get(x, x),
        yaxis_title="count",
        showlegend=False,
        bargap=0.1,
    )
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    return fig


def create_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    labels: dict[str, str],
    color: str,
) -> plotly.graph_objects.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        labels=labels,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        xaxis_title=labels.get(x, x),
        yaxis_title=labels.get(y, y),
        showlegend=False,
    )
    return fig


def display_video_card(row: pd.Series) -> None:
    if not pd.isna(row["path"]):
        st.video(get_video(row["path"]))
        st.write(f"**{row['id']}**")
        st.write(f"Task: {row['task_language_instruction']}")
        st.write(f"Upload Date: {row['ingestion_time'].strftime('%Y-%m-%d')}")
    else:
        st.warning(f"Invalid video path for {row['id'], row['video_path']}")


def show_dataframe(
    df: pd.DataFrame,
    title: str,
    show_columns: list[str] = None,
    hide_columns: list[str] = None,
) -> None:
    """Helper function to display DataFrames with consistent styling.

    Args:
        df: DataFrame to display
        title: Subheader title for the table
        show_columns: List of column names to show (exclusive with hide_columns)
        hide_columns: List of column names to hide (exclusive with show_columns)
    """
    if show_columns and hide_columns:
        raise ValueError("Cannot specify both show_columns and hide_columns")

    st.subheader(title)

    # Create copy and filter columns
    display_df = df.copy()
    if show_columns:
        display_df = display_df[show_columns]
    elif hide_columns:
        display_df = display_df.drop(hide_columns, axis=1)

    # Auto-generate column configs based on data types
    column_config = {}
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            column_config[col] = st.column_config.DateColumn(
                col.replace("_", " ").title()
            )
        elif pd.api.types.is_numeric_dtype(display_df[col]):
            if "rate" in col.lower() or "percent" in col.lower():
                column_config[col] = st.column_config.NumberColumn(
                    col.replace("_", " ").title(), format="%.2%"
                )
            else:
                column_config[col] = st.column_config.NumberColumn(
                    col.replace("_", " ").title(), format="%g"
                )
        else:
            column_config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title()
            )

    st.dataframe(
        display_df,
        column_config=column_config,
        hide_index=True,
    )


def infer_visualization_type(
    column_name: str,
    data: pd.DataFrame,
    skip_columns: list | None = None,
    max_str_length: int = 500,
) -> dict[str, Any]:
    """Infer the appropriate visualization type based on column characteristics.

    Args:
        column_name: Name of the column to analyze
        data: DataFrame containing the data
        skip_columns: List of column names to skip
        max_str_length: Maximum string length allowed for categorical plots. Longer strings will be skipped.

    Returns:
        dict containing:
            - viz_type: Type of visualization ('line', 'bar', 'histogram', etc.) or None if should be skipped
            - dtype: Data type as string
            - nunique: Number of unique values
    """
    # Get column data type and unique count
    dtype = str(data[column_name].dtype)
    nunique = data[column_name].nunique()

    result = {"viz_type": None, "dtype": dtype, "nunique": nunique}

    # Skip certain columns that shouldn't be plotted
    skip_columns = skip_columns or ["path", "id"]
    if column_name.lower() in skip_columns:
        return result

    # Skip columns with long string values
    if pd.api.types.is_string_dtype(data[column_name]):
        if data[column_name].str.len().max() > max_str_length:
            return result

    # Time series data (look for time/date columns)
    if pd.api.types.is_datetime64_any_dtype(data[column_name]):
        return result  # We'll use this as an x-axis instead

    # Numeric continuous data
    if pd.api.types.is_numeric_dtype(data[column_name]):
        if nunique > 20:  # Arbitrary threshold for continuous vs discrete
            result["viz_type"] = "histogram"
        else:
            result["viz_type"] = "bar"
        return result

    # Categorical data
    if pd.api.types.is_string_dtype(data[column_name]) or nunique < 20:
        result["viz_type"] = "bar"
        return result

    print(
        f"Skipping column {column_name} with dtype {dtype} and nunique {nunique} -- didn't fit into any graph types"
    )
    print(data[column_name].value_counts())
    return result


def generate_success_rate_visualizations(df: pd.DataFrame) -> list[dict]:
    """Generate success rate visualizations for categorical columns."""
    visualizations = []
    categorical_cols = sorted(
        [
            col
            for col in df.columns
            if infer_visualization_type(col, df)["viz_type"] == "bar"
        ]
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
    columns = sorted(df.columns)

    for col in columns:
        viz_info = infer_visualization_type(col, df)
        if col == time_column or viz_info["viz_type"] is None:
            continue

        col_title = col.replace("_", " ").replace("-", " ").title()
        viz_type = viz_info["viz_type"]

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
    visualizations: list[dict] = []
    numeric_cols = sorted(df.select_dtypes(include=["int64", "float64"]).columns)

    for col in numeric_cols:
        viz_info = infer_visualization_type(col, df)
        if col == time_column or viz_info["viz_type"] is None:
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
            display_video_card(dict(row))


def get_video(data: str) -> str | bytes | io.BytesIO | np.ndarray:
    if not isinstance(data, (str, bytes, io.BytesIO, np.ndarray)):
        raise ValueError(f"Invalid video data type: {type(data)}")

    if isinstance(data, str):
        # determine if remote or local
        if data.startswith(
            "http"
        ):  # TODO: add more checks for remote data + local cache
            return data
        else:
            # hack for now --> replace with PI demo vids
            valid_files = [
                f
                for f in os.listdir(PI_DEMO_PATH)
                if not f.startswith(".") and "ds_store" not in f.lower()
            ]
            return os.path.join(PI_DEMO_PATH, random.choice(valid_files))
    else:
        return data


def create_embedding_similarity_visualization(
    row: pd.Series,
    df: pd.DataFrame,
    index_manager: IndexManager,
    feature_type: str,
    n_most_similar: int,
) -> dict:
    """Create visualization for similar trajectories based on embedding feature type."""
    name = rollout_to_index_name(row, feature_type)
    trajectory_key = f"trajectory_{feature_type}"
    distances, ids, matrices = index_manager.search_matrix(
        name,
        np.array(json.loads(row[trajectory_key])),
        n_most_similar + 1,  # to avoid self
    )

    # check index of id_str to see if it matches row.id then remove that index
    idx = np.where(ids == str(row.id))
    if len(idx[0]) != 0:
        idx = idx[0][0]
        distances = np.delete(distances, idx)
        ids = np.delete(ids, idx)
        matrices = np.delete(matrices, idx)

    # Return the data instead of creating the visualization
    return {
        "distances": distances[:n_most_similar],
        "ids": ids[:n_most_similar],
        "matrices": matrices[:n_most_similar] if matrices is not None else None,
    }


def create_similarity_tabs(
    visualizations: list[dict],
    tab_names: list[str],
    df: pd.DataFrame,
) -> None:
    """Create tabs specifically for similarity visualizations."""
    tabs = st.tabs(tab_names)
    for tab, viz_data in zip(tabs, visualizations):
        with tab:
            similar_cols = st.columns(len(viz_data["ids"]))
            for i, (dist, id_str) in enumerate(
                zip(viz_data["distances"], viz_data["ids"])
            ):
                with similar_cols[i]:
                    st.write(f"Distance: {dist:.3f}")
                    # Convert id_str to UUID only if it's not already a UUID
                    search_id = (
                        id_str if isinstance(id_str, uuid.UUID) else uuid.UUID(id_str)
                    )
                    found_rows = df[df["id"] == search_id]
                    if len(found_rows) == 0:
                        st.write(f"No row found for id: {id_str}")
                    else:
                        display_video_card(found_rows.iloc[0])


def show_one_row(
    df: pd.DataFrame,
    idx: int,
    all_vecs: dict,
    show_n: int,
    index_manager: IndexManager,
) -> None:
    """
    Row 1: text
    Row 2: video col, detail + robot array plots
    Row 3: n tabs covering most similar based on state, action, video, text (embedding), text (metric)
    """
    st.write(f"Showing row {idx}")
    row = df.iloc[idx]
    # video card
    col1, col2 = st.columns(2)
    with col1:
        st.video(get_video(row["path"]))
    with col2:
        with st.expander("Row Details", expanded=False):
            for col, val in row.items():
                if len(str(val)) > 1000:
                    continue
                st.write(f"{col}: {val}")
        generate_robot_array_plot_visualizations(
            row, all_vecs, show_n, highlight_idx=idx
        )

    # Row 3: n tabs covering most similar based on state, action, text
    st.write(f"Most similar examples to {row['id']}, based on:")
    tab_names = ["State", "Action", "Text"]
    n_most_similar = 3

    # Get the similarity data
    state_viz_data = create_embedding_similarity_visualization(
        row, df, index_manager, "states", n_most_similar
    )
    action_viz_data = create_embedding_similarity_visualization(
        row, df, index_manager, "actions", n_most_similar
    )
    text_viz_data = create_text_similarity_visualization(row, df, n_most_similar)

    # Create the tabs with the data
    create_similarity_tabs(
        [state_viz_data, action_viz_data, text_viz_data],
        tab_names,
        df,
    )


def create_robot_array_plot(
    robot_array: np.ndarray,
    title_base: str,
    highlight_idx: int | None = None,
    show_n: int | None = None,
    scores: np.ndarray | None = None,
    colorscale: str = "RdYlGn",
) -> plotly.graph_objects.Figure:
    """Plot a 3D array with dimensions (trajectory, timestep, robot_state_dim).

    Args:
        robot_array: Array of shape (trajectory, timestep, robot_state_dim)
        title_base: Base title for the plot
        highlight_idx: Index of trajectory to highlight
        show_n: Number of trajectories to show
        scores: Optional array of scores for each trajectory (0 to 1)
        colorscale: Plotly colorscale to use for scoring
    """
    assert (
        robot_array.ndim == 3
    ), f"robot_array must have 3 dimensions; received shape: {robot_array.shape}"

    if show_n is not None:
        # Sample trajectories (including highlighted idx if specified)
        indices = np.arange(robot_array.shape[0])
        sampled_indices = np.random.choice(
            indices, min(show_n, len(indices)), replace=False
        )
        if highlight_idx is not None:
            sampled_indices = np.unique(np.append(sampled_indices, highlight_idx))

        robot_array = robot_array[sampled_indices]
        if scores is not None:
            scores = scores[sampled_indices]

    # Calculate grid dimensions - try to make it as square as possible
    n_dims = robot_array.shape[2]
    grid_size = int(np.ceil(np.sqrt(n_dims)))

    # Create a grid of columns
    cols = st.columns(grid_size)

    # Create a figure for each dimension of the robot state
    for dim in range(robot_array.shape[2]):
        # Calculate row and column position
        row = dim // grid_size
        col = dim % grid_size

        # If we need a new row of columns
        if col == 0 and dim != 0:
            cols = st.columns(grid_size)

        with cols[col]:
            fig = px.line(
                title=title_base + f" - Dimension {dim}",
                labels={
                    "x": "Relative Timestep",
                    "y": "Value",
                    "color": "Score" if scores is not None else "Trajectory",
                },
            )

            if scores is not None:
                # Color by score mode
                for i in range(len(robot_array)):
                    if highlight_idx is not None and i == highlight_idx:
                        continue  # Skip highlighted trajectory for now

                    color = px.colors.sample_colorscale(colorscale, float(scores[i]))[0]

                    fig.add_scatter(
                        x=list(range(robot_array.shape[1])),
                        y=robot_array[i, :, dim],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.5,
                        name=f"Score: {scores[i]:.2f}",
                        showlegend=(dim == 0 and i < 5),  # Show first few in legend
                        hovertemplate=f"Score: {scores[i]:.2f}<br>Value: %{{y}}<extra></extra>",
                    )
            else:
                # Original background trajectory mode
                mask = np.ones(robot_array.shape[0], dtype=bool)
                if highlight_idx is not None:
                    mask[highlight_idx] = False

                x = np.tile(np.arange(robot_array.shape[1]), mask.sum())
                y = robot_array[mask, :, dim].flatten()
                traj_ids = np.repeat(
                    np.arange(robot_array.shape[0])[mask], robot_array.shape[1]
                )

                fig.add_scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="blue", width=1),
                    opacity=0.3,
                    name="Other Trajectories",
                    legendgroup="other",
                    showlegend=(dim == 0),
                    hovertemplate="Trajectory %{customdata}<br>Value: %{y}<extra></extra>",
                    customdata=traj_ids,
                )

            # Add highlighted trajectory last (on top) if specified
            if highlight_idx is not None and highlight_idx < robot_array.shape[0]:
                fig.add_scatter(
                    x=list(range(robot_array.shape[1])),
                    y=robot_array[highlight_idx, :, dim],
                    mode="lines",
                    name=f"Trajectory {highlight_idx}",
                    line=dict(color="red", width=3),
                    opacity=1.0,
                    showlegend=(dim == 0),
                )

            fig.update_layout(showlegend=False)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{title_base}-dim-{dim}-{len(robot_array)}-{highlight_idx}",
            )

    return fig


def generate_robot_array_plot_visualizations(
    row: pd.Series,
    all_vecs: dict,
    show_n: int = 1000,
    keys: list[str] | None = None,
    highlight_idx: int | None = None,
    scores: dict[str, np.ndarray] | None = None,
) -> list[plotly.graph_objects.Figure]:
    if highlight_idx is not None and scores is not None:
        raise ValueError(
            f"Cannot provide both highlight_idx and scores. Received highlight_idx: {highlight_idx}, scores: {scores}"
        )
    keys = keys or ["states", "actions"]
    figs = []
    for key in keys:
        name_key = row.dataset_name + "-" + row.robot_embodiment + "-" + key
        # HACK: if key not found, substitute random key
        if name_key not in all_vecs:
            print(f"Key {name_key} not found in all_vecs!! substituting random")
            name_key = list(all_vecs.keys())[0]
            if highlight_idx is not None and highlight_idx >= len(all_vecs[name_key]):
                print(f"also substituting highlight_idx")
                highlight_idx = np.random.choice(len(all_vecs[name_key]))

        these_vecs = all_vecs[name_key]
        these_scores = scores.get(name_key) if scores else None

        with st.expander(f"Trajectory {key.title()} Display", expanded=False):
            fig = create_robot_array_plot(
                these_vecs,
                title_base=f"Trajectory {key.title()} Display",
                highlight_idx=highlight_idx,
                show_n=show_n,
                scores=these_scores,
            )
            figs.append(fig)
    return figs


def create_text_similarity_visualization(
    row: pd.Series,
    df: pd.DataFrame,
    n_most_similar: int,
) -> dict:
    """Create visualization for similar trajectories based on text similarity of task instructions."""
    # Get the reference instruction
    reference_instruction = row["task_language_instruction"]

    # Calculate distances for all instructions
    distances = df["task_language_instruction"].apply(
        lambda x: levenshtein_distance(reference_instruction, x)
    )

    # Get indices of most similar (excluding self)
    sorted_indices = distances.argsort()
    # Remove self from results (where distance = 0)
    sorted_indices = sorted_indices[distances[sorted_indices] > 0]
    top_indices = sorted_indices[:n_most_similar]

    return {
        "distances": distances[top_indices].values,
        "ids": df.iloc[top_indices]["id"].values,
        "matrices": None,  # Keeping consistent with other similarity visualization returns
    }
