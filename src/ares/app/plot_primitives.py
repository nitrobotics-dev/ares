import io
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ares.task_utils import PI_DEMO_PATH


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


def create_robot_array_plot(
    robot_array: np.ndarray,
    title_base: str,
    highlight_idx: int | None = None,
    show_n: int | None = None,
    scores: np.ndarray | None = None,
    colorscale: str = "RdYlGn",
) -> plotly.graph_objects.Figure:
    assert (
        robot_array.ndim == 3
    ), f"robot_array must have 3 dimensions; received shape: {robot_array.shape}"

    if show_n is not None:
        indices = np.arange(robot_array.shape[0])
        sampled_indices = np.random.choice(
            indices, min(show_n, len(indices), 1000), replace=False
        )
        if highlight_idx is not None:
            sampled_indices = np.unique(np.append(sampled_indices, highlight_idx))
        robot_array = robot_array[sampled_indices]

    # limit number of timesteps to 100 per trace
    if robot_array.shape[1] > 100:
        step = robot_array.shape[1] // 100
        robot_array = robot_array[:, ::step, :]

    # limit scores to sampled indices
    if scores is not None:
        scores = scores[sampled_indices]
        scores = scores[::step]

    # Create subplots - one for each dimension
    n_dims = robot_array.shape[2]
    fig = make_subplots(
        rows=n_dims,
        cols=1,
        subplot_titles=[f"Dimension {i+1}" for i in range(n_dims)],
        shared_xaxes=False,
        vertical_spacing=0.05,
    )
    fig.update_layout(title=title_base, showlegend=True)

    traces = []
    for dim in range(n_dims):
        dim_traces = []  # Create a new list for this dimension's traces
        if scores is not None:
            for i in range(len(robot_array)):
                if highlight_idx is not None and i == highlight_idx:
                    continue
                color = px.colors.sample_colorscale(colorscale, float(scores[i]))[0]
                dim_traces.append(
                    go.Scatter(
                        x=list(range(robot_array.shape[1])),
                        y=robot_array[i, :, dim],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.5,
                        name=f"Score: {scores[i]:.2f}",
                        showlegend=i < 5
                        and dim == 0,  # Only show legend for first dimension
                        hovertemplate=f"Score: {scores[i]:.2f}<br>Value: %{{y}}<extra></extra>",
                    ),
                )
        else:
            mask = np.ones(robot_array.shape[0], dtype=bool)
            if highlight_idx is not None:
                mask[highlight_idx] = False

            x = np.tile(np.arange(robot_array.shape[1]), mask.sum())
            y = robot_array[mask, :, dim].flatten()
            traj_ids = np.repeat(
                np.arange(robot_array.shape[0])[mask], robot_array.shape[1]
            )

            dim_traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="blue", width=1),
                    opacity=0.3,
                    name="Other Trajectories",
                    legendgroup="other",
                    showlegend=dim == 0,  # Only show legend for first dimension
                    hovertemplate="Trajectory %{customdata}<br>Value: %{y}<extra></extra>",
                    customdata=traj_ids,
                ),
            )

        if highlight_idx is not None and highlight_idx < robot_array.shape[0]:
            dim_traces.append(
                go.Scatter(
                    x=list(range(robot_array.shape[1])),
                    y=robot_array[highlight_idx, :, dim],
                    mode="lines",
                    name=f"Trajectory {highlight_idx}",
                    line=dict(color="red", width=3),
                    opacity=1.0,
                    showlegend=dim == 0,  # Only show legend for first dimension
                ),
            )

        # Add traces for this dimension to its subplot
        fig.add_traces(dim_traces, rows=dim + 1, cols=1)

    # Update layout
    fig.update_layout(
        height=250 * n_dims,  # Adjust height based on number of dimensions
        yaxis_title="Value",
        xaxis_title="Relative Timestep",
        hovermode="closest",
    )
    return fig


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
