"""
Cluster embeddings using UMAP for dimensionality reduction and HDBSCAN for clustering.
"""

import json
import os
from typing import Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap

from ares.app.plot_primitives import create_line_plot

# At the top with other imports
SELECTION_FILE = "/workspaces/ares/data/tmp/selected_points.json"


def cluster_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster embeddings using UMAP for dimensionality reduction and HDBSCAN for clustering.

    Args:
        embeddings: Input embeddings array of shape (n_samples, n_dimensions)
        n_neighbors: UMAP parameter for local neighborhood size
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        random_state: Random seed for reproducibility

    Returns:
        reduced_embeddings: UMAP-reduced embeddings (3D)
        cluster_labels: Cluster assignments for each embedding
        probabilities: Cluster membership probabilities
    """
    # Reduce dimensionality to 2D for visualization
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, random_state=random_state
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Perform clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_
    return reduced_embeddings, cluster_labels, probabilities


def visualize_clusters(
    reduced_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    raw_data: list,
    ids: Optional[list] = None,
    custom_data_keys: Optional[list] = None,
    keep_mask: Optional[list] = None,
) -> Tuple[go.Figure, pd.DataFrame, dict[str, int | list[int]]]:
    """
    Create an interactive 2D visualization of the clustered embeddings.
    Returns figure and dataframe for selection tracking.

    Args:
        reduced_embeddings: UMAP-reduced embeddings (2D)
        cluster_labels: Cluster assignments for each embedding
        raw_data: Original text/data for each point
        ids: Optional list of IDs for each point
        keep_mask: Optional mask to gray out some points
    """
    # Initialize trace counter and mapping at the start
    current_trace = 0
    trace_mapping: dict[str, int | list[int]] = {}
    custom_data_keys = custom_data_keys or ["raw_data", "id"]

    n_clusters = len(np.unique(cluster_labels))
    colors = (
        px.colors.qualitative.Set1
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Dark2
    )

    if n_clusters > len(colors):
        print(
            f"Warning: More clusters ({n_clusters}) than available colors ({len(colors)}). Colors will be reused."
        )
    colors = colors[:n_clusters]

    df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "cluster": [str(x) if x != -1 else "Noise" for x in cluster_labels],
            "point_index": range(len(cluster_labels)),
            "raw_data": [str(x)[:100] for x in raw_data],  # HACK
            "id": ids if ids is not None else range(len(cluster_labels)),
            "x_coord": reduced_embeddings[:, 0].round(3),
            "y_coord": reduced_embeddings[:, 1].round(3),
            "masked": False,
        }
    )

    # If mask is provided, update the masked column
    if keep_mask is not None:
        mask_array = np.zeros(len(df), dtype=bool)
        mask_array[keep_mask] = True
        df["masked"] = ~mask_array

    # If mask is provided, create two separate dataframes
    if keep_mask is not None:
        # Plot masked (grayed out) points first
        masked_df = df[df["masked"]]
        fig = px.scatter(masked_df, x="x", y="y", template="plotly_white")
        fig.update_traces(
            marker=dict(color="lightgray", size=5, opacity=0.3),
            showlegend=False,
            hoverinfo="skip",
            selectedpoints=None,
        )
        current_trace += 1  # Increment for masked points trace

        # Plot unmasked points
        unmasked_df = df[~df["masked"]].copy()
        unmasked_df["original_index"] = unmasked_df.index
        cluster_traces = px.scatter(
            unmasked_df,
            x="x",
            y="y",
            color="cluster",
            color_discrete_sequence=colors,
            custom_data=custom_data_keys,
            hover_data={
                "x": False,
                "y": False,
                "cluster": True,
                "id": True,
                "point_index": True,
                "raw_data": True,
            },
            # hover_name="cluster",
        ).data

        # Update selection properties and track traces
        for trace in cluster_traces:
            trace.selected = dict(marker=dict(color="red", size=5))
            trace.unselected = dict(marker=dict(opacity=0.3, size=5, color="lightgray"))
            trace.marker.size = 5
            # Map cluster name to trace index
            trace_mapping[trace.name] = current_trace
            current_trace += 1

        fig.add_traces(cluster_traces)
    else:
        # Original plotting logic for no mask
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            color_discrete_sequence=colors,
            template="plotly_white",
            custom_data=custom_data,
            hover_data={
                "x": False,
                "y": False,
                "cluster": True,
                "id": True,
                "point_index": True,
                "raw_data": True,
            },
            # hover_name="cluster",
        )

        # Update selection properties and track traces
        for trace in fig.data:
            trace.selected = dict(marker=dict(color="red", size=5))
            trace.unselected = dict(marker=dict(opacity=0.3, size=5, color="lightgray"))
            trace.marker.size = 5
            # Map cluster name to trace index
            trace_mapping[trace.name] = current_trace
            current_trace += 1

    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        showlegend=True,
        coloraxis_colorbar=dict(
            tickmode="array",
            ticktext=[str(i) for i in sorted(df["cluster"].unique())],
            tickvals=list(range(len(df["cluster"].unique()))),
        ),
        dragmode="select",
        clickmode="event+select",
        selectionrevision=True,
    )

    # Add centroids
    centroid_x = []
    centroid_y = []
    centroid_cluster = []
    centroid_names = []
    centroid_count = []
    total = len(cluster_labels)

    cluster_to_count = dict(zip(*np.unique(cluster_labels, return_counts=True)))
    for cluster, count in cluster_to_count.items():
        mask = cluster_labels == cluster
        if (
            mask.any() and cluster != -1
        ):  # Only add centroid if cluster has points and is not noise
            centroid = reduced_embeddings[mask].mean(axis=0)
            centroid_x.append(centroid[0])
            centroid_y.append(centroid[1])
            centroid_names.append(f"Centroid {cluster}")
            centroid_cluster.append(str(cluster))
            centroid_count.append(count)

    if centroid_x:
        centroid_df = pd.DataFrame(
            {
                "x": centroid_x,
                "y": centroid_y,
                "cluster": centroid_cluster,
                "name": centroid_names,
                "count": centroid_count,
            }
        )
        centroid_traces = px.scatter(
            centroid_df,
            x="x",
            y="y",
            hover_data={"name": True, "x": False, "y": False, "count": True},
            color="cluster",
            color_discrete_sequence=colors,
            # symbol_sequence=["triangle-up"],
            symbol_sequence=["circle"],
            size=[x for x in centroid_count],
        ).data

        # Update centroid traces
        centroid_trace_indices = []  # Create list to store centroid trace indices
        for trace in centroid_traces:
            cluster_num = trace.name
            if cluster_num in df["cluster"].unique():
                cluster_color = [
                    t.marker.color for t in fig.data if t.name == cluster_num
                ][0]
                trace.marker.color = cluster_color
                trace.marker.line = dict(color="black", width=2)
                trace.showlegend = False
                trace.opacity = 0.5
                trace.selected = dict(marker=dict(color="red"))
                trace.unselected = dict(marker=dict(opacity=0.15))
                centroid_trace_indices.append(current_trace)  # Store the trace index
                current_trace += 1

        fig.add_traces(centroid_traces)
        trace_mapping["centroids"] = centroid_trace_indices  # Store list of indices

    return fig, df, trace_mapping
