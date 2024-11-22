"""
Cluster embeddings using UMAP for dimensionality reduction and HDBSCAN for clustering.
"""

import json
import os
from typing import Optional, Tuple, Union

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap

# At the top with other imports
SELECTION_FILE = "/tmp/selected_points.json"


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
    keep_mask: Optional[list] = None,
) -> Tuple[go.Figure, pd.DataFrame, dict[str, int]]:
    """
    Create an interactive 2D visualization of the clustered embeddings.
    Returns figure and dataframe for selection tracking.
    """
    # Create a color map
    n_clusters = len(np.unique(cluster_labels))
    # Use Set1 palette but ensure discrete colors
    colors = px.colors.qualitative.Set1[:n_clusters]

    df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "cluster": [str(x) if x != -1 else "Noise" for x in cluster_labels],
            "point_index": range(len(cluster_labels)),
        }
    )

    # If mask is provided, create two separate dataframes
    if keep_mask is not None:
        mask_array = np.zeros(len(df), dtype=bool)
        mask_array[keep_mask] = True
        df["masked"] = ~mask_array

    # Initialize trace mapping at the start of visualization
    trace_mapping = {}
    current_trace = 0

    # Create figure with masked points first (if mask exists)
    if keep_mask is not None:
        # Plot masked (grayed out) points first
        masked_df = df[df["masked"]]
        fig = px.scatter(
            masked_df,
            x="x",
            y="y",
            # title=title,
            template="plotly_white",
        )
        fig.update_traces(
            marker=dict(color="lightgray", size=3, opacity=0.3),
            showlegend=False,
            hoverinfo="skip",
            selectedpoints=None,
        )
        trace_mapping["masked_points"] = current_trace
        current_trace += 1

        # Plot unmasked points on top, but track original indices
        unmasked_df = df[~df["masked"]].copy()
        unmasked_df["original_index"] = unmasked_df.index  # Store original indices
        cluster_traces = px.scatter(
            unmasked_df,
            x="x",
            y="y",
            color="cluster",
            color_discrete_sequence=colors,
            custom_data=["original_index"],  # Include original indices in hover data
        ).data
        fig.add_traces(cluster_traces)
        # Map each cluster trace
        for cluster in sorted(unmasked_df["cluster"].unique()):
            trace_mapping[f"cluster_{cluster}"] = current_trace
            current_trace += 1
    else:
        # Original plotting logic for no mask
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            title=title,
            color_discrete_sequence=colors,
            template="plotly_white",
        )
        # Map each cluster trace
        for cluster in sorted(df["cluster"].unique()):
            trace_mapping[f"cluster_{cluster}"] = current_trace
            current_trace += 1

    # Update traces and layout
    fig.update_traces(
        marker_size=3,
        selectedpoints=[],
        mode="markers",
        selected=dict(marker=dict(color="red")),
        selector=dict(type="scatter"),
    )

    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        showlegend=True,
        coloraxis_colorbar=dict(
            tickmode="array",
            ticktext=[str(i) for i in sorted(df["cluster"].unique())],
            tickvals=list(range(len(df["cluster"].unique()))),
            yanchor="top",
            y=1,
            x=1.2,
        ),
        # Update selection behavior
        dragmode="select",
        clickmode="event+select",
        selectionrevision=True,  # This helps persist selections
    )

    # Add centroids
    centroid_x = []
    centroid_y = []
    centroid_colors = []
    centroid_names = []

    for cluster in np.unique(cluster_labels):
        # if cluster != -1:  # Skip noise points
        mask = cluster_labels == cluster
        if mask.any():  # Only add centroid if cluster has points
            centroid = reduced_embeddings[mask].mean(axis=0)
            centroid_x.append(centroid[0])
            centroid_y.append(centroid[1])
            centroid_colors.append(colors[cluster])
            centroid_names.append(f"Centroid {cluster}")

    if centroid_x:
        centroid_df = pd.DataFrame(
            {"x": centroid_x, "y": centroid_y, "cluster": centroid_names}
        )
        fig.add_traces(
            px.scatter(
                centroid_df,
                x="x",
                y="y",
                color_discrete_sequence=["black"],
                symbol_sequence=["triangle-up"],
                size=[5] * len(centroid_df),
            ).data
        )
        trace_mapping["centroids"] = current_trace

    return fig, df, trace_mapping
