import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ares.app.data_analysis import infer_visualization_type
from ares.clustering import visualize_clusters

# TODO: use form filter button! https://discuss.streamlit.io/t/faq-how-to-prevent-app-reruns/63916


def create_structured_data_filters(
    df: pd.DataFrame, max_options: int = 9
) -> tuple[pd.DataFrame, list]:
    """Create filter controls for dataframe columns based on their types."""
    filtered_df = df.copy()
    skipped_cols = []

    # Initialize temporary filter values if not exists
    if "temp_filter_values" not in st.session_state:
        st.session_state.temp_filter_values = {}
    if "active_filter_values" not in st.session_state:
        st.session_state.active_filter_values = {}

    filter_cols = st.columns(3)

    for idx, col in enumerate(df.columns):
        viz_info = infer_visualization_type(col, df)
        if viz_info["viz_type"] is None:
            skipped_cols.append(col)
            continue
        with filter_cols[idx % 3]:
            if (
                pd.api.types.is_numeric_dtype(df[col])
                and viz_info["nunique"] > max_options
            ):
                min_val = float(df[col].min())
                max_val = float(df[col].max())

                # Initialize with current active values or defaults
                if f"{col}_range" not in st.session_state.temp_filter_values:
                    st.session_state.temp_filter_values[f"{col}_range"] = (
                        st.session_state.active_filter_values.get(
                            f"{col}_range", (min_val, max_val)
                        )
                    )

                # Update temporary values without filtering
                values = st.slider(
                    f"Filter {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=st.session_state.temp_filter_values[f"{col}_range"],
                )
                st.session_state.temp_filter_values[f"{col}_range"] = values

                # Only apply active filters
                active_values = st.session_state.active_filter_values.get(
                    f"{col}_range"
                )
                if active_values:
                    filtered_df = filtered_df[
                        (filtered_df[col] >= active_values[0])
                        & (filtered_df[col] <= active_values[1])
                    ]

            elif viz_info["viz_type"] == "bar":
                options = sorted(df[col].unique().tolist())
                # bad ux for > 9 options
                if len(options) > max_options:
                    skipped_cols.append(col)
                    continue

                # Initialize with current active values or defaults
                if f"{col}_select" not in st.session_state.temp_filter_values:
                    st.session_state.temp_filter_values[f"{col}_select"] = (
                        st.session_state.active_filter_values.get(
                            f"{col}_select", options
                        )
                    )

                # Update temporary values without filtering
                selected = st.multiselect(
                    f"Filter {col}",
                    options=options,
                    default=st.session_state.temp_filter_values[f"{col}_select"],
                )
                st.session_state.temp_filter_values[f"{col}_select"] = selected

                # Only apply active filters
                active_selected = st.session_state.active_filter_values.get(
                    f"{col}_select"
                )
                if active_selected:
                    filtered_df = filtered_df[filtered_df[col].isin(active_selected)]

    return filtered_df, skipped_cols


def structured_data_filters_display(df: pd.DataFrame) -> pd.DataFrame:
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.subheader("Structured Data Filters")
    with col2:
        if st.button("Reset Filters", type="primary"):
            # Clear all filter states
            st.session_state.temp_filter_values = {}
            st.session_state.active_filter_values = {}
            st.rerun()
    with col3:
        if st.button("Apply Filters", type="primary"):
            # Copy temporary values to active values
            st.session_state.active_filter_values = (
                st.session_state.temp_filter_values.copy()
            )
            st.rerun()

    with st.expander("Filter Data", expanded=False):
        value_filtered_df, skipped_cols = create_structured_data_filters(df)
        if skipped_cols:
            st.warning(
                f"Skipped columns: {skipped_cols} due to high cardinality or lack of unique values"
            )
    return value_filtered_df


def handle_selection(
    value_filtered_df: pd.DataFrame,
    selection: dict,
    cluster_to_trace: dict,
    fig: go.Figure,
) -> tuple[bool, list[int], dict]:
    # determine if selection was made OR if just no points selected
    selection_flag = any(
        len(selection.get(k, [])) > 0 for k in ["box", "lasso", "points"]
    )
    if selection_flag:
        # selection made!
        indices = []
        valid_traces = [v for k, v in cluster_to_trace.items() if "cluster" in k]
        for point in selection["points"]:
            if point["curve_number"] in valid_traces:
                # Get the point's position within its trace
                point_number = point["point_number"]
                # Get the trace data
                trace_data = fig.data[point["curve_number"]]
                # Get the original index from custom_data
                original_idx = trace_data.customdata[point_number][0]
                indices.append(original_idx)

    else:
        # no selection made, use all filtered points
        indices = value_filtered_df.index.tolist()
    return selection_flag, indices, selection


def create_embedding_data_filters(
    value_filtered_df: pd.DataFrame, fig: go.Figure, cluster_to_trace: dict
) -> tuple[bool, list[int], dict]:
    # Display the plot and capture selected points from the event data
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="cluster_plot",
        selection_mode=["points", "box", "lasso"],
        on_select="rerun",
    )
    selection = getattr(event, "selection", dict())
    # handle finding the indices of the selected points
    selection_flag, indices, selection = handle_selection(
        value_filtered_df, selection, cluster_to_trace, fig
    )
    return selection_flag, indices, selection


def embedding_data_filters_display(
    df: pd.DataFrame,
    reduced: np.ndarray,
    labels: np.ndarray,
    raw_data_key: str,
    id_key: str = "id",
) -> tuple[pd.DataFrame, go.Figure, dict, bool]:
    st.subheader(f"Embedding Filters")
    with st.expander("Embedding Selection", expanded=False):
        cluster_fig, cluster_df, cluster_to_trace = visualize_clusters(
            reduced,
            labels,
            raw_data=df[raw_data_key].tolist(),
            ids=df[id_key].apply(str).tolist(),
            keep_mask=df.index.tolist(),  # TODO: change to ID
        )
        selection_flag, indices, selection = create_embedding_data_filters(
            df, cluster_fig, cluster_to_trace
        )
        st.write("**Selection Controls**")
        st.write("Double click to clear selection")
        if st.button("Summarize Selection (todo)"):
            st.write(f"selected {len(selection['points'])} points")
            st.write(f"ex: {selection['points'][:5]}")

    n_pts = len(indices)
    clusters = labels[indices] if n_pts > 0 else []
    n_clusters = len(np.unique(clusters))
    filtered_df = df.loc[indices]
    return filtered_df, cluster_fig, selection, selection_flag


def select_row_from_df_user(df: pd.DataFrame) -> pd.Series:
    # Create columns for row selection options
    col1, col2, col3, col4 = st.columns(4)
    row = None

    # Option 1: Select by index
    with col1:
        idx_options = df.index.tolist()
        idx: int | str = st.selectbox(
            "Select by index",
            options=["Choose an option"] + idx_options,
            key="idx_select",
        )
        if st.button("Select by Index"):
            if idx == "Choose an option":
                st.warning("No index selected, defaulting to first row")
                row = df.iloc[0]
            else:
                row = df.iloc[int(idx)]

    # Option 2: Select by ID
    with col2:
        id_options = df.id.unique().tolist()
        selected_id = int(
            st.selectbox("Select by ID", options=id_options, key="id_select")
        )
        if st.button("Select by ID"):
            row = df[df.id == selected_id].iloc[0]

    # Option 3: Select by Path
    with col3:
        path_options = df.path.unique().tolist()
        selected_path = st.selectbox(
            "Select by Path", options=path_options, key="path_select"
        )
        if st.button("Select by Path"):
            row = df[df.path == selected_path].iloc[0]

    with col4:
        if st.button("Select Random"):
            row = df.sample(1).iloc[0]

    if row is None:
        st.warning("No row selected, defaulting to first row")
        row = df.iloc[0]

    return row
