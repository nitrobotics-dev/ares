from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ares.app.data_analysis import infer_visualization_type
from ares.models.shortcuts import summarize
from ares.utils.clustering import visualize_clusters


def create_structured_data_filters(
    df: pd.DataFrame, max_options: int = 25, n_cols: int = 3, debug: bool = False
) -> tuple[pd.DataFrame, list]:
    """Create filter controls for dataframe columns based on their types."""
    filtered_df = df.copy()
    skipped_cols = []

    if debug:
        # Debug prints for initial state
        print("Initial df shape:", df.shape)
        print("Initial temp filters:", st.session_state.get("temp_filter_values", {}))
        print(
            "Initial active filters:", st.session_state.get("active_filter_values", {})
        )

    # Add debug print to see actual values in the DataFrame
    if debug:
        print("\nActual unique values in key columns:")
        for col in ["robot_embodiment", "robot_rgb_cams"]:
            print(f"{col} unique values:", filtered_df[col].unique())

    # Initialize temporary filter values if not exists
    if "temp_filter_values" not in st.session_state:
        st.session_state.temp_filter_values = {}
    if "active_filter_values" not in st.session_state:
        st.session_state.active_filter_values = {}

    filter_cols = st.columns(n_cols)

    for idx, col in enumerate(df.columns):
        viz_info = infer_visualization_type(col, df)
        if viz_info["viz_type"] is None:
            skipped_cols.append(col)
            continue
        with filter_cols[idx % n_cols]:
            if (
                pd.api.types.is_numeric_dtype(df[col])
                and viz_info["nunique"] > max_options
            ):
                min_val = float(df[col].dropna().min())
                max_val = float(df[col].dropna().max())

                # Add checkbox for including None values
                if f"{col}_include_none" not in st.session_state.temp_filter_values:
                    st.session_state.temp_filter_values[f"{col}_include_none"] = (
                        st.session_state.active_filter_values.get(
                            f"{col}_include_none", True
                        )
                    )

                include_none = st.checkbox(
                    f"Include None values for {col}",
                    value=st.session_state.temp_filter_values[f"{col}_include_none"],
                    key=f"{col}_none_checkbox",
                )
                st.session_state.temp_filter_values[f"{col}_include_none"] = (
                    include_none
                )

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

                # Debug prints for numeric filters
                if debug:
                    print(f"\nNumeric filter for {col}:")
                    print(
                        f"Temp value: {st.session_state.temp_filter_values.get(f'{col}_range')}"
                    )
                    print(
                        f"Active value: {st.session_state.active_filter_values.get(f'{col}_range')}"
                    )
                    print(
                        f"Include None - Temp: {st.session_state.temp_filter_values.get(f'{col}_include_none')}"
                    )
                    print(
                        f"Include None - Active: {st.session_state.active_filter_values.get(f'{col}_include_none')}"
                    )

                # Only apply active filters
                active_values = st.session_state.active_filter_values.get(
                    f"{col}_range"
                )
                active_include_none = st.session_state.active_filter_values.get(
                    f"{col}_include_none"
                )

                if active_values:
                    old_shape = filtered_df.shape[0]
                    mask = (
                        (filtered_df[col] >= active_values[0])
                        & (filtered_df[col] <= active_values[1])
                    ) | (filtered_df[col].isna() if active_include_none else False)
                    filtered_df = filtered_df[mask]
                    if debug:
                        print(
                            f"After filtering {col}: {old_shape} -> {filtered_df.shape[0]} rows"
                        )

            elif viz_info["viz_type"] == "bar":
                # Add debug print before filtering
                if col in ["robot_rgb_cams"]:
                    if debug:
                        print(f"\nDEBUG {col} before filtering:")
                        print(f"Unique values in df: {filtered_df[col].unique()}")
                        print(f"Type of values in df: {filtered_df[col].dtype}")

                # Convert None to "(None)" for display, but keep other values as is
                options = [
                    str(x) if x is not None else "(None)" for x in df[col].unique()
                ]
                options = sorted(options)

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

                # Debug prints for categorical filters
                if debug:
                    print(f"\nCategory filter for {col}:")
                    print(
                        f"Temp value: {st.session_state.temp_filter_values.get(f'{col}_select')}"
                    )
                    print(
                        f"Active value: {st.session_state.active_filter_values.get(f'{col}_select')}"
                    )

                # Only apply active filters
                active_selected = st.session_state.active_filter_values.get(
                    f"{col}_select"
                )
                if active_selected:
                    old_shape = filtered_df.shape[0]
                    # Convert values based on column type
                    if pd.api.types.is_bool_dtype(df[col]):
                        active_values = [
                            x == "True" if x != "(None)" else None
                            for x in active_selected
                        ]
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        active_values = [
                            float(x) if x != "(None)" else None for x in active_selected
                        ]
                        # Convert to int if the original column was integer
                        if pd.api.types.is_integer_dtype(df[col]):
                            active_values = [
                                int(x) if x is not None else None for x in active_values
                            ]
                    else:
                        active_values = [
                            (None if x == "(None)" else x) for x in active_selected
                        ]

                    # Debug prints for problematic filters
                    if debug:
                        print(f"\nFiltering {col}:")
                        print(f"Column dtype: {df[col].dtype}")
                        print(f"Active values: {active_values}")
                        print(
                            f"Types of active values: {[type(x) for x in active_values]}"
                        )

                    filtered_df = filtered_df[filtered_df[col].isin(active_values)]
                    if debug:
                        print(
                            f"After filtering {col}: {old_shape} -> {filtered_df.shape[0]} rows"
                        )

    if debug:
        print("\nFinal df shape:", filtered_df.shape)
    # check if len(filtered_df) == 0
    if len(filtered_df) == 0:
        breakpoint()
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


def _handle_selection(
    df: pd.DataFrame, selected_value: str | int | None, selector_fn: Callable
) -> pd.Series:
    """Helper function to handle row selection with default fallback.

    Args:
        df: DataFrame to select from
        selected_value: The selected value from the dropdown
        selector_fn: Function that returns the row given the selected value
    """
    if selected_value == "Choose an option" or selected_value is None:
        st.warning("No selection made, defaulting to first row")
        return df.iloc[0]
    return selector_fn(selected_value)


def select_row_from_df_user(df: pd.DataFrame) -> None:
    """Select a row from DataFrame with user input.

    Args:
        df: DataFrame to select from

    Returns:
        pd.Series: selected row
    """
    col1, col2, col3, col4 = st.columns(4)
    row = None

    # Option 1: Select by index
    with col1:
        idx_options = list(range(len(df)))
        idx: int | str = st.selectbox(
            "Select by position",
            options=["Choose an option"] + sorted(idx_options),
            key="idx_select",
        )
        if st.button("Select by Position"):
            if idx != "Choose an option":
                st.session_state["selected_row"] = df.iloc[int(idx)]

    # Option 2: Select by ID
    with col2:
        id_options = df.id.apply(str).tolist()
        selected_id: str | None = st.selectbox(
            "Select by ID",
            options=["Choose an option"] + sorted(id_options),
            key="id_select",
        )
        if st.button("Select by ID"):
            if selected_id != "Choose an option":
                mask = df.id.apply(str) == selected_id
                st.session_state["selected_row"] = df[mask].iloc[0]

    # Option 3: Select by Dataset Name + Path
    with col3:
        path_options = pd.Series(
            [f"{name}/{path}" for name, path in zip(df.dataset_name, df.path)]
        )
        selected_path: str | None = st.selectbox(
            "Select by Path",
            options=["Choose an option"] + path_options.tolist(),
            key="path_select",
        )
        if st.button("Select by Path"):
            if selected_path != "Choose an option":
                st.session_state["selected_row"] = df[
                    path_options == selected_path
                ].iloc[0]

    with col4:
        if st.button("Select Random"):
            random_idx = np.random.randint(len(df))
            st.session_state["selected_row"] = df.iloc[random_idx]

    if st.session_state.get("selected_row") is None:
        st.warning("No row selected, defaulting to first row")
        st.session_state["selected_row"] = df.iloc[0]


def handle_selection(
    value_filtered_df: pd.DataFrame,
    selection: dict,
    cluster_to_trace: dict[str, int],
    fig: go.Figure,
    custom_data_keys: list[str],
) -> tuple[bool, list[int], dict]:
    # determine if selection was made OR if just no points selected
    if selection_flag := any(
        len(selection.get(k, [])) > 0 for k in ["box", "lasso", "points"]
    ):
        # selection made!
        selected_ids = []
        valid_traces = [
            v for k, v in cluster_to_trace.items() if "centroid" not in k.lower()
        ]
        id_idx = custom_data_keys.index("id")
        for point in selection["points"]:
            if point["curve_number"] in valid_traces:
                point_number = point["point_number"]
                trace_data = fig.data[point["curve_number"]]
                original_idx = trace_data.customdata[point_number][id_idx]
                selected_ids.append(original_idx)

    else:
        # no selection made, use all filtered points
        selected_ids = value_filtered_df.id.tolist()
    return selection_flag, selected_ids, selection


def create_embedding_data_filters(
    value_filtered_df: pd.DataFrame,
    fig: go.Figure,
    cluster_to_trace: dict[str, int],
    custom_data_keys: list[str],
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
    selection_flag, selected_ids, selection = handle_selection(
        value_filtered_df, selection, cluster_to_trace, fig, custom_data_keys
    )
    return selection_flag, selected_ids, selection


def embedding_data_filters_display(
    df: pd.DataFrame,
    reduced: np.ndarray,
    labels: np.ndarray,
    raw_data_key: str,
    id_key: str = "id",
    keep_mask: list[str] | None = None,
) -> tuple[pd.DataFrame, go.Figure, dict, bool]:
    st.subheader(f"Embedding Filters")
    custom_data_keys = ["raw_data", "id"]
    if keep_mask is None:
        keep_mask = df[id_key].apply(str).tolist()
    with st.expander("Embedding Selection", expanded=False):
        cluster_fig, cluster_df, cluster_to_trace = visualize_clusters(
            reduced,
            labels,
            raw_data=df[raw_data_key].tolist(),
            ids=df[id_key].apply(str).tolist(),
            custom_data_keys=custom_data_keys,
            keep_mask=keep_mask,
        )
        selection_flag, selected_ids, selection = create_embedding_data_filters(
            df, cluster_fig, cluster_to_trace, custom_data_keys
        )

        # need to only include IDs in selection that are in KEPT
        selected_ids = [x for x in selected_ids if str(x) in keep_mask]

        st.write("**Selection Controls**")
        st.write("Double click to clear selection")
        if st.button("Summarize Selection"):
            raw_data_idx = custom_data_keys.index("raw_data")
            cluster_to_data = defaultdict(list)
            for p in selection["points"]:
                if p["curve_number"] not in cluster_to_trace.get("centroids", []):
                    cluster_to_data[p["curve_number"]].append(
                        p["customdata"][raw_data_idx]
                    )
            st.write(
                f"Selected {sum(len(v) for v in cluster_to_data.values())} points in {len(cluster_to_data)} clusters"
            )
            for k, v in cluster_to_data.items():
                v_sample = np.random.choice(v, min(10, len(v)), replace=False)
                st.write(f"**Cluster {k}**")
                st.write(f"Ex: {'; '.join(v_sample[:5])}")
                st.write(
                    f"Summary: {summarize(st.session_state['models']['summarizer'], v_sample, description='a cluster of points describing robot tasks.')}"
                )

    filtered_df = df[df.id.astype(str).isin(map(str, selected_ids))]

    st.write(
        f"Selection found! Using '{'box' if selection['box'] else 'lasso' if selection['lasso'] else 'points'}' as bounds"
        if selection_flag
        else "No selection found, using all points"
    )
    st.write(
        f"Selected {len(filtered_df)} rows out of {len(keep_mask)} available via embedding filters"
    )
    return filtered_df, cluster_fig, selection, selection_flag
