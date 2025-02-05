"""
General helpers for building filters for our data. Inferring data types can be messy, so see `ares.app.data_analysis.infer_visualization_type` for tricks.

For structured data, We focus on categorical data with low cardinality and ranges of numberical data.
For unstructured (embedding) data, we focus on tools to explore and select ranges of the reduced embedding space, 
including a `summarize` tool to quickly understand a cluster.

Note: we maintain sets of `temp` and `active` filter states to try and avoid re-rendering streamlit as much as possible.
Note: there are lots of debug logs throughout; these filters and data types can be very finicky due to data storage conversions. Always look at your data!
"""

import traceback
import typing as t
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ares.app.data_analysis import infer_visualization_type
from ares.constants import IGNORE_COLS
from ares.models.shortcuts import summarize
from ares.utils.clustering import visualize_clusters


def numberic_col_data_filter(
    df: pd.DataFrame, filtered_df: pd.DataFrame, col: str, debug: bool
) -> pd.DataFrame:
    """
    Helper function for generating data filters based on numeric categories. Due to storage, data types may need to be coerced to numeric types.
    Numeric types create `slider` type filters such that the user can select a range using the min and max range desired.
    Some numeric columns have nan values, so we include a checkbox as to whether or not to pretend nan values are "in range" or not.
    """
    numeric_col = pd.to_numeric(df[col], errors="coerce")
    min_val = float(numeric_col.dropna().min())
    max_val = float(numeric_col.dropna().max())

    # Checkbox for including None values
    checkbox_key = f"{col}_include_none"
    if checkbox_key not in st.session_state.temp_filter_values:
        st.session_state.temp_filter_values[checkbox_key] = True

    include_none = st.checkbox(
        f"Include None values for {col}",
        value=st.session_state.temp_filter_values[checkbox_key],
        key=f"{col}_none_checkbox",
    )
    st.session_state.temp_filter_values[checkbox_key] = include_none

    # Slider for numeric range
    range_key = f"{col}_range"
    if range_key not in st.session_state.temp_filter_values:
        st.session_state.temp_filter_values[range_key] = (min_val, max_val)

    slider_key = f"slider_{col}"
    values = st.slider(
        f"Filter {col}",
        min_value=min_val,
        max_value=max_val,
        value=st.session_state.temp_filter_values[range_key],
        key=slider_key,
    )
    st.session_state.temp_filter_values[range_key] = values

    # Debug prints for numeric filters
    if debug:
        print(f"\nNumeric filter for {col}:")
        print(f"Temp value: {st.session_state.temp_filter_values.get(range_key)}")
        print(
            f"Include None - Temp: {st.session_state.temp_filter_values.get(checkbox_key)}"
        )

    # Apply temp filters
    active_values = st.session_state.temp_filter_values.get(range_key)
    active_include_none = st.session_state.temp_filter_values.get(checkbox_key)

    # if there are active values, we want to apply those filters to our dataframe
    if active_values:
        old_shape = filtered_df.shape[0]
        numeric_col = pd.to_numeric(filtered_df[col], errors="coerce")
        mask = (numeric_col >= active_values[0]) & (numeric_col <= active_values[1])
        if active_include_none:
            mask |= filtered_df[col].isna()
        filtered_df = filtered_df[mask]
        if debug:
            print(f"After filtering {col}: {old_shape} -> {filtered_df.shape[0]} rows")
    return filtered_df


def categorical_col_data_filter(
    df: pd.DataFrame, filtered_df: pd.DataFrame, col: str, debug: bool, max_options: int
) -> tuple[pd.DataFrame, t.Optional[str]]:
    """
    Helper function for categorical data filters. Streamlit handles some of the multiselect logic, so we just enforce some types
    and limits for visual clarity.
    """
    # try to sort options for visual clarity, may have to convert to numbers if possible!
    if pd.api.types.is_numeric_dtype(df[col]):
        options = sorted(df[col].unique())
    else:
        options = sorted(
            [str(x) if x is not None else "(None)" for x in df[col].unique()]
        )

    # the dashboard gets quickly filled if too many options, so limit cardinality of options
    if len(options) > max_options:
        return filtered_df, col

    select_key = f"{col}_select"
    multiselect_key = f"multiselect_{col}"

    if select_key not in st.session_state.temp_filter_values:
        st.session_state.temp_filter_values[select_key] = options.copy()

    selected = st.multiselect(
        f"Filter {col}",
        options=options,
        default=st.session_state.temp_filter_values[select_key],
        key=multiselect_key,
    )

    st.session_state.temp_filter_values[select_key] = selected
    active_selected = st.session_state.temp_filter_values.get(select_key)

    # in the event of active filters selected, apply those to the dataframe
    if active_selected:
        old_shape = filtered_df.shape[0]
        # Convert selected options back to original data type if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Ensure all selections are numeric
            try:
                active_selected_converted = [
                    (float(x) if isinstance(x, float) or "." in str(x) else int(x))
                    for x in active_selected
                ]
            except ValueError as ve:
                if debug:
                    print(f"ValueError converting selections for {col}: {ve}")
                active_selected_converted = []
        else:
            # Handle None selection
            active_selected_converted = [
                None if x == "(None)" else x for x in active_selected
            ]

        # Apply the filter only if conversion was successful
        if active_selected_converted:
            filtered_df = filtered_df[filtered_df[col].isin(active_selected_converted)]
            if debug:
                print(
                    f"After filtering {col}: {old_shape} -> {filtered_df.shape[0]} rows"
                )
    return filtered_df, None


def numeric_coercable_or_float_range(
    df: pd.DataFrame, col: str, viz_info: dict[str, t.Any], max_options: int
) -> bool:
    """
    Helper function for coercion of data into a numeric type in a float range of [0,1] but with low cardinality.
    Useful for e.g. `success_estimate`, where values are typically increments of 0.1; we would like to display these as a range
    datatype not a categorical datatype despite low cardinality.
    """
    return (
        pd.api.types.is_numeric_dtype(df[col])
        or (
            str(df[col].dtype) == "object"
            and len(df[col].dropna()) > 0
            and pd.to_numeric(df[col].dropna(), errors="coerce").notna().all()
        )
    ) and (
        viz_info["nunique"] > max_options
        or (
            pd.api.types.is_float_dtype(df[col])
            and df[col].min() >= 0
            and df[col].max() <= 1
            and df[col].min() != df[col].max()
        )
    )


def create_structured_data_filters(
    df: pd.DataFrame,
    max_options: int = 25,
    n_cols: int = 3,
    ignore_cols: list[str] | None = None,
    debug: bool = False,
) -> tuple[pd.DataFrame, list]:
    """
    Function to apply filters over all the available dataframe columns, excluding some simple ones.
    """
    ignore_cols = ignore_cols or IGNORE_COLS
    filtered_df = df.copy()
    skipped_cols = []

    if debug:
        # Debug prints for initial state
        print("Initial df shape:", df.shape)
        print("Initial temp filters:", st.session_state.get("temp_filter_values", {}))

    # Initialize temporary filter values if not exists
    if "temp_filter_values" not in st.session_state:
        st.session_state.temp_filter_values = {}

    filter_cols = st.columns(n_cols)

    for idx, col in enumerate(df.columns):
        viz_info = infer_visualization_type(col, df)

        if viz_info["viz_type"] is None or col.lower() in ignore_cols:
            skipped_cols.append(col)
            continue

        # filtered_df = df
        with filter_cols[idx % n_cols]:
            if numeric_coercable_or_float_range(df, col, viz_info, max_options):
                filtered_df = numberic_col_data_filter(df, filtered_df, col, debug)

            elif viz_info["viz_type"] == "bar":
                filtered_df, maybe_skipped_col = categorical_col_data_filter(
                    df, filtered_df, col, debug, max_options
                )
                if maybe_skipped_col:
                    skipped_cols.append(maybe_skipped_col)
            else:
                skipped_cols.append(col)

    if debug:
        print("\nFinal df shape:", filtered_df.shape)

    # Check if len(filtered_df) == 0! something wrong
    if len(filtered_df) == 0:
        breakpoint()
    return filtered_df, skipped_cols


def structured_data_filters_display(
    df: pd.DataFrame, debug: bool = False
) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    """
    Create the streamlit wrappers for the structured data filters display, wrapping everything in a streamlit `form` in order to
    avoid re-rendering until form submission. Also add a `reset` button to return all temp and active values to the original defaults.
    """
    col1, col2, _, _ = st.columns([5, 1, 1, 3])
    with col1:
        st.subheader("Structured Data Filters")

    # Reset Button outside the form to allow resetting anytime
    with col2:
        if st.button("Reset Filters", type="primary"):
            # Clear all filter states
            st.session_state.temp_filter_values = {}
            st.session_state.active_filter_values = {}
            st.rerun()

    st.write(
        "Apply changes by clicking the 'Apply Filters' button at the bottom of the form."
    )
    st.write(
        "Reset all filters by clicking the 'Reset Filters' button at the top of the form."
    )

    with st.expander("Filter Data", expanded=False):
        # really nice streamlit element to avoid re-render when messing with the filters
        with st.form(key="filter_form"):
            value_filtered_df, skipped_cols = create_structured_data_filters(
                df, debug=debug
            )
            submit = st.form_submit_button("Apply Filters")
            if submit:
                # Copy temporary values to active values
                st.session_state.active_filter_values = (
                    st.session_state.temp_filter_values.copy()
                )
                st.rerun()

        if skipped_cols:
            st.warning(
                f"Skipped columns: {skipped_cols} due to high cardinality or lack of unique values"
            )

    active_filters = (
        dict()
        if "active_filter_values" not in st.session_state
        else st.session_state.active_filter_values
    )
    return value_filtered_df, active_filters


def select_row_from_df_user(df: pd.DataFrame) -> None:
    """
    Select a row from DataFrame with user input which is then stored in state for reuse.
    Options: select by index, select by id, select by path, or select by random.
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
        # Create path_options with the same index as df
        path_options = pd.Series(
            [f"{name}/{path}" for name, path in zip(df.dataset_filename, df.path)],
            index=df.index,
        )
        selected_path: str | None = st.selectbox(
            "Select by Path",
            options=["Choose an option"] + sorted(path_options.tolist()),
            key="path_select",
        )
        if st.button("Select by Path"):
            if selected_path != "Choose an option":
                # Use boolean indexing with aligned indices
                st.session_state["selected_row"] = df[
                    path_options == selected_path
                ].iloc[0]

    with col4:
        if st.button("Select Random (I'm Feeling Lucky)"):
            random_idx = np.random.randint(len(df))
            st.session_state["selected_row"] = df.iloc[random_idx]

    selected_row = st.session_state.get("selected_row")
    if selected_row is None or selected_row.id not in df.id.values:
        st.warning("No row selected, defaulting to first row")
        st.session_state["selected_row"] = df.iloc[0]


def handle_selection(
    value_filtered_df: pd.DataFrame,
    selection: dict,
    cluster_to_trace: dict[str, int],
    fig: go.Figure,
    custom_data_keys: list[str],
) -> tuple[bool, list[int], dict]:
    """
    Interactive helper for embedding data visualization.
    Utilizes the selection element to find which points are selected then map those to clusters.
    """
    # Determine if selection was made OR if just no points selected
    if selection_flag := any(
        len(selection.get(k, [])) > 0 for k in ["box", "lasso", "points"]
    ):
        # Selection made!
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
        # No selection made, use all filtered points
        selected_ids = value_filtered_df.id.tolist()
    return selection_flag, selected_ids, selection


def create_embedding_data_filters(
    value_filtered_df: pd.DataFrame,
    fig: go.Figure,
    cluster_to_trace: dict[str, int],
    custom_data_keys: list[str],
    name: str,
) -> tuple[bool, list[int], dict]:
    """
    Embedding filters to explore or select regions of space in the reduced embedding space.
    We create a 2D plot by reducing embeddings and handle clustering and selection.
    """
    # Display the plot and capture selected points from the event data
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"cluster_plot_{name}",
        selection_mode=["points", "box", "lasso"],
        on_select="rerun",
    )
    selection = getattr(event, "selection", dict())
    # Handle finding the indices of the selected points
    selection_flag, selected_ids, selection = handle_selection(
        value_filtered_df, selection, cluster_to_trace, fig, custom_data_keys
    )
    return selection_flag, selected_ids, selection


def summarize_selection(
    selection: t.Any, cluster_to_trace: dict[str, int], custom_data_keys: list[str]
) -> str:
    """
    Helpful tool to understand clusters. Given a selection users can send a sample of a selection to an LLM and request a summary.
    """
    raw_data_idx = custom_data_keys.index("raw_data")
    cluster_to_data = defaultdict(list)
    for p in selection["points"]:
        if p["curve_number"] not in cluster_to_trace.get("centroids", []):
            if "customdata" not in p:
                continue
            cluster_to_data[p["curve_number"]].append(p["customdata"][raw_data_idx])
    st.write(
        f"Selected {sum(len(v) for v in cluster_to_data.values())} points in {len(cluster_to_data)} clusters"
    )
    for k, v in cluster_to_data.items():
        v_sample = np.random.choice(v, min(10, len(v)), replace=False)
        st.write(f"**Cluster {k}**")
        auto_summary = summarize(
            st.session_state["models"]["summarizer"],
            v_sample,
            description="a subset of a cluster of points describing robot tasks.",
        )
        st.write(f"**Summary**: {auto_summary}")
        st.write(f"**Examples**: " + "\n- ".join(v_sample[:5]))


def embedding_data_filters_display(
    df: pd.DataFrame,
    reduced: np.ndarray,
    labels: np.ndarray,
    raw_data_key: str,
    id_key: str = "id",
    keep_mask: list[str] | None = None,
) -> tuple[pd.DataFrame, go.Figure, dict, bool]:
    """
    Wrapper for displaying the actual embedding filters and handling interaction with the filtered_df.
    The keep_mask determines which indices should actually be grayed-out due to deselection from the structured data filters.
    """
    custom_data_keys = ["raw_data", "id"]
    if keep_mask is None:
        keep_mask = df[id_key].apply(str).tolist()

    # for visual clarity, hide the embedding selection until the user requests it.
    with st.expander(
        f"Embedding Selection ({raw_data_key.replace('_', ' ').title()})",
        expanded=False,
    ):
        cluster_fig, cluster_df, cluster_to_trace = visualize_clusters(
            reduced,
            labels,
            raw_data=df[raw_data_key].tolist(),
            ids=df[id_key].apply(str).tolist(),
            custom_data_keys=custom_data_keys,
            keep_mask=keep_mask,
        )
        selection_flag, selected_ids, selection = create_embedding_data_filters(
            df, cluster_fig, cluster_to_trace, custom_data_keys, name=raw_data_key
        )

        # Need to only include IDs in selection that are in KEPT
        selected_ids = [x for x in selected_ids if str(x) in keep_mask]

        st.write("**Selection Controls**")
        st.write("Double click to clear selection")
        if st.button("Summarize Selection", key=f"summarize_selection_{raw_data_key}"):
            summarize_selection(selection, cluster_to_trace, custom_data_keys)

    filtered_df = df[df.id.astype(str).isin(map(str, selected_ids))]
    st.write(
        f"Selection found! Using '{'box' if selection.get('box') else 'lasso' if selection.get('lasso') else 'points'}' as bounds"
        if selection_flag
        else "No selection found, using all points"
    )
    st.write(
        f"Selected {len(filtered_df)} rows out of {len(keep_mask)} available via embedding filters"
    )
    return filtered_df, cluster_fig, selection, selection_flag


def create_embedding_data_filter_display(
    df: pd.DataFrame,
    id_key: str,
    raw_data_key: str,
    kept_ids: list[str],
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Locates values from state and attempts to construct the embedding data filter displays
    """
    reduced_embeddings = st.session_state[f"{raw_data_key}_reduced"]
    labels = st.session_state[f"{raw_data_key}_labels"]
    state_ids = st.session_state[f"{raw_data_key}_ids"]
    try:
        desired_ids = df[id_key].apply(str)
        id_to_idx_state = {_id: idx for idx, _id in enumerate(state_ids)}
        desired_state_indices = [id_to_idx_state[_id] for _id in desired_ids]
        reduced_embeddings = reduced_embeddings[desired_state_indices]
        labels = labels[desired_state_indices]

        filtered_df, cluster_fig, selection, selection_flag = (
            embedding_data_filters_display(
                df=df,
                reduced=reduced_embeddings,
                labels=labels,
                raw_data_key=raw_data_key,
                id_key=id_key,
                keep_mask=kept_ids,
            )
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        breakpoint()
    return filtered_df, cluster_fig
