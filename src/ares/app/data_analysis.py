"""
Simple helpers for inferring data types and generating visualizations. For example, a column of data containing a small number of unique strings can 
be inferred to be a bar chart, while a column of data containing a large number of unique floats can be inferred to be a histogram. We also choose
to *not* display certain columns that have poor visualizations, such as a column of data containing a large number of unique strings such as IDs or file paths.
"""

import typing as t

import pandas as pd

from ares.app.plot_primitives import create_bar_plot, create_histogram
from ares.constants import IGNORE_COLS


def infer_visualization_type(
    column_name: str,
    data: pd.DataFrame,
    ignore_cols: list | None = None,
    max_str_length: int = 500,
) -> dict[str, t.Any]:
    """
    Heuristic solution for transforming a column of data into a visualization type,
    focusing on numeric ranges or category counts.
    """
    ignore_cols = ignore_cols or IGNORE_COLS

    dtype = str(data[column_name].dtype)
    nunique = data[column_name].nunique()

    result = {"viz_type": None, "dtype": dtype, "nunique": nunique}

    if column_name.lower() in ignore_cols:
        return result

    # Add special handling for boolean columns
    if pd.api.types.is_bool_dtype(data[column_name]):
        result["viz_type"] = "bar"
        return result

    if pd.api.types.is_string_dtype(data[column_name]):
        if data[column_name].str.len().max() > max_str_length:
            return result

    if pd.api.types.is_datetime64_any_dtype(data[column_name]):
        return result

    if pd.api.types.is_numeric_dtype(data[column_name]) or (
        dtype == "object"
        and len(data[column_name].dropna()) > 0
        and pd.to_numeric(data[column_name].dropna(), errors="coerce").notna().all()
    ):
        # check if lots of unique values or if it's a float between 0 and 1
        if nunique > 20 or (
            pd.api.types.is_float_dtype(data[column_name])
            and data[column_name].min() >= 0
            and data[column_name].max() <= 1
        ):
            result["viz_type"] = "histogram"
        else:
            result["viz_type"] = "bar"
        return result

    if pd.api.types.is_string_dtype(data[column_name]) or nunique < 20:
        result["viz_type"] = "bar"
        return result

    return result


def generate_automatic_visualizations(
    df: pd.DataFrame,
    time_column: str = "creation_time",
    ignore_cols: list[str] | None = None,
    max_x_bar_options: int = 100,
) -> list[dict]:
    """
    After inferring the 'type' of a column, we can create automatic visualizations.
    """
    ignore_cols = ignore_cols or IGNORE_COLS
    visualizations = []

    # Pre-calculate visualization types for all columns at once
    viz_infos = {
        col: infer_visualization_type(col, df)
        for col in sorted(df.columns)
        if col != time_column and col.lower() not in ignore_cols
    }

    # Group columns by visualization type
    histogram_cols = []
    bar_cols = []
    for col, info in viz_infos.items():
        if not info["nunique"] or (
            info["viz_type"] == "bar" and info["nunique"] > max_x_bar_options
        ):
            continue
        if info["viz_type"] == "histogram":
            histogram_cols.append(col)
        elif info["viz_type"] == "bar":
            bar_cols.append(col)

    # Create histogram visualizations
    for col in histogram_cols:
        col_title = col.replace("_", " ").replace("-", " ").title()
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

    # Create bar visualizations - handle each column separately
    for col in bar_cols:
        col_title = col.replace("_", " ").replace("-", " ").title()

        # Special handling for boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            # Convert to string before aggregation to preserve True/False
            agg_data = (
                df[col]
                .astype(str)
                .groupby(df[col])
                .agg(count=(time_column, "count"))
                .reset_index()
            )
            agg_data.columns = [
                col,
                time_column,
            ]  # Rename columns to match expected format
        else:
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
