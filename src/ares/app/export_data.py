"""
Simple methods for exporting ARES dashboards for a report to external users.
"""

import os
import traceback
import typing as t
from datetime import datetime

import pandas as pd
import pdfkit
import plotly.graph_objects as go
import streamlit as st

from ares.constants import ARES_DATA_DIR


def export_dataframe(df: pd.DataFrame, base_path: str) -> str:
    """Export dataframe to CSV file with timestamp.

    Args:
        df: DataFrame containing all data
        base_path: Base directory path where file should be saved

    Returns:
        Path where file was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(base_path, "exports")

    # Create exports directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Save dataframe
    export_path = os.path.join(export_dir, f"video_analytics_{timestamp}.csv")
    df.to_csv(export_path, index=False)

    return export_path


def pdf_from_html(all_html_content: str, full_path: str) -> None:
    html_path = f"/tmp/export.html"
    with open(html_path, "w") as f:
        f.write(all_html_content)
    pdfkit.from_file(html_path, full_path)
    os.remove(html_path)  # Clean up temp file


def data_only_export(df: pd.DataFrame, export_path: str, format: str) -> str:
    # Simple data-only export
    full_path = f"{export_path}.{format}"
    print(f"Exporting data-only format to {full_path}")
    if format == "csv":
        df.to_csv(full_path, index=False)
    else:
        df.to_excel(full_path, index=False)
    return full_path


def pretty_dashboard_export(
    df: pd.DataFrame,
    export_path: str,
    title: str,
    structured_filters: dict[str, t.Any],
    visualizations: list[dict],
    format: str,
    go_figs: dict[str, go.Figure],
) -> str:
    # Full dashboard export
    full_path = f"{export_path}.{format}"
    img_dir = f"{export_path}_files"
    os.makedirs(img_dir, exist_ok=True)

    # Generate HTML content
    html_content = [
        "<html><head>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        ".plot-container { margin: 20px 0; }",
        ".stats-container { margin: 20px 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # add any selected structured filters
    if structured_filters:
        html_content.extend(
            [
                "<h2>Selected Filters</h2>",
                "<ul>",
                *[
                    f"<li><strong>{key.replace('_select', '').replace('_', ' ').title()}:</strong> {', '.join(map(str, values))}</li>"
                    for key, values in structured_filters.items()
                ],
                "</ul>",
            ]
        )
    else:
        html_content.extend(["<h2>Selected Filters</h2>", "<p>No filters selected</p>"])

    # Modified plotly graph objects visualization section
    if go_figs:
        for name, fig in go_figs.items():
            im_path = os.path.join(img_dir, f"{name}_go_fig.png")
            fig.write_image(im_path)
            html_content.extend(
                [
                    f"<h2>{name} Analysis</h2>",
                    f'<img src="{im_path}" style="max-width:100%">',
                ]
            )

    # Add all visualizations
    html_content.append("<h2>Visualizations</h2>")
    for i, viz in enumerate(visualizations):
        img_path = os.path.join(img_dir, f"plot_{i}.png")
        viz["figure"].write_image(img_path)
        html_content.extend(
            [
                f"<div class='plot-container'>",
                f"<h3>{viz['title']}</h3>",
                f'<img src="{os.path.basename(img_dir)}/plot_{i}.png" style="max-width:100%">',
                "</div>",
            ]
        )

    # Add summary statistics
    html_content.extend(
        [
            "<h2>Summary Statistics</h2>",
            "<div class='stats-container'>",
            df.describe().to_html(),
            "</div>",
        ]
    )

    # Add data table
    # Truncate any long text cells to 1000 chars
    df_truncated = df.copy()
    truncate_length = 200
    for col in df_truncated.select_dtypes(["object"]):
        df_truncated[col] = (
            df_truncated[col]
            .astype(str)
            .apply(
                lambda x: (
                    x[:truncate_length] + "..." if len(x) > truncate_length else x
                )
            )
        )
    html_content.extend(
        [
            "<h2>Data Sample</h2>",
            "<div class='data-container'>",
            df_truncated.head(100).to_html(),  # First 100 rows
            "</div>",
            "</body></html>",
        ]
    )

    all_html_content = "\n".join(html_content)
    if format == "html":
        with open(full_path, "w") as f:
            f.write(all_html_content)
    else:
        pdf_from_html(all_html_content, full_path)
    return full_path


def export_dashboard(
    df: pd.DataFrame,
    structured_filters: dict[str, t.Any],
    visualizations: list[dict],
    base_path: str,
    title: str,
    go_figs: dict[str, go.Figure],
    format: str = "html",
) -> str:
    """
    Export dashboard including data, visualizations, and analytics.

    Args:
        df: DataFrame containing all data
        visualizations: List of visualization dictionaries with figures and titles
        base_path: Base directory path where file should be saved
        title: Title of the dashboard
        format: Export format ("html", "pdf", "csv", "xlsx")
        go_figs: Optional dict of plotly graph objects to include in export as images

    Note on formats:
    - data-only export formats ('csv', 'xlsx') will just dump a dataframe with the selected rows, no visualizations
    - "pretty" export formats ('html', 'pdf') will render the entire dashboard and save as a single artifact

    Returns:
        Path where file was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(base_path, "exports")
    os.makedirs(export_dir, exist_ok=True)

    base_filename = f"dashboard_export_{timestamp}"
    export_path = os.path.join(export_dir, base_filename)

    if format in ["csv", "xlsx"]:
        return data_only_export(df, export_path, format)
    else:
        return pretty_dashboard_export(
            df=df,
            export_path=export_path,
            title=title,
            structured_filters=structured_filters,
            visualizations=visualizations,
            format=format,
            go_figs=go_figs,
        )


def export_options(
    filtered_df: pd.DataFrame,
    structured_filters: dict[str, t.Any],
    visualizations: list[dict],
    title: str,
    go_figs: dict[str, go.Figure],
) -> None:
    """Display and handle export controls for the dashboard.

    Args:
        filtered_df: DataFrame to be exported
        visualizations: List of visualization dictionaries
        cluster_fig: Optional plotly figure for cluster visualization
    """
    st.header("Export Options")
    export_col1, export_col2, export_col3, _ = st.columns([1, 1, 1, 1])

    with export_col1:
        export_path = st.text_input(
            "Export Directory",
            value=ARES_DATA_DIR,
            help="Directory where exported files will be saved",
        )

    with export_col2:
        export_format = st.selectbox(
            "Export Format",
            options=["html", "pdf", "csv", "xlsx"],
            help="Choose the format for your export. HTML/PDF include visualizations. CSV/XLSX include filtered data only.",
        )

    with export_col3:
        if st.button("Export Dashboard"):
            try:
                with st.spinner(f"Exporting dashboard as {export_format}..."):
                    export_path = export_dashboard(
                        df=filtered_df,
                        structured_filters=structured_filters,
                        visualizations=visualizations,
                        base_path=export_path,
                        title=title,
                        go_figs=go_figs,
                        format=export_format,
                    )
                    st.success(f"Dashboard exported successfully to: {export_path}")
            except Exception as e:
                st.error(
                    f"Failed to export dashboard: {str(e)}\n{traceback.format_exc()}"
                )
