"""
SMP (Single Member Plurality) plotting functions for comparison with Majority Judgment.

This module contains validated plotting functions for comparing traditional polls (SMP)
with majority judgment results. These functions work with the refactored SMPData class.

Functions:
- comparison_ranking_plot: Compare MJ rankings with SMP rankings
- plot_comparison_intention: Compare voting intentions between SMP and MJ for a candidate
- plot_intention: Plot intention data for a candidate (helper function)
- plot_intention_data: Plot raw SMP data points (helper function)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pandas import DataFrame

from ..core.smp_data import SMPData
from ..plotting.plot_utils import load_colors, _extended_name_annotations, _add_image_to_fig
from ..plotting.plots import ranking_plot, plot_time_merit_profile


def comparison_ranking_plot(
    df: DataFrame, smp_data: SMPData, source: str = None, on_rolling_data: bool = False
) -> go.Figure:
    """
    Create a comparison plot of candidate rankings between MJ and SMP.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing MJ survey data.
    smp_data : SMPData
        SMPData object with SMP polling data.
    source : str, optional
        Source attribution for MJ data.
    on_rolling_data : bool, optional
        Whether to use rolling average data. Default is False.

    Returns
    -------
    go.Figure
        Plotly figure with two subplots showing MJ and SMP rankings.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)

    # Plot MJ ranking (top subplot)
    fig = ranking_plot(
        df,
        source=None,
        sponsor=None,
        show_best_grade=False,
        show_rank=True,
        show_no_opinion=False,
        show_grade_area=False,
        breaks_in_names=False,
        fig=fig,
        row=1,
        col=1,
        on_rolling_data=on_rolling_data,
    )

    # Get SMP ranking data filtered by MJ date range
    df_smp = smp_data.get_ranks()
    df_smp = df_smp[df_smp["end_date"] >= df["end_date"].min()]

    # Plot SMP ranking (bottom subplot)
    fig = ranking_plot(
        df_smp,
        source=None,
        sponsor=None,
        show_best_grade=False,
        show_rank=True,
        show_no_opinion=False,
        show_grade_area=False,
        breaks_in_names=False,
        fig=fig,
        row=2,
        col=1,
    )

    # Configure layout
    fig.update_yaxes(row=2, col=1, visible=False, autorange="reversed", title="Scrutin uninominal")
    fig.update_layout(width=1200, height=800)

    source_str = f"sources jugement majoritaire: {source}"
    title = (
        "<b>Comparaison des classement des candidats à l'élection présidentielle 2027"
        + "<br> au jugement majoritaire et au scrutin uninominal</b><br>"
        + f"<i>{source_str}"
        + "<br>sources scrutin uninominal: nsppolls.fr </i>"
    )
    fig.update_layout(title=title, title_x=0.5)

    return fig


def plot_intention(
    df: DataFrame,
    col_intention: str,
    fig: go.Figure = None,
    colored: bool = True,
    row: int = None,
    col: int = None,
) -> go.Figure:
    """
    Plot voting intention line for a candidate (helper function).

    Parameters
    ----------
    df : DataFrame
        DataFrame with candidate intention data.
    col_intention : str
        Name of the column containing intention values.
    fig : go.Figure, optional
        Existing figure to add trace to.
    colored : bool, optional
        Whether to use candidate's color (True) or gray (False).
    row : int, optional
        Subplot row position.
    col : int, optional
        Subplot column position.

    Returns
    -------
    go.Figure
        Updated figure with intention line.
    """
    candidate = df["candidat"].unique()[0]
    colors = load_colors()
    color = colors[candidate]["couleur"] if colored else "#d3d3d3"
    opacity = 1 if colored else 0.3
    width = 3 if colored else 1

    fig.add_trace(
        go.Scatter(
            x=df["fin_enquete"],
            y=df[col_intention],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            showlegend=False,
            hovertemplate=f"<b>{candidate}</b><br>"
            + "Date: %{x|%d/%m/%Y}<br>"
            + "Intention: %{y:.1f}%<br>"
            + "<extra></extra>",
        ),
        row=row,
        col=col,
    )
    return fig


def plot_intention_data(
    df: DataFrame,
    col_intention: str,
    fig: go.Figure = None,
    colored: bool = True,
    row: int = None,
    col: int = None,
) -> go.Figure:
    """
    Plot raw SMP data points for a candidate (helper function).

    Parameters
    ----------
    df : DataFrame
        DataFrame with raw SMP polling data.
    col_intention : str
        Name of the column containing intention values.
    fig : go.Figure, optional
        Existing figure to add trace to.
    colored : bool, optional
        Whether to use candidate's color (True) or gray (False).
    row : int, optional
        Subplot row position.
    col : int, optional
        Subplot column position.

    Returns
    -------
    go.Figure
        Updated figure with data points.
    """
    candidate = df["candidat"].unique()[0]
    colors = load_colors()
    color = colors[candidate]["couleur"] if colored else "#d3d3d3"
    opacity = 1 if colored else 0.3

    fig.add_trace(
        go.Scatter(
            x=df["fin_enquete"],
            y=df[col_intention],
            mode="markers",
            marker=dict(color=color, size=5),
            opacity=opacity,
            showlegend=False,
            hovertemplate=f"<b>{candidate}</b><br>"
            + "Date: %{x|%d/%m/%Y}<br>"
            + "Intention: %{y:.1f}%<br>"
            + "<extra></extra>",
        ),
        row=row,
        col=col,
    )
    return fig


def plot_comparison_intention(
    df: DataFrame,
    smp_data: SMPData,
    source: str = None,
    sponsor: str = None,
    on_rolling_data: bool = False,
) -> go.Figure:
    """
    Create a comparison plot of voting intentions between SMP and MJ for one candidate.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing MJ data for one candidate.
    smp_data : SMPData
        SMPData object with SMP polling data.
    source : str, optional
        Source attribution for MJ data.
    sponsor : str, optional
        Sponsor attribution for MJ survey.
    on_rolling_data : bool, optional
        Whether to use rolling average data. Default is False.

    Returns
    -------
    go.Figure
        Plotly figure with side-by-side comparison of SMP intentions and MJ merit profile.
    """
    # Prepare subplot titles
    subplot_title_1 = "<b>Scrutin uninominal</b>"
    subplot_title_1 += "<br><i>source: nsppolls.fr</i>"
    subplot_title_2 = "<b>Jugement majoritaire</b>"
    subplot_title_2 += f"<br><i>source: {source}</i>" if source is not None else ""
    subplot_title_2 += f", commanditaire: {sponsor}</i>" if sponsor is not None else ""

    candidate = df["candidate"].unique()[0]  # MJ uses "candidate" column

    # Create figure with two subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=(subplot_title_1, subplot_title_2),
        vertical_spacing=1,
    )

    # Plot MJ merit profile (right subplot)
    fig = plot_time_merit_profile(
        df,
        source=None,
        sponsor=None,
        show_no_opinion=True,
        fig=fig,
        row=1,
        col=2,
        on_rolling_data=on_rolling_data,
        show_logo=False,
    )

    # Add marker at last date for MJ
    fig.add_trace(
        go.Scatter(
            x=df["end_date"].iloc[-1:],
            y=[50],
            mode="markers",
            marker=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Add candidate name annotation
    ext_candidate = _extended_name_annotations(
        df, candidate=candidate, show_rank=True, show_best_grade=True, show_no_opinion=True, breaks_in_names=False
    )
    fig["layout"]["annotations"] += (
        dict(
            x=pd.to_datetime(df["end_date"].iloc[-1:].tolist()[0]),
            y=35 / 2,
            xanchor="center",
            xshift=65,
            yshift=30,
            yanchor="middle",
            text=f"<b>{ext_candidate}</b>",
            font=dict(family="Arial", color="black"),
            showarrow=False,
            xref="x2",
            yref="y1",
        ),
    )

    # Get SMP ranking data filtered by MJ date range
    df_smp = smp_data.get_ranks()
    df_smp = df_smp[df_smp["end_date"] >= df["end_date"].min()]

    # Check if candidate exists in SMP data
    df_smp_candidate = df_smp[df_smp["candidat"] == candidate]
    if df_smp_candidate.empty:
        # Candidate not in SMP data, return None to skip
        print(f"  ⚠️  Skipping {candidate}: no SMP data available")
        return None

    # Plot other candidates in gray (left subplot)
    df_smpother = df_smp[df_smp["candidat"] != candidate]
    for c in df_smpother["candidat"]:
        df_temp = df_smpother[df_smpother["candidat"] == c]
        fig = plot_intention(df_temp, col_intention="valeur", fig=fig, row=1, col=1, colored=False)

    # Plot target candidate in color (left subplot)
    fig = plot_intention(df_smp_candidate, col_intention="valeur", fig=fig, row=1, col=1, colored=True)

    # Add raw SMP data points for target candidate
    df_smp_data = smp_data.get_intentions()
    df_smp_data = df_smp_data[df_smp_data["end_date"] >= df["end_date"].min()]
    df_smp_data = df_smp_data[df_smp_data["candidat"] == candidate]
    fig = plot_intention_data(df_smp_data, col_intention="intentions", fig=fig, row=1, col=1, colored=True)

    # Configure axes
    fig.update_yaxes(row=1, col=1, visible=True, title="Intention de vote (%)", range=[0, 35])
    fig.update_yaxes(row=1, col=2, visible=True, title="Mentions (%)", range=[0, 100])
    fig.update_xaxes(
        row=1,
        col=1,
        range=[df_smp["end_date"].min(), "2027-04-15"],  # Updated for 2027
        visible=True,
        ticklabelposition="outside bottom",
    )
    fig.update_xaxes(
        row=1,
        col=2,
        range=[df_smp["end_date"].min(), "2027-04-15"],  # Updated for 2027
        visible=True,
        ticklabelposition="outside bottom",
    )

    # Configure layout
    title = f"<b>Comparaison des intentions de votes à l'élection présidentielle 2027<br>{candidate}</b><br>"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.95),
        width=1200,
        height=600,
        legend=dict(orientation="h", x=0.5, xanchor="center", font=dict(size=12)),
    )
    fig = _add_image_to_fig(fig, x=1.00, y=1.1, sizex=0.10, sizey=0.10, xanchor="right")

    return fig
