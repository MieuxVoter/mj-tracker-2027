"""
Legacy plotting functions for SMP (Single Member Plurality) data analysis.

⚠️ MAINTENANCE MODE ⚠️
This module contains older plotting functions for traditional polling data (intentions de vote).
New code should use batch_plots_smp.py or plots_v2.py instead.

These functions are kept for:
- Backward compatibility with existing scripts
- Historical SMP data visualization
- Comparing traditional polls with majority judgment results

If you need to add new features, please use the newer modules.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from seaborn import color_palette
import numpy as np
import pandas as pd
from pandas import DataFrame
from .smp_data import SMPData
from .utils import get_intentions_colheaders, get_candidates, get_grades, rank2str
from .misc.enums import PollingOrganizations, AggregationMode
from .constants import CANDIDATS
from .plot_utils import load_colors, _extended_name_annotations, _add_image_to_fig, _add_election_date
from .plots import plot_time_merit_profile


def plot_intention(
    df: DataFrame,
    col_intention: str,
    fig: go.Figure = None,
    colored: bool = True,
    row: int = None,
    col: int = None,
) -> go.Figure:
    candidate = df["candidat"].unique()[0]
    colors = load_colors()
    color = colors[candidate]["couleur"] if colored else "#d3d3d3"
    opacity = 1 if colored else 0.3
    width = 3 if colored else 1
    fig.add_trace(
        go.Scatter(
            x=df["fin_enquete"],
            y=df[col_intention],
            hoverinfo="all",
            mode="lines",
            line=dict(color=color, width=width),
            name=candidate,
            showlegend=False,
            legendgroup=None,
        ),
        row=row,
        col=col,
    )
    rank = df["rang"].iloc[-1:].to_numpy()[0]
    fig.add_trace(
        go.Scatter(
            x=df["fin_enquete"].iloc[-1:],
            y=df[col_intention].iloc[-1:],
            mode="markers",
            name=candidate,
            marker=dict(color=color, opacity=opacity),
            legendgroup=None,
            showlegend=False,
            text=[f"{rank2str(rank)}"],
        ),
        row=row,
        col=col,
    )
    if colored:
        c = px.colors.hex_to_rgb(color)
        opacity = 0.2 if colored else 0.02
        c_alpha = str(f"rgba({c[0]},{c[1]},{c[2]},{opacity})")
        x_date = df["fin_enquete"].tolist()
        y_upper = df["erreur_sup"].tolist()
        y_lower = df["erreur_inf"].tolist()

        fig.add_scatter(
            x=x_date + x_date[::-1],  # x, then x reversed
            y=y_upper + y_lower[::-1],  # upper, then lower reversed
            fill="toself",
            fillcolor=c_alpha,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name=candidate,
            legendgroup=None,
            row=row,
            col=col,
        )

        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None
        candidate = _extended_name_annotations(
            df, candidate=candidate, show_rank=True, show_intention=True, breaks_in_names=True
        )
        fig["layout"]["annotations"] += (
            dict(
                x=pd.to_datetime(df["fin_enquete"].iloc[-1:].tolist()[0]),
                y=df[col_intention].iloc[-1:].tolist()[0],
                xanchor="left",
                xshift=10,
                yanchor="middle",
                text=f"<b>{candidate}</b>",
                font=dict(family="Arial", size=12, color=color),
                showarrow=False,
            ),
        )

    return fig


def plot_intention_data(
    df, col_intention: str, fig: go.Figure = None, colored: bool = True, row: int = None, col: int = None
) -> go.Figure:
    candidate = df["candidat"].unique()[0]
    colors = load_colors()
    color = colors[candidate]["couleur"] if colored else "lightgray"
    opacity = 0.5 if colored else 0.25
    hoverinfo = "all" if colored else "y+name"
    fig.add_trace(
        go.Scatter(
            x=df["fin_enquete"],
            y=df[col_intention],
            hoverinfo=hoverinfo,
            mode="markers",
            marker=dict(color=color, opacity=opacity, size=2),
            name=candidate,
            showlegend=False,
            legendgroup=None,
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
    Plot the intention of the candidates in the two voting systems.

    ----------
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data of the survey.
    smp_data : SMPData
        SMPData containing the data of the uninominal survey.
    source : str, optional
        Source of the data survey.
    sponsor : str, optional
        Sponsor of the data survey.
    on_rolling_data : bool, optional
        If True, the data is on rolling data.
    ----------
    Returns
    -------
    fig : go.Figure
        Figure containing the plot.
    """
    subplot_title_1 = "<b>Scrutin uninominal</b>"
    subplot_title_1 += f"<br><i>source: nsppolls.fr</i>"
    subplot_title_2 = "<b>Jugement majoritaire</b>"
    subplot_title_2 += f"<br><i>source: {source}</i>" if source is not None else ""
    subplot_title_2 += f", commanditaire: {sponsor}</i>" if source is not None else ""
    candidate = df["candidat"].unique()[0]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=(subplot_title_1, subplot_title_2),
        vertical_spacing=1,
    )

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
    fig.add_trace(
        go.Scatter(
            x=df["fin_enquete"].iloc[-1:],
            y=[50],
            mode="markers",
            marker=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    ext_candidate = _extended_name_annotations(
        df, candidate=candidate, show_rank=True, show_best_grade=True, show_no_opinion=True, breaks_in_names=False
    )
    fig["layout"]["annotations"] += (
        dict(
            x=pd.to_datetime(df["fin_enquete"].iloc[-1:].tolist()[0]),
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

    df_smp = smp_data.get_ranks()
    df_smp = df_smp[df_smp["fin_enquete"] >= df["fin_enquete"].min()]

    df_smpother = df_smp[df_smp["candidat"] != df["candidat"].unique()[0]]
    for c in df_smpother["candidat"]:
        df_temp = df_smpother[df_smpother["candidat"] == c]
        fig = plot_intention(df_temp, col_intention="valeur", fig=fig, row=1, col=1, colored=False)

    df_smp = df_smp[df_smp["candidat"] == df["candidat"].unique()[0]]
    fig = plot_intention(df_smp, col_intention="valeur", fig=fig, row=1, col=1, colored=True)

    df_smp_data = smp_data.get_intentions()
    df_smp_data = df_smp_data[df_smp_data["fin_enquete"] >= df["fin_enquete"].min()]

    df_smp_data = df_smp_data[df_smp_data["candidat"] == df["candidat"].unique()[0]]
    fig = plot_intention_data(df_smp_data, col_intention="intentions", fig=fig, row=1, col=1, colored=True)

    # fig = _add_election_date(fig=fig, row=1, col=1)
    # fig = _add_election_date(fig=fig, row=1, col=2)
    fig.update_yaxes(row=1, col=1, visible=True, title="Intention de vote (%)", range=[0, 35])
    fig.update_yaxes(row=1, col=2, visible=True, title="Mentions (%)", range=[0, 100])
    fig.update_xaxes(
        row=1,
        col=1,
        range=[df_smp["fin_enquete"].min(), "2022-04-15"],
        visible=True,
        ticklabelposition="outside bottom",
    )
    fig.update_xaxes(
        row=1,
        col=2,
        range=[df_smp["fin_enquete"].min(), "2022-04-15"],
        visible=True,
        ticklabelposition="outside bottom",
    )

    title = "<b>Comparaison des intentions de votes à l'élection présidentielle 2027" + f"<br>{candidate}</b><br>"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.95),
        width=1200,
        height=600,
        legend=dict(orientation="h", x=0.5, xanchor="center", font=dict(size=12)),
    )
    fig = _add_image_to_fig(fig, x=1.00, y=1.1, sizex=0.10, sizey=0.10, xanchor="right")

    return fig
