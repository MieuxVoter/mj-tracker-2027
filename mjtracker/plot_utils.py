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

FIRST_ROUND_DATE = "2022-04-10"
URL_IMAGE_MIEUX_VOTER = "https://raw.githubusercontent.com/MieuxVoter/majority-judgment-tracker/main/icons/logo.svg"


def load_colors() -> dict:
    """
    Load the colors used for the different candidates
    ---
    Returns
    -------
    a dict of colors
    """
    return CANDIDATS


def _extended_name_annotations(
    df,
    candidate: str = None,
    breaks_in_names: bool = False,
    show_grade_area: bool = False,
    show_best_grade: bool = False,
    show_no_opinion: bool = False,
    show_intention: bool = False,
    show_rank: bool = False,
):
    """
    Get the name of the candidate formatted to be displayed on figures

     Parameters
    ----------
    df : DataFrame
        DataFrame containing the surveys
    candidate : str
        Name of the candidate to format the annotations
    breaks_in_names : bool
        If True, the name of the candidate is split in several lines
    show_grade_area : bool
        If True, the grade area is displayed
    show_best_grade : bool
        If True, the best grade is displayed
    show_no_opinion : bool
        If True, the number of people who did not give a opinion is displayed
    show_intention : bool
        If True, the intention of voters for the candidate is displayed
    show_rank : bool
        If True, the rank of the candidate is displayed

    Returns
    -------
    The annotations of the name of the candidate
    """

    if breaks_in_names:
        idx_space = candidate.find(" ")
        name_label = candidate[:idx_space] + "<br>" + candidate[idx_space + 1 :]
    else:
        name_label = candidate

    extended_name_label = f"<b>{name_label}</b>"
    if show_rank:
        extended_name_label += " " + rank2str(df["rang"].iloc[-1])
        if show_best_grade and df["mention_majoritaire"].iloc[-1] != "nan":
            extended_name_label += (
                "<br>" + df["mention_majoritaire"].iloc[-1][0].upper() + df["mention_majoritaire"].iloc[-1][1:]
            )
        if show_no_opinion and "sans opinion" in df.columns and not np.isnan(df["sans_opinion"].iloc[-1]):
            extended_name_label += "<br><i>(sans opinion " + str(df["sans_opinion"].iloc[-1]) + "%)</i>"
    if show_best_grade and not show_grade_area and not show_rank:
        extended_name_label += (
            "<br>" + df["mention_majoritaire"].iloc[-1][0].upper() + df["mention_majoritaire"].iloc[-1][1:]
        )
        if show_no_opinion and not np.isnan(df["sans_opinion"].iloc[-1]):
            extended_name_label += " <i>(sans opinion " + str(df["sans_opinion"].iloc[-1]) + "%)</i>"

    if show_intention:
        extended_name_label += "<br>(" + str(round(df["valeur"].iloc[-1], 1)) + " %)"

    return extended_name_label


def _add_election_date(fig: go.Figure, y: float = 34, xshift: float = 0, row: int = None, col: int = None):
    """
    Add the date of the election to the figure.

    Parameters
    ----------
    fig : go.Figure
       figure to add the date to
    row : int
        Row of the subplot
    Returns
    -------
    The figure with the date of the election
    """
    xref = f"x{col}" if row is not None else None
    yref = f"y{row}" if row is not None else None

    fig.add_vline(x=FIRST_ROUND_DATE, line_dash="dot", row=row, col=col, line=dict(color="rgba(0,0,0,0.5)"))
    fig["layout"]["annotations"] += (
        dict(
            x=FIRST_ROUND_DATE,
            y=y,
            xanchor="left",
            xshift=xshift,
            yanchor="middle",
            text="1er Tour",
            font=dict(family="Arial", size=12),
            showarrow=False,
            xref=xref,
            yref=yref,
        ),
    )

    return fig


def _generate_windows_size(nb: int) -> tuple:
    """
    Defines the number of column and rows of subplots from the number of variables to plot.

    Parameters
    ----------
    nb: int
        Number of variables to plot

    Returns
    -------
    The optimized number of rows and columns
    """
    if nb > 2:
        n_rows = int(round(np.sqrt(nb)))
        return n_rows + 1 if n_rows * n_rows < nb else n_rows, n_rows
    else:
        return 1, 2


def _add_image_to_fig(
    fig: go.Figure, x: float, y: float, sizex: float, sizey: float, xanchor: str = "left"
) -> go.Figure:
    """
    Add mieux voter logo to the figure

    Parameters
    ----------
    fig : go.Figure
       figure to add the date to
    x : float
        x position of the logo
    y : float
        y position of the logo
    sizex : float
        x size of the logo
    sizey : float
        y size of the logo
    xanchor : str
        x anchor of the logo (left, center, right)
    Returns
    -------
    The figure with the logo
    """
    fig.add_layout_image(
        dict(
            source=URL_IMAGE_MIEUX_VOTER,
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            sizex=sizex,
            sizey=sizey,
            xanchor=xanchor,
            yanchor="bottom",
        )
    )
    return fig


def export_fig(fig: go.Figure, args, filename: str):
    """
    Export the figure to a file in the specified format

    Parameters
    ----------
    fig : go.Figure
       figure to add the date to
    args : Arguments
        Arguments of the export
    filename : str
        name of the file to export
    """
    if args.show:
        fig.show(config=dict(displaylogo=False))
    if args.html:
        fig.write_html(f"{args.dest}/{filename}.html", config=dict(displaylogo=False))
    if args.png:
        fig.write_image(f"{args.dest}/{filename}.png", height=fig.layout.height, width=fig.layout.width, scale=3)
    if args.svg:
        fig.write_image(f"{args.dest}/{filename}.svg")
    if args.json:
        # dont resize the figure to handle react.js and remove interactive mode for legends
        fig.update_layout(width=None, height=None, legend_itemclick=False)
        filename = f"{args.dest}/{filename}.json"
        fig.write_json(filename)
