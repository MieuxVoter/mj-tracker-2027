import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from seaborn import color_palette
import numpy as np
import pandas as pd
from pandas import DataFrame

from . import SurveyInterface
from . import SurveysInterface
from .smp_data import SMPData
from .utils import get_intentions_colheaders, get_candidates, get_grades, rank2str
from .misc.enums import PollingOrganizations, AggregationMode
from .constants import CANDIDATS
from .color_utils import get_grade_color_palette
from .plot_utils import load_colors, _add_image_to_fig, _generate_windows_size, _add_election_date


def plot_animated_merit_profile(
    df: DataFrame,
    grades: list,
    font_size: int = 20,
    date: str = None,
    sponsor: str = None,
    source: str = None,
    show_no_opinion: bool = True,
) -> go.Figure:
    """
    This function creates an animated plot of the merit profile.
    It successively plots the scores for each grade for each candidates.
    First it plots the scores for the first grade, with ordered candidates.
    Then it plots the scores for the second grade, with re-ordered candidates in function of the total score, etc...

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the data to plot.
    grades : list
        The list of grades to plot.
    font_size : int, optional
        The font size, by default 20
    date : str, optional
        The date of the data, by default None
    sponsor : str, optional
        The sponsor of the data, by default None
    source : str, optional
        The source of the data, by default None
    show_no_opinion : bool, optional
        If True, the no opinion data is plotted, by default True

    Returns
    -------
    go.Figure
        The figure.
    """
    df = df.copy()
    nb_grades = len(grades)
    grades = get_intentions_colheaders(df, nb_grades)

    # ANIMATION DATAFRAME
    df_animation = df.copy()
    # add a column animation_sequence 0 for all rows
    df_animation["animation_sequence"] = 0
    # set to zero all grades except the first one
    for grade in grades[1:]:
        df_animation[grade] = 0

    # duplicate concatenate the dataframe 7 times with animation_sequence 1 to 6
    for i in range(1, len(grades) + 1):
        df_temp = df.copy()
        df_temp["animation_sequence"] = i
        # set to zero all grades except to the i-th one
        if i + 1 < len(grades):
            for grade in grades[i + 1 :]:
                df_temp[grade] = 0
        # concatenate the dataframe
        df_animation = pd.concat([df_animation, df_temp])

    # compute the list sorted of candidat names to order y axis.
    candidat_list = list(df["candidate"])
    rank_list = list(df["rang"] - 1)
    sorted_candidat_list = [i[1] for i in sorted(zip(rank_list, candidat_list))]
    r_sorted_candidat_list = sorted_candidat_list.copy()
    r_sorted_candidat_list.reverse()

    colors = color_palette(palette="coolwarm", n_colors=nb_grades)
    color_dict = {f"intention_mention_{i + 1}": f"rgb{str(colors[i])}" for i in range(nb_grades)}

    # build the dataframe for the animation
    fig = px.bar(
        df_animation,
        x=grades,
        y="candidate",
        orientation="h",
        color_discrete_map=color_dict,
        animation_frame="animation_sequence",
        animation_group="candidate",
    )

    # animate smooth transition of 0.5 seconds
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
    # wait 0.5 before going to the next frame
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

    fig.update_traces(textfont_size=font_size, textangle=0, textposition="auto", cliponaxis=False, width=0.5)

    # replace variable names with grades
    new_names = {f"intention_mention_{i + 1}": grades[i] for i in range(nb_grades)}
    fig.for_each_trace(
        lambda t: t.update(
            name=new_names[t.name],
            legendgroup=new_names[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_names[t.name]),
        )
    )

    # vertical line
    fig.add_vline(x=50, line_width=2, line_color="black")

    # Legend
    fig.update_layout(
        legend_title_text=None,
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width
    )

    fig.update(data=[{"hovertemplate": "Intention: %{x}<br>Candidat: %{y}"}])
    # todo: need to plot grades in hovertemplate.

    # no back ground
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # xticks and y ticks
    # Add sans opinion to y tick label # todo : it may be simplified !
    if show_no_opinion and not np.isnan(df["sans_opinion"].unique()[0]):
        df["candidat_sans_opinion"] = None
        for ii, cell in enumerate(df["candidate"]):
            df["candidat_sans_opinion"].iat[ii] = (
                "<b>" + cell + "</b>" + "     <br><i>(sans opinion " + str(df["sans_opinion"].iloc[ii]) + "%)</i>     "
            )
        # compute the list sorted of candidat names to order y axis.
        candidat_list = list(df["candidat_sans_opinion"])
        rank_list = list(df["rang"] - 1)
        sorted_candidat_list = [i[1] for i in sorted(zip(rank_list, candidat_list))]
        r_sorted_candidat_no_opinion_list = sorted_candidat_list.copy()
        r_sorted_candidat_no_opinion_list.reverse()
        yticktext = r_sorted_candidat_no_opinion_list
    else:
        yticktext = ["<b>" + s + "</b>" + "     " for s in r_sorted_candidat_list]

    # xticks and y ticks
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            tickmode="array",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            tickfont_size=font_size,
            title="",  # intentions
        ),
        yaxis=dict(
            tickfont_size=font_size * 0.75,
            title="",  # candidat
            automargin=True,
            ticklabelposition="outside left",
            ticksuffix="   ",
            tickmode="array",
            tickvals=[i for i in range(len(df))],
            ticktext=yticktext,
            categoryorder="array",
            categoryarray=r_sorted_candidat_list,
        ),  # space
    )

    # Title and detailed

    date_str = f"date: {date}, " if date is not None else ""
    source_str = f"source: {source}" if source is not None else ""
    source_str += ", " if sponsor is not None else ""
    sponsor_str = f"commanditaire: {sponsor}" if sponsor is not None else ""
    title = "<b>Evaluation au jugement majoritaire</b> <br>" + f"<i>{date_str}{source_str}{sponsor_str}</i>"
    fig.update_layout(title=title, title_x=0.5)

    # font family
    fig.update_layout(font_family="arial")

    fig = _add_image_to_fig(fig, x=0.9, y=1.01, sizex=0.15, sizey=0.15)

    # size of the figure
    fig.update_layout(width=1000, height=600)

    return fig


def comparison_ranking_plot(df, smp_data: SMPData, source: str = None, on_rolling_data: bool = False) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)

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

    df_smp = smp_data.get_ranks()
    df_smp = df_smp[df_smp["fin_enquete"] >= df["fin_enquete"].min()]

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


def plot_time_merit_profile(
    df: DataFrame,
    fig: go.Figure = None,
    sponsor: str = None,
    source: str = None,
    show_no_opinion: bool = True,
    show_legend: bool = True,
    show_logo: bool = True,
    on_rolling_data: bool = False,
    no_layout: bool = False,
    row: int = None,
    col: int = None,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    suffix = "_roll" if on_rolling_data else ""

    nb_grades = df["nombre_mentions"].unique()[0]
    colors = color_palette(palette="coolwarm", n_colors=nb_grades)
    col_intentions = [f"intention_mention_{i}{suffix}" for i in range(1, nb_grades + 1)]
    color_dict = {col: f"rgb{str(colors[i])}" for i, col in enumerate(col_intentions)}

    y_cumsum = df[col_intentions].to_numpy()

    grade_list = get_grades(df)
    grade_list.reverse()
    col_intentions.reverse()
    y_cumsum = np.flip(y_cumsum.T, axis=0)

    for g, cur_int, cur_y in zip(grade_list, col_intentions, y_cumsum):
        fig.add_trace(
            go.Scatter(
                x=df["end_date"],
                y=cur_y,
                hoverinfo="x+y",
                mode="lines",
                line=dict(width=0.5, color=color_dict[cur_int]),
                stackgroup="one",  # define stack group
                name=g,
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
    # if show_no_opinion:
    #     fig = add_no_opinion_time_merit_profile(fig, df, suffix, row=row, col=col, show_legend=show_legend)

    for d in df["end_date"]:
        fig.add_vline(x=d, line_dash="dot", line_width=1, line_color="black", opacity=0.2, row=row, col=col)

    fig.add_hline(
        y=50,
        line_dash="solid",
        line_width=1,
        line_color="black",
        annotation_text="50 %",
        annotation_position="bottom right",
        row=row,
        col=col,
    )

    if not no_layout:
        fig.update_layout(
            yaxis_range=(0, 100),
            width=1200,
            height=800,
            legend_title_text="Mentions",
            autosize=True,
            legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width/
            yaxis=dict(
                tickfont_size=15,
                title="Mentions (%)",  # candidat
                automargin=True,
            ),
            plot_bgcolor="white",
        )

        # Title and detailed
        date = df["end_date"].max()
        source_str = f"source: {source}" if source is not None else ""
        source_str += ", " if sponsor is not None else ""
        sponsor_str = f"commanditaire: {sponsor}" if sponsor is not None else ""
        title = (
            f"<b>Evolution des mentions au jugement majoritaire"
            + f"<br> pour le candidat {df.candidate.unique().tolist()[0]}</b><br>"
            + f"<i>{source_str}{sponsor_str}, dernier sondage: {date}.</i>"
        )
        fig.update_layout(title=title, title_x=0.5)
        if show_logo:
            fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    return fig


def plot_ranked_time_merit_profile(
    df: DataFrame,
    sponsor: str = None,
    source: str = None,
    show_no_opinion: bool = True,
    on_rolling_data: bool = False,
) -> go.Figure:
    # Candidat list sorted the rank in the last poll
    most_recent_date = df["end_date"].max()
    temp_df = df[df["end_date"] == most_recent_date]
    temp_df = temp_df.sort_values(by="rang_glissant" if on_rolling_data else "rang")
    candidates = get_candidates(temp_df)
    titles_candidates = [f"{c} {rank2str(i+1)}" for i, c in enumerate(candidates)]

    # size of the figure
    n_rows, n_cols = _generate_windows_size(len(candidates))
    idx_rows, idx_cols = np.unravel_index([i for i in range(len(candidates))], (n_rows, n_cols))
    idx_rows += 1
    idx_cols += 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=titles_candidates,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    show_legend = True
    for row, col, c in zip(idx_rows, idx_cols, candidates):
        temp_df = df[df["candidate"] == c]
        fig = plot_time_merit_profile(
            df=temp_df,
            fig=fig,
            on_rolling_data=on_rolling_data,
            show_no_opinion=show_no_opinion,
            show_legend=show_legend,
            no_layout=True,
            row=row,
            col=col,
        )
        fig.update_yaxes(range=[0, 100], row=row, col=col, title="Mentions (%)" if col == 1 else "")
        show_legend = False

    fig.update_layout(
        yaxis_range=(0, 100),
        width=1200,
        height=900 if n_rows > 1 else 450,
        legend_title_text="Mentions",
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width/
        yaxis=dict(
            tickfont_size=15,
            title="Mentions (%)",  # candidat
            automargin=True,
        ),
        plot_bgcolor="white",
    )

    # Title and detailed
    source_str = f"source: {source}" if source is not None else ""
    source_str += ", " if sponsor is not None else ""
    sponsor_str = f"commanditaire: {sponsor}" if sponsor is not None else ""
    pop_str = " - population: " + str(df["population"].iloc[0]) if "population" in df.columns else ""
    title = (
        f"<b>Classement des candidats au jugement majoritaire</b>{pop_str}"
        + f"<br><i>{source_str}{sponsor_str}, dernier sondage: {most_recent_date}.</i>"
    )
    fig.update_layout(title=title, title_x=0.5)
    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    return fig


def plot_time_merit_profile_all_polls(df, aggregation, on_rolling_data: bool = False) -> go.Figure:
    name_subplot = tuple([poll.value for poll in PollingOrganizations if poll != PollingOrganizations.ALL])
    suffix = "_roll" if on_rolling_data else ""
    fig = make_subplots(rows=3, cols=1, subplot_titles=name_subplot)
    count = 0
    date_max = df["fin_enquete"].max()
    date_min = df["fin_enquete"].min()

    if aggregation == AggregationMode.NO_AGGREGATION:
        group_legend = [i for i in name_subplot]
    else:
        group_legend = ["mentions" for _ in name_subplot]

    for poll in PollingOrganizations:
        if poll == PollingOrganizations.ALL:
            continue
        count += 1
        show_legend = True if (count == 1 or aggregation == AggregationMode.NO_AGGREGATION) else False

        df_poll = df[df["nom_institut"] == poll.value].copy() if poll != PollingOrganizations.ALL else df.copy()
        if df_poll.empty:
            continue
        nb_grades = len(get_grades(df_poll))
        colors = color_palette(palette="coolwarm", n_colors=nb_grades)
        color_dict = {f"intention_mention_{i + 1}": f"rgb{str(colors[i])}" for i in range(nb_grades)}

        col_intention = get_intentions_colheaders(df_poll, nb_grades)
        y_cumsum = df_poll[col_intention].to_numpy()
        for g, col, cur_y in zip(get_grades(df_poll), col_intention, y_cumsum.T):
            fig.add_trace(
                go.Scatter(
                    x=df_poll["fin_enquete"],
                    y=cur_y,
                    hoverinfo="x+y",
                    mode="lines",
                    line=dict(width=0.5, color=color_dict[col]),
                    stackgroup="one",  # define stack group
                    name=g,
                    showlegend=show_legend,
                    legendgroup=group_legend[count - 1],
                    legendgrouptitle_text=group_legend[count - 1],
                ),
                row=count,
                col=1,
            )
        show_legend_no_opinion = True if count == 1 else False
        # fig = add_no_opinion_time_merit_profile(
        #     fig, df_poll, suffix, row=count, col=1, show_legend=show_legend_no_opinion
        # )

        for d in df_poll["end_date"]:
            fig.add_vline(
                x=d,
                line_dash="dot",
                line_width=1,
                line_color="black",
                opacity=0.2,
                row=count,
                col=1,
            )

        fig.add_hline(
            y=50,
            line_dash="dot",
            line_width=1,
            line_color="black",
            annotation_text="50 %",
            annotation_position="bottom right",
            row=count,
            col=1,
        )
        fig.update_yaxes(title_text="Mentions (%)", tickfont_size=15, range=[0, 100], row=count, col=1)
        fig.update_xaxes(title_text="Mentions (%)", tickfont_size=15, range=[date_min, date_max], row=count, col=1)
    fig.update_layout(
        width=600,
        height=800,
        plot_bgcolor="white",
    )

    # Title and detailed
    date = df["end_date"].max()
    title = (
        f"<b>Evolution des mentions au jugement majoritaire"
        + f"<br> pour le candidat {df.candidat.unique().tolist()[0]}</b>"
    )
    fig.update_layout(title=title, title_x=0.5)
    fig = _add_image_to_fig(fig, x=1.1, y=0.15, sizex=0.25, sizey=0.25)

    return fig
