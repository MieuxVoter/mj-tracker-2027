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
from .plot_utils import _add_election_date, _generate_windows_size, _add_image_to_fig, load_colors


def plot_merit_profiles(
    df: DataFrame,
    grades: list,
    auto_text: bool = True,
    font_size: int = 20,
    date: str = None,
    sponsor: str = None,
    source: str = None,
    show_no_opinion: bool = True,
) -> go.Figure:
    df = df.copy()

    nb_grades = len(grades)

    # compute the list sorted of candidat names to order y axis.
    candidat_list = list(df["candidate"])
    rank_list = list(df["rang"] - 1)
    sorted_candidat_list = [i[1] for i in sorted(zip(rank_list, candidat_list))]
    r_sorted_candidat_list = sorted_candidat_list.copy()
    r_sorted_candidat_list.reverse()

    colors = color_palette(palette="coolwarm", n_colors=nb_grades)
    color_dict = {f"intention_mention_{i + 1}": f"rgb{str(colors[i])}" for i in range(nb_grades)}
    fig = px.bar(
        df,
        x=get_intentions_colheaders(df, nb_grades),
        y="candidate",
        orientation="h",
        text_auto=auto_text,
        color_discrete_map=color_dict,
    )

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
    has_no_opinion_col = "sans_opinion" in df.columns
    if has_no_opinion_col and show_no_opinion and not np.isnan(df["sans_opinion"].unique()[0]):
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
    fig.update_layout(width=1000, height=900)

    return fig


def plot_merit_profiles_in_number(
    df: DataFrame,
    grades: list,
    auto_text: bool = True,
    font_size: int = 20,
    date: str = None,
    sponsor: str = None,
    source: str = None,
    show_no_opinion: bool = True,
) -> go.Figure:
    df = df.copy()

    nb_grades = len(grades)

    # compute the list sorted of candidat names to order y axis.
    candidat_list = list(df["candidat"])
    rank_list = list(df["rang"] - 1)
    sorted_candidat_list = [i[1] for i in sorted(zip(rank_list, candidat_list))]
    r_sorted_candidat_list = sorted_candidat_list.copy()
    r_sorted_candidat_list.reverse()

    # colors = color_palette(palette="coolwarm", n_colors=nb_grades)
    # Gold, Silver, Bronze, No Medal
    colors_olympics = [(255, 215, 0), (192, 192, 192), (205, 127, 50), (139, 69, 19)]
    colors = colors_olympics
    color_dict = {f"intention_mention_{i + 1}": f"rgb{str(colors[i])}" for i in range(nb_grades)}
    fig = px.bar(
        df,
        x=get_intentions_colheaders(df, nb_grades),
        y="candidat",
        orientation="h",
        text_auto=auto_text,
        color_discrete_map=color_dict,
    )

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
    sum_of_intentions = df[get_intentions_colheaders(df, nb_grades)].sum(axis=1).max()
    fig.add_vline(x=sum_of_intentions / 2, line_width=2, line_color="black")

    # Legend
    fig.update_layout(
        legend_title_text=None,
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width
    )

    fig.update(data=[{"hovertemplate": "Intention: %{x}<br>Candidat: %{y}"}])
    # todo: need to plot grades in hovertemplate.

    # no background
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # xticks and y ticks
    # Add sans opinion to y tick label # todo : it may be simplified !
    if show_no_opinion and not np.isnan(df["sans_opinion"].unique()[0]):
        df["candidat_sans_opinion"] = None
        for ii, cell in enumerate(df["candidat"]):
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
            # range=[0, 100],
            # tickmode="array",
            # tickvals=[0, 20, 40, 60, 80, 100],
            # ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            # tickfont_size=font_size,
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


def ranking_plot(
    df,
    on_rolling_data: bool = False,
    source: str = None,
    sponsor: str = None,
    show_best_grade: bool = True,
    show_rank: bool = True,
    show_no_opinion: bool = True,
    show_grade_area: bool = True,
    breaks_in_names: bool = True,
    fig: go.Figure = None,
    row=None,
    col=None,
) -> go.Figure:
    if on_rolling_data:
        if "rang_glissant" not in df.columns:
            raise ValueError("This dataframe hasn't been smoothed with rolling average.")
        df["rang"] = df["rang_glissant"]
        df["mention_majoritaire"] = df["mention_majoritaire_glissante"]

    COLORS = load_colors()
    if fig is None:
        fig = go.Figure()

    df = df.sort_values(by="end_date")

    # Grade area
    if show_grade_area:
        grades = get_grades(df)
        nb_grades = len(grades)
        c_rgb = color_palette(palette="coolwarm", n_colors=nb_grades)
        for g, c in zip(grades, c_rgb):
            temp_df = df[df["mention_majoritaire"] == g]
            if not temp_df.empty:
                c_alpha = str(f"rgba({c[0]},{c[1]},{c[2]},0.2)")
                x_date = temp_df["end_date"].unique().tolist()
                y_upper = []
                y_lower = []
                for d in x_date:
                    y_upper.append(temp_df[temp_df["end_date"] == d]["rang"].min() - 0.5)
                    y_lower.append(temp_df[temp_df["end_date"] == d]["rang"].max() + 0.5)

                fig.add_scatter(
                    x=x_date + x_date[::-1],  # x, then x reversed
                    y=y_upper + y_lower[::-1],  # upper, then lower reversed
                    fill="toself",
                    fillcolor=c_alpha,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name=g,
                    row=row,
                    col=col,
                )

    for ii in get_candidates(df):

        color = COLORS.get(ii, {"couleur": "black"})["couleur"]

        temp_df = df[df["candidate"] == ii]
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"],
                y=temp_df["rang"],
                mode="lines",
                name=ii,
                marker=dict(color=color),
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"].iloc[0:1],
                y=temp_df["rang"].iloc[0:1],
                mode="markers",
                name=ii,
                marker=dict(color=color),
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"].iloc[-1:],
                y=temp_df["rang"].iloc[-1:],
                mode="markers",
                name=ii,
                marker=dict(color=color),
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

        # PREPARE ANNOTATIONS
        # name with break btw name and surname
        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None
        name_label = _extended_name_annotations(
            temp_df,
            candidate=ii,
            show_rank=False,
            show_best_grade=False,
            show_no_opinion=False,
            breaks_in_names=breaks_in_names,
        )
        size_annotations = 12

        # first dot annotation
        if temp_df["end_date"].iloc[-1] != temp_df["end_date"].iloc[0]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[0],
                    y=temp_df["rang"].iloc[0],
                    xanchor="right",
                    xshift=-10,
                    text=f"{name_label}",
                    font=dict(family="Arial", size=size_annotations, color=color),
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                ),
            )

        # Nice name label
        extended_name_label = _extended_name_annotations(
            temp_df,
            candidate=ii,
            show_rank=show_rank,
            show_best_grade=show_best_grade,
            show_no_opinion=show_no_opinion,
            breaks_in_names=breaks_in_names,
        )

        # last dot annotation
        # only if the last dot is correspond to the last polls
        if df["end_date"].max() == temp_df["end_date"].iloc[-1]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[-1],
                    y=temp_df["rang"].iloc[-1],
                    xanchor="left",
                    xshift=10,
                    yanchor="middle",
                    text=extended_name_label,
                    font=dict(family="Arial", size=size_annotations, color=color),
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                ),
            )

    # fig = _add_election_date(fig, y=0.25, xshift=10)

    fig.update_layout(
        yaxis=dict(autorange="reversed", tick0=1, dtick=1, visible=False),
        # annotations=annotations,
        plot_bgcolor="white",
        showlegend=True,
    )

    source_str = f"source: {source}" if source is not None else ""
    source_str += ", " if sponsor is not None else ""
    sponsor_str = f"commanditaire: {sponsor}" if sponsor is not None else ""

    date = df["end_date"].max()
    title = (
        "<b>Classement des candidats à l'élection présidentielle 2027<br> au jugement majoritaire </b> <br>"
        + f"<i>{source_str}{sponsor_str}, dernier sondage: {date}.</i>"
    )
    fig.update_layout(title=title, title_x=0.5)
    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")
    # SIZE OF THE FIGURE
    fig.update_layout(width=1200, height=1000)

    # Legend
    fig.update_layout(
        legend_title_text="Mentions majoritaires",
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width/
    )
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


def add_no_opinion_time_merit_profile(
    fig: go.Figure, df: DataFrame, suffix: str, show_legend: bool = True, row: int = None, col: int = None
) -> go.Figure:
    sub_df = df[["end_date", f"sans_opinion{suffix}"]]
    sub_df = sub_df.sort_values(by="end_date").dropna()
    # sub_df = sub_df[df[f"sans_opinion{suffix}"] is not None]
    fig.add_trace(
        go.Scatter(
            x=sub_df["fin_enquete"],
            y=sub_df[f"sans_opinion{suffix}"] / 2 + 50,
            hoverinfo="x+y",
            mode="lines",
            line=dict(width=2, color="black", dash="dash"),
            name="sans opinion",
            opacity=0.5,
            showlegend=show_legend,
            legendgroup="sans opinion",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=sub_df["fin_enquete"],
            y=50 - sub_df[f"sans_opinion{suffix}"] / 2,
            hoverinfo="x+y",
            mode="lines",
            line=dict(width=2, color="black", dash="dash"),
            name="sans opinion",
            opacity=0.5,
            showlegend=False,
            legendgroup="sans opinion",
        ),
        row=row,
        col=col,
    )
    return fig
