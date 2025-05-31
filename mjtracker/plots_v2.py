import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from seaborn import color_palette
import numpy as np
import pandas as pd
from pandas import DataFrame

from . import SurveyInterface
from . import SurveysInterface
from .utils import get_intentions_colheaders, get_candidates, get_grades, rank2str
from .misc.enums import PollingOrganizations, AggregationMode
from .color_utils import get_grade_color_palette
from .plot_utils import load_colors, export_fig, _extended_name_annotations, _add_image_to_fig, _generate_windows_size


def plot_merit_profiles(
    si: SurveyInterface,
    auto_text: bool = True,
    font_size: int = 20,
    show_no_opinion: bool = True,
) -> go.Figure:

    # colors = color_palette(palette="coolwarm", n_colors=si.nb_grades)

    colors = get_grade_color_palette(si.nb_grades)
    colors.reverse()

    color_dict = {f"intention_mention_{i + 1}": f"rgb{str(colors[i])}" for i in range(si.nb_grades)}
    fig = px.bar(
        si.df.copy().sort_values(by="rang", ascending=False, na_position="last"),
        x=si._intentions_colheaders,
        y="candidate",
        orientation="h",
        text_auto=auto_text,
        color_discrete_map=color_dict,
    )

    fig.update_traces(textfont_size=font_size, textangle=0, textposition="auto", cliponaxis=False, width=0.5)

    # replace variable names with grades
    grades = si.grades
    new_names = {f"intention_mention_{i + 1}": grades[i] for i in range(si.nb_grades)}
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
    # todo: need to display grades in hovertemplate.

    # no background
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # xticks and y ticks
    yticktext = si.formated_ranked_candidates(show_no_opinion)
    yticktext.reverse()
    ycategoryarray = si.ranked_candidates
    ycategoryarray.reverse()
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            tickmode="array",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            tickfont_size=font_size,
            title="",
        ),
        yaxis=dict(
            tickfont_size=font_size * 0.75,
            title="",
            automargin=True,
            ticklabelposition="outside left",
            ticksuffix="   ",
            tickmode="array",
            tickvals=[i for i in range(si.nb_candidates)],
            ticktext=yticktext,
            categoryorder="array",
            categoryarray=ycategoryarray,
        ),
    )

    # Title
    title = "<b>Evaluation au jugement majoritaire</b>"

    date_str = f"date: {si.end_date}, " if si.end_date is not None else ""
    source_str = f"source: {si.source}" if si.source is not None else ""
    source_str += ", " if si.sponsor is not None else ""
    sponsor_str = f"commanditaire: {si.sponsor}" if si.sponsor is not None else ""
    subtitle = f"<br><i>{date_str}{source_str}{sponsor_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    # font family
    fig.update_layout(font_family="arial")

    fig = _add_image_to_fig(fig, x=0.9, y=1.01, sizex=0.15, sizey=0.15)

    # size of the figure
    fig.update_layout(width=1000, height=900)

    return fig


def plot_merit_profiles_in_number(
    si: SurveyInterface,
    auto_text: bool = True,
    font_size: int = 20,
    show_no_opinion: bool = True,
) -> go.Figure:

    # TODO: GENERALIZED VERSION OF THE PREVIOUS ONE,
    #  TRY OUT IF IT COULD FIT ALWAYS EVEN IF THE TOTAL INTENTION IS NOT 100.

    # colors = color_palette(palette="coolwarm", n_colors=si.nb_grades)
    colors = get_grade_color_palette(si.nb_grades)
    colors.reverse()

    color_dict = {f"intention_mention_{i + 1}": f"rgb{str(colors[i])}" for i in range(si.nb_grades)}
    fig = px.bar(
        si.df.copy().sort_values(by="rang", ascending=False, na_position="last"),
        x=si._intentions_colheaders,
        y="candidate",
        orientation="h",
        text_auto=auto_text,
        color_discrete_map=color_dict,
    )

    fig.update_traces(textfont_size=font_size, textangle=0, textposition="auto", cliponaxis=False, width=0.5)

    # replace variable names with grades
    grades = si.grades
    new_names = {f"intention_mention_{i + 1}": grades[i] for i in range(si.nb_grades)}
    fig.for_each_trace(
        lambda t: t.update(
            name=new_names[t.name],
            legendgroup=new_names[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_names[t.name]),
        )
    )

    # vertical line
    sum_of_intentions = si.df[si._intentions_colheaders].sum(axis=1).max()
    fig.add_vline(x=sum_of_intentions / 2, line_width=2, line_color="black")

    # Legend
    fig.update_layout(
        legend_title_text=None,
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width
    )

    fig.update(data=[{"hovertemplate": "Intention: %{x}<br>Candidat: %{y}"}])

    # no background
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # xticks and y ticks
    yticktext = si.formated_ranked_candidates(show_no_opinion)
    yticktext.reverse()
    ycategoryarray = si.ranked_candidates
    ycategoryarray.reverse()
    fig.update_layout(
        xaxis=dict(
            # range=[0, 100],
            # tickmode="array",
            # tickvals=[0, 20, 40, 60, 80, 100],
            # ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            # tickfont_size=font_size,
            title="",
        ),
        yaxis=dict(
            tickfont_size=font_size * 0.75,
            title="",
            automargin=True,
            ticklabelposition="outside left",
            ticksuffix="   ",
            tickmode="array",
            tickvals=[i for i in range(si.nb_candidates)],
            ticktext=yticktext,
            categoryorder="array",
            categoryarray=ycategoryarray,
        ),
    )

    # Title
    title = "<b>Evaluation au jugement majoritaire</b>"

    date_str = f"date: {si.end_date}, " if si.end_date is not None else ""
    source_str = f"source: {si.source}" if si.source is not None else ""
    source_str += ", " if si.sponsor is not None else ""
    sponsor_str = f"commanditaire: {si.sponsor}" if si.sponsor is not None else ""
    subtitle = f"<br><i>{date_str}{source_str}{sponsor_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    # font family
    fig.update_layout(font_family="arial")

    fig = _add_image_to_fig(fig, x=0.9, y=1.01, sizex=0.15, sizey=0.15)

    # size of the figure
    fig.update_layout(width=1000, height=600)

    return fig


def ranking_plot(
    si: SurveysInterface,
    on_rolling_data: bool = False,
    show_best_grade: bool = True,
    show_rank: bool = True,
    show_no_opinion: bool = True,
    show_grade_area: bool = True,
    breaks_in_names: bool = True,
    fig: go.Figure = None,
    row=None,
    col=None,
) -> go.Figure:

    COLORS = load_colors()
    TRANSPARENCY = 0.5
    AREA_EDGE_OFFSET = 0.42

    if fig is None:
        fig = go.Figure()

    if not si.is_aggregated:
        raise ValueError("The ranking plot requires the data to be aggregated into a unique set of grades.")

    df = si.df.copy().sort_values(by="end_date")

    # Grade area
    if show_grade_area:
        grades = si.grades
        # c_rgb = color_palette(palette="coolwarm", n_colors=si.nb_grades)

        c_rgb = get_grade_color_palette(si.nb_grades)
        c_rgb.reverse()

        x_date = si.dates

        for grade, color in zip(grades, c_rgb):
            temp_df = df[df["mention_majoritaire"] == grade]
            if temp_df.empty:
                fig.add_scatter(
                    x=[x_date[0], x_date[-1], x_date[-1], x_date[0]],
                    y=[0, 0, 0, 0],
                    fill="toself",
                    fillcolor=str(f"rgba({color[0]},{color[1]},{color[2]},{TRANSPARENCY})"),
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name=grade,
                    row=row,
                    col=col,
                )
                continue

            y_upper = []
            y_lower = []
            for d in x_date:
                y_upper.append(temp_df[temp_df["end_date"] == d]["rang"].min() - AREA_EDGE_OFFSET)
                y_lower.append(temp_df[temp_df["end_date"] == d]["rang"].max() + AREA_EDGE_OFFSET)

            fig.add_scatter(
                x=x_date + x_date[::-1],  # x, then x reversed
                y=y_upper + y_lower[::-1],  # upper, then lower reversed
                fill="toself",
                fillcolor=str(f"rgba({color[0]},{color[1]},{color[2]},{TRANSPARENCY})"),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=grade,
                row=row,
                col=col,
            )

    for candidate in si.candidates:
        color = COLORS.get(candidate, {"couleur": "black"})["couleur"]

        temp_df = si.select_candidate(candidate).df.copy().sort_values(by="end_date")
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"],
                y=temp_df["rang"],
                mode="lines",
                name=candidate,
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
                name=candidate,
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
                name=candidate,
                marker=dict(color=color),
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

        # PREPARE ANNOTATIONS - name with break btw name and surname
        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None
        name_label = _extended_name_annotations(
            temp_df,
            candidate=candidate,
            show_rank=False,
            show_best_grade=False,
            show_no_opinion=False,
            breaks_in_names=breaks_in_names,
        )
        size_annotations = 12
        name_shift = 10

        # first dot annotation
        if temp_df["end_date"].iloc[-1] != temp_df["end_date"].iloc[0]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[0],
                    y=temp_df["rang"].iloc[0],
                    xanchor="right",
                    xshift=-name_shift,
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
            candidate=candidate,
            show_rank=show_rank,
            show_best_grade=show_best_grade,
            show_no_opinion=show_no_opinion,
            breaks_in_names=breaks_in_names,
        )

        # last dot annotation
        # only if the last dot correspond to the last polls
        if df["end_date"].max() == temp_df["end_date"].iloc[-1]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[-1],
                    y=temp_df["rang"].iloc[-1],
                    xanchor="left",
                    xshift=name_shift,
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

    # Title
    title = "<b>Classement des candidats à l'élection présidentielle 2027<br> au jugement majoritaire</b> "

    end_date = df["end_date"].max()
    date_str = f"date: {end_date}, " if end_date is not None else ""
    source_str = f"source: {si.sources_string}" if si.sources_string is not None else ""
    source_str += ", " if si.sponsors_string is not None else ""
    sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors_string is not None else ""
    subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {date_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    # Legend
    fig.update_layout(
        width=1200,
        height=1000,
        legend_title_text="Mentions majoritaires",
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width/
    )
    return fig


def plot_time_merit_profile(
    si: SurveyInterface,
    fig: go.Figure = None,
    show_legend: bool = True,
    show_logo: bool = True,
    no_layout: bool = False,
    row: int = None,
    col: int = None,
) -> go.Figure:
    # todo: reverse legend -> from positive to negative

    if fig is None:
        fig = go.Figure()

    si.df.sort_values(by="end_date")

    # colors = color_palette(palette="coolwarm", n_colors=si.nb_grades)

    colors = get_grade_color_palette(si.nb_grades)
    colors.reverse()

    color_dict = {f"{i}": f"rgb{str(colors[i])}" for i in range(si.nb_grades)}

    for i, (cur_y, grade) in enumerate(zip(si.intentions.T, si.grades)):
        fig.add_trace(
            go.Scatter(
                x=si.dates,
                y=cur_y,
                hoverinfo="x+y",
                mode="lines",
                line=dict(width=0.5, color=color_dict[f"{i}"]),
                stackgroup="one",
                fillcolor=color_dict[f"{i}"],
                name=grade,
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

    for d in si.dates:
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
            legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width
            yaxis=dict(
                tickfont_size=15,
                title="Mentions (%)",
                automargin=True,
            ),
            plot_bgcolor="white",
        )

        # Title and detailed
        title = (
            f"<b>Evolution des mentions au jugement majoritaire"
            + f"<br> pour le candidat {si.df.candidate.unique().tolist()[0]}</b>"
        )
        source_str = f"source: {si.sources_string}" if si.sources is not None else ""
        source_str += ", " if si.sponsors is not None else ""
        sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors is not None else ""
        subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {si.most_recent_dates}.</i>"

        fig.update_layout(title=title + subtitle, title_x=0.5)

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
