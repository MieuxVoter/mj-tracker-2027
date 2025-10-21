import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from seaborn import color_palette
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import timedelta, datetime

from ..core import SurveyInterface
from ..core import SurveysInterface
from ..utils.utils import get_intentions_colheaders, get_candidates, get_grades, rank2str
from ..misc.enums import PollingOrganizations, AggregationMode
from .color_utils import get_grade_color_palette
from .plot_utils import load_colors, export_fig, _extended_name_annotations, _add_image_to_fig, _generate_windows_size

LOGO_X_LOCATION = 0.88
LOGO_Y_LOCATION = 0.965


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
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.03),  # 50 % of the figure width
    )

    fig.update(data=[{"hovertemplate": "Intention: %{x}<br>Candidat: %{y}"}])
    # todo: need to display grades in hovertemplate.

    # no background
    fig.update_layout(paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(255,255,255,1)")

    # xticks and y ticks
    yticktext = si.formated_ranked_candidates(show_no_opinion)
    yticktext.reverse()
    ycategoryarray = si.ranked_candidates
    ycategoryarray.reverse()
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            tickmode="array",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
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

    fig = _add_image_to_fig(fig, x=LOGO_X_LOCATION, y=LOGO_Y_LOCATION, sizex=0.15, sizey=0.15)

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


def plot_approval_profiles(
    si: SurveyInterface,
    auto_text: bool = True,
    font_size: int = 20,
    show_no_opinion: bool = True,
) -> go.Figure:

    color_rgb = (36, 0, 253)

    color_dict = {"approbation": f"rgb{str(color_rgb)}"}
    fig = px.bar(
        si.df.copy().sort_values(by="rang", ascending=False, na_position="last"),
        x="approbation",
        y="candidate",
        orientation="h",
        text_auto=False,
        color_discrete_map=color_dict,
    )

    fig.update_traces(textfont_size=font_size, textangle=0, textposition="auto", cliponaxis=False, width=0.5)

    # Legend
    fig.update_layout(
        legend_title_text=None,
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width
    )

    fig.update(data=[{"hovertemplate": "Approbation: %{x}<br>Candidat: %{y}"}])

    # no background
    fig.update_layout(paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(255,255,255,1)")

    # xticks and y ticks
    yticktext = si.formated_ranked_candidates(show_no_opinion)
    yticktext.reverse()
    ycategoryarray = si.ranked_candidates
    ycategoryarray.reverse()
    fig.update_layout(
        xaxis=dict(
            range=[0, 101],
            tickmode="array",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            tickfont_size=font_size,
            title="",
            gridcolor="black",
            gridwidth=1,
            griddash="solid",
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
    title = "<b>Evaluation à l'approbation</b>"

    date_str = f"date: {si.end_date}, " if si.end_date is not None else ""
    source_str = f"source: {si.source}" if si.source is not None else ""
    source_str += ", " if si.sponsor is not None else ""
    sponsor_str = f"commanditaire: {si.sponsor}" if si.sponsor is not None else ""
    subtitle = f"<br><i>{date_str}{source_str}{sponsor_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    # font family
    fig.update_layout(font_family="arial")

    fig = _add_image_to_fig(fig, x=LOGO_X_LOCATION, y=LOGO_Y_LOCATION, sizex=0.15, sizey=0.15)

    # size of the figure
    fig.update_layout(width=1000, height=900)

    return fig


def ranking_plot(
    si: SurveysInterface,
    on_rolling_data: bool = False,
    show_best_grade: bool = True,
    show_rank: bool = True,
    show_no_opinion: bool = True,
    show_grade_area: bool = True,
    breaks_in_names: bool = True,
    voting_str_title: str = "au jugement majoritaire",
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
                continue
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

    margin_one_day = timedelta(days=1)

    # Convert string dates to datetime objects
    start_date = datetime.strptime(si.dates[0], "%Y-%m-%d")
    end_date = datetime.strptime(si.dates[-1], "%Y-%m-%d")

    x_range = [start_date - margin_one_day, end_date + 5 * margin_one_day]

    fig.update_layout(
        yaxis=dict(autorange="reversed", tick0=1, dtick=1, visible=False),
        xaxis=dict(range=x_range, tickformat="%m/%Y"),
        # annotations=annotations,
        plot_bgcolor="white",
        showlegend=True,
    )

    # Title
    title = "<b>Classement des candidats à l'élection présidentielle 2027<br>" f"{voting_str_title}</b> "

    end_date = df["end_date"].max()
    date_str = f"date: {end_date}, " if end_date is not None else ""
    source_str = f"source: {si.sources_string}" if si.sources_string is not None else ""
    source_str += ", " if si.sponsors_string is not None else ""
    sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors_string is not None else ""
    subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {date_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    fig = _add_image_to_fig(fig, x=1.00, y=0.985, sizex=0.10, sizey=0.10, xanchor="right")

    # Legend
    fig.update_layout(
        width=1200,
        height=1000,
        legend_title_text="Mentions majoritaires",
        # autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.025),  # 50 % of the figure width/
    )
    return fig


def ranking_plot_constant_area(
    si: "SurveysInterface",
    on_rolling_data: bool = False,
    show_best_grade: bool = True,
    show_rank: bool = True,
    show_no_opinion: bool = True,
    show_grade_area: bool = True,
    breaks_in_names: bool = True,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Return a Plotly figure where each *mention majoritaire* band keeps a **constant
    vertical extent** (1 unit). For every survey date, the candidates that share the
    same grade are evenly spaced inside the corresponding band, meaning that their
    trajectories compress whenever many candidates share the grade and expand when
    the band contains only a few of them.

    The signature is deliberately identical to ``ranking_plot`` so that you can use
    it as a drop‑in replacement.
    """

    # ------------------------------------------------------------------
    # Constants & sanity checks
    # ------------------------------------------------------------------
    COLORS = load_colors()
    TRANSPARENCY = 0.5
    GRADE_HEIGHT = 1.0  # constant height for every grade band
    INNER_MARGIN = 0.10 * GRADE_HEIGHT  # free space at top/bottom of each band

    if fig is None:
        fig = go.Figure()

    if not si.is_aggregated:
        raise ValueError("The ranking plot requires the data to be aggregated into a unique set of grades.")

    # ------------------------------------------------------------------
    # 1 – Build a working DataFrame and compute the y‑coordinate to plot
    # ------------------------------------------------------------------
    df = si.df.copy().sort_values(by=["end_date", "rang"])
    grades = list(si.grades)  # best → worst
    grade_to_index = {g: i for i, g in enumerate(grades)}  # 0 == top band

    df["grade_index"] = df["mention_majoritaire"].map(grade_to_index)
    df["y_pos"] = np.nan

    # For each poll date, evenly space candidates inside their band
    for poll_date, date_group in df.groupby("end_date"):
        for grade, grade_group in date_group.groupby("mention_majoritaire"):
            band_idx = grade_to_index[grade]
            k = len(grade_group)

            band_top = band_idx + INNER_MARGIN
            band_bottom = (band_idx + GRADE_HEIGHT) - INNER_MARGIN

            if k == 1:
                y_vals = [0.5 * (band_top + band_bottom)]
            else:
                # Upper position = best (smallest original rank)
                y_vals = np.linspace(band_top, band_bottom, k)

            # Preserve original ordering by rank so that the best sits on top
            ordered_idx = grade_group.sort_values("rang").index
            df.loc[ordered_idx, "y_pos"] = y_vals

    # ------------------------------------------------------------------
    # 2 – Draw fixed grade areas (simple 1‑unit rectangles)
    # ------------------------------------------------------------------
    if show_grade_area:
        x_dates = si.dates
        x_left, x_right = x_dates[0], x_dates[-1]

        colors_rgb = get_grade_color_palette(si.nb_grades)
        colors_rgb.reverse()  # keep same color ordering as original plot

        for grade, rgb in zip(grades, colors_rgb):
            idx = grade_to_index[grade]
            y0, y1 = idx, idx + GRADE_HEIGHT

            fig.add_scatter(
                x=[x_left, x_right, x_right, x_left],
                y=[y0, y0, y1, y1],
                fill="toself",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{TRANSPARENCY})",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=grade,
                row=row,
                col=col,
            )

    # ------------------------------------------------------------------
    # 3 – Plot candidate trajectories using the rescaled y‑coordinate
    # ------------------------------------------------------------------
    for candidate in si.candidates:
        color = COLORS.get(candidate, {"couleur": "black"})["couleur"]

        # The helper keeps the candidate name column consistent across SI versions
        temp_df = df[df.get("nom", df.get("candidate", "")) == candidate].copy().sort_values("end_date")
        if temp_df.empty:  # fall‑back when the column name is unknown
            temp_df = si.select_candidate(candidate).df.copy().sort_values("end_date")
            temp_df = temp_df.merge(df[["end_date", "y_pos"]], on="end_date", how="left")

        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"],
                y=temp_df["y_pos"],
                mode="lines",
                name=candidate,
                marker=dict(color=color),
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

        # Start & end markers -------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"].iloc[[0]],
                y=temp_df["y_pos"].iloc[[0]],
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
                x=temp_df["end_date"].iloc[[-1]],
                y=temp_df["y_pos"].iloc[[-1]],
                mode="markers",
                name=candidate,
                marker=dict(color=color),
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

        # ------------------------------------------------------------------
        # 4 – Text annotations
        # ------------------------------------------------------------------
        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None

        bare_name_label = _extended_name_annotations(
            temp_df,
            candidate=candidate,
            show_rank=False,
            show_best_grade=False,
            show_no_opinion=False,
            breaks_in_names=breaks_in_names,
        )
        full_name_label = _extended_name_annotations(
            temp_df,
            candidate=candidate,
            show_rank=show_rank,
            show_best_grade=show_best_grade,
            show_no_opinion=show_no_opinion,
            breaks_in_names=breaks_in_names,
        )

        size_annotations = 12
        name_shift = 10

        if temp_df["end_date"].iloc[-1] != temp_df["end_date"].iloc[0]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[0],
                    y=temp_df["y_pos"].iloc[0],
                    xanchor="right",
                    xshift=-name_shift,
                    text=bare_name_label,
                    font=dict(family="Arial", size=size_annotations, color=color),
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                ),
            )

        if df["end_date"].max() == temp_df["end_date"].iloc[-1]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[-1],
                    y=temp_df["y_pos"].iloc[-1],
                    xanchor="left",
                    xshift=name_shift,
                    yanchor="middle",
                    text=full_name_label,
                    font=dict(family="Arial", size=size_annotations, color=color),
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                ),
            )

    # ------------------------------------------------------------------
    # 5 – Layout, title, legend, etc.
    # ------------------------------------------------------------------
    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            tick0=0,
            dtick=GRADE_HEIGHT,
            range=[0, len(grades)],
            visible=False,
        ),
        plot_bgcolor="white",
        showlegend=True,
    )

    # Title ------------------------------------------------------------------
    title = "<b>Classement des candidats à l'élection présidentielle 2027" "<br> au jugement majoritaire</b> "
    end_date = df["end_date"].max()
    date_str = f"date: {end_date}, " if end_date is not None else ""
    source_str = f"source: {si.sources_string}" if si.sources_string else ""
    source_str += ", " if si.sponsors_string else ""
    sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors_string else ""
    subtitle = f"<br><i>{source_str}{sponsor_str}dernier sondage: {date_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    fig.update_layout(
        width=1200,
        height=1000,
        legend_title_text="Mentions majoritaires",
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),
    )

    return fig


def ranking_plot_variable_band_height(
    si: "SurveysInterface",
    on_rolling_data: bool = False,
    show_best_grade: bool = True,
    show_rank: bool = True,
    show_no_opinion: bool = True,
    show_grade_area: bool = True,
    breaks_in_names: bool = True,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot candidates' trajectories with fixed *area* bands whose heights are
    proportional to the **average** number of candidates they host across all
    surveys.

    * Inside every band, candidates are evenly spaced **per survey date**, so the
      curves compress when many contenders share a grade and relax when they are
      few.
    * Replace your original ``ranking_plot`` with this function for adaptive yet
      comparable grade areas.
    """

    # ------------------------------------------------------------------
    # Constants & helpers
    # ------------------------------------------------------------------
    COLORS = load_colors()
    TRANSPARENCY = 0.5
    INNER_MARGIN_RATIO = 0.10  # 10 % margin inside each band (top & bottom)
    MIN_HEIGHT = 0.35  # safety height if a grade would receive ~0 cand.

    if fig is None:
        fig = go.Figure()

    if not si.is_aggregated:
        raise ValueError("The ranking plot requires the data to be aggregated into a unique set of grades.")

    # ------------------------------------------------------------------
    # 1 – Prepare DataFrame & compute average counts per grade
    # ------------------------------------------------------------------
    df = si.df.copy().sort_values(by=["end_date", "rang"])
    grades = list(si.grades)  # best → worst
    grade_to_idx = {g: i for i, g in enumerate(grades)}

    # Count candidates per date & grade, then take the mean over dates
    counts = df.groupby(["end_date", "mention_majoritaire"]).size().unstack(fill_value=0)
    mean_counts = counts.mean(axis=0).reindex(grades).fillna(0)

    # Scale heights so that the *sum* of band heights equals ``len(grades)``
    scale = len(grades) / mean_counts.sum() if mean_counts.sum() else 1.0
    heights = (mean_counts * scale).clip(lower=MIN_HEIGHT)

    # Re‑compute the scale if total height changed after clipping
    total_height = heights.sum()
    heights *= len(grades) / total_height

    # Build cumulative bounds for every grade  ---------------------------
    bounds = {}
    cum_y = 0.0
    for g in grades:
        h = heights[g]
        bounds[g] = (cum_y, cum_y + h)
        cum_y += h

    # Cache the total vertical span for axis limits later
    AXIS_HEIGHT = cum_y

    # ------------------------------------------------------------------
    # 2 – Compute y‑coordinate (compressed) for every candidate & date
    # ------------------------------------------------------------------
    df["y_pos"] = np.nan

    for poll_date, date_group in df.groupby("end_date"):
        for grade, grade_group in date_group.groupby("mention_majoritaire"):
            y0, y1 = bounds[grade]
            band_height = y1 - y0
            inner_margin = INNER_MARGIN_RATIO * band_height
            inner_top = y0 + inner_margin
            inner_bottom = y1 - inner_margin

            k = len(grade_group)
            if k == 1:
                y_values = [0.5 * (inner_top + inner_bottom)]
            else:
                # Best rank (lowest "rang") should sit on *inner_top*
                y_values = np.linspace(inner_top, inner_bottom, k)

            ordered_idx = grade_group.sort_values("rang").index
            df.loc[ordered_idx, "y_pos"] = y_values

    # ------------------------------------------------------------------
    # 3 – Draw grade rectangles (scaled height) if requested
    # ------------------------------------------------------------------
    if show_grade_area:
        colors_rgb = get_grade_color_palette(si.nb_grades)
        colors_rgb.reverse()
        x_left, x_right = si.dates[0], si.dates[-1]

        for grade, rgb in zip(grades, colors_rgb):
            y0, y1 = bounds[grade]
            fig.add_scatter(
                x=[x_left, x_right, x_right, x_left],
                y=[y0, y0, y1, y1],
                fill="toself",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{TRANSPARENCY})",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=grade,
                row=row,
                col=col,
            )

    # ------------------------------------------------------------------
    # 4 – Plot candidates
    # ------------------------------------------------------------------
    for candidate in si.candidates:
        color = COLORS.get(candidate, {"couleur": "black"})["couleur"]

        temp_df = df[df.get("nom", df.get("candidate", "")) == candidate]
        if temp_df.empty:
            temp_df = (
                si.select_candidate(candidate).df.copy().merge(df[["end_date", "y_pos"]], on="end_date", how="left")
            )
        temp_df = temp_df.sort_values("end_date")

        # Lines -----------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"],
                y=temp_df["y_pos"],
                mode="lines",
                name=candidate,
                marker=dict(color=color),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Start & end dots ----------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"].iloc[[0]],
                y=temp_df["y_pos"].iloc[[0]],
                mode="markers",
                marker=dict(color=color),
                name=candidate,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"].iloc[[-1]],
                y=temp_df["y_pos"].iloc[[-1]],
                mode="markers",
                marker=dict(color=color),
                name=candidate,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # ----------------------------------------------------------------
        # 5 – Text annotations
        # ----------------------------------------------------------------
        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None
        size_annotations, name_shift = 12, 10

        bare_label = _extended_name_annotations(
            temp_df,
            candidate=candidate,
            show_rank=False,
            show_best_grade=False,
            show_no_opinion=False,
            breaks_in_names=breaks_in_names,
        )
        full_label = _extended_name_annotations(
            temp_df,
            candidate=candidate,
            show_rank=show_rank,
            show_best_grade=show_best_grade,
            show_no_opinion=show_no_opinion,
            breaks_in_names=breaks_in_names,
        )

        if temp_df["end_date"].iloc[-1] != temp_df["end_date"].iloc[0]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[0],
                    y=temp_df["y_pos"].iloc[0],
                    xanchor="right",
                    xshift=-name_shift,
                    text=bare_label,
                    font=dict(family="Arial", size=size_annotations, color=color),
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                ),
            )

        if df["end_date"].max() == temp_df["end_date"].iloc[-1]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[-1],
                    y=temp_df["y_pos"].iloc[-1],
                    xanchor="left",
                    xshift=name_shift,
                    yanchor="middle",
                    text=full_label,
                    font=dict(family="Arial", size=size_annotations, color=color),
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                ),
            )

    # ------------------------------------------------------------------
    # 6 – Layout & cosmetics
    # ------------------------------------------------------------------
    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            tick0=0,
            dtick=1,  # logical tick every "unit" (can be hidden)
            range=[0, AXIS_HEIGHT],
            visible=False,
        ),
        plot_bgcolor="white",
        showlegend=True,
    )

    title = "<b>Classement des candidats à l'élection présidentielle 2027<br> au jugement majoritaire</b> "
    end_date = df["end_date"].max()
    date_str = f"date: {end_date}, " if end_date is not None else ""
    source_str = f"source: {si.sources_string}" if si.sources_string else ""
    source_str += ", " if si.sponsors_string else ""
    sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors_string else ""
    subtitle = f"<br><i>{source_str}{sponsor_str}dernier sondage: {date_str}</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)

    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    fig.update_layout(
        width=1200,
        height=1000,
        legend_title_text="Mentions majoritaires",
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),
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
        subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {si.most_recent_date}.</i>"

        fig.update_layout(title=title + subtitle, title_x=0.5)

        if show_logo:
            fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    return fig


def plot_ranked_time_merit_profile(
    si: SurveysInterface,
    show_no_opinion: bool = True,
    on_rolling_data: bool = False,
) -> go.Figure:
    # Candidat list sorted the rank in the last poll
    si_most_recent = si.most_recent_survey
    si_most_recent.df = si_most_recent.df.sort_values(by="rang")
    titles_candidates = [f"{c} {rank2str(i+1)}" for i, c in enumerate(si_most_recent.candidates)]

    # size of the figure
    n_rows, n_cols = _generate_windows_size(len(si_most_recent.candidates))
    idx_rows, idx_cols = np.unravel_index([i for i in range(si_most_recent.nb_candidates)], (n_rows, n_cols))
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
    for row, col, c in zip(idx_rows, idx_cols, si_most_recent.candidates):
        fig = plot_time_merit_profile(
            si=si.select_candidate(c),
            fig=fig,
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

    # Title
    pop_str = " - population: " + str(si.df["population"].iloc[0]) if "population" in si.df.columns else ""
    title = f"<b>Classement des candidats au jugement majoritaire</b>{pop_str}"

    source_str = f"source: {si.sources_string}" if si.sources_string is not None else ""
    source_str += ", " if si.sponsors_string is not None else ""
    sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors_string is not None else ""
    subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {si.most_recent_date}.</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)
    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    return fig


def plot_time_merit_profile_all_polls(si: SurveysInterface, aggregation: AggregationMode) -> go.Figure:
    name_subplot = tuple([poll.value for poll in PollingOrganizations if poll != PollingOrganizations.ALL])
    fig = make_subplots(rows=len(name_subplot), cols=1, subplot_titles=name_subplot)

    count = 0
    date_max = si.most_recent_date
    date_min = si.oldest_date

    if aggregation == AggregationMode.NO_AGGREGATION:
        group_legend = [i for i in name_subplot]
    else:
        group_legend = ["mentions" for _ in name_subplot]

    for poll in PollingOrganizations:
        if poll == PollingOrganizations.ALL:
            continue
        count += 1
        show_legend = True if (count == 1 or aggregation == AggregationMode.NO_AGGREGATION) else False

        si_poll = si.select_polling_organization(poll)
        if si_poll.df.empty:
            continue

        colors = color_palette(palette="coolwarm", n_colors=si_poll.nb_grades)
        color_dict = {f"{i}": f"rgb{str(colors[i])}" for i in range(si_poll.nb_grades)}
        y_cumsum = si_poll.intentions
        for i, (g, cur_y) in enumerate(zip(si_poll.grades, y_cumsum.T)):
            fig.add_trace(
                go.Scatter(
                    x=si_poll.dates,
                    y=cur_y,
                    hoverinfo="x+y",
                    mode="lines",
                    line=dict(width=0.5, color=color_dict[f"{i}"]),
                    stackgroup="one",  # define stack group
                    name=g,
                    showlegend=show_legend,
                    legendgroup=group_legend[count - 1],
                    legendgrouptitle_text=group_legend[count - 1],
                ),
                row=count,
                col=1,
            )

        # I don't remember what this is.
        show_legend_no_opinion = True if count == 1 else False
        # fig = add_no_opinion_time_merit_profile(
        #     fig, df_poll, suffix, row=count, col=1, show_legend=show_legend_no_opinion
        # )

        for d in si_poll.dates:
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
    title = f"<b>Evolution des mentions au jugement majoritaire<br>pour le candidat {si.df.candidate.unique().tolist()[0]}</b>"
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


def plot_time_approval_profiles(
    si: SurveysInterface,
    on_rolling_data: bool = False,
    show_rank: bool = True,
    show_no_opinion: bool = True,
    breaks_in_names: bool = True,
    fig: go.Figure = None,
    row=None,
    col=None,
) -> go.Figure:

    COLORS = load_colors()

    if fig is None:
        fig = go.Figure()

    if not si.is_aggregated:
        raise ValueError("The ranking plot requires the data to be aggregated into a unique set of grades.")

    df = si.df.copy().sort_values(by=["end_date", "rang"])

    df_with_offsets = add_vertical_offset(df, "candidate", "end_date", "approbation")

    for candidate in si.candidates:
        color = COLORS.get(candidate, {"couleur": "black"})["couleur"]

        temp_df = df_with_offsets[df_with_offsets["candidate"] == candidate].copy().sort_values(by="end_date")
        fig.add_trace(
            go.Scatter(
                x=temp_df["end_date"],
                y=temp_df["approbation"],
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
                y=temp_df["approbation"].iloc[0:1],
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
                y=temp_df["approbation"].iloc[-1:],
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
            breaks_in_names=False,
        )
        size_annotations = 12
        name_shift = 10

        # first dot annotation
        if temp_df["end_date"].iloc[-1] != temp_df["end_date"].iloc[0]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[0],
                    y=temp_df["approbation"].iloc[0] + temp_df["y_offset"].iloc[0],
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
            show_no_opinion=show_no_opinion,
            breaks_in_names=False,
        )

        # last dot annotation
        # only if the last dot correspond to the last polls
        if df["end_date"].max() == temp_df["end_date"].iloc[-1]:
            fig["layout"]["annotations"] += (
                dict(
                    x=temp_df["end_date"].iloc[-1],
                    y=temp_df["approbation"].iloc[-1] + temp_df["y_offset"].iloc[-1],
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
        # yaxis=dict(autorange="reversed", tick0=1, dtick=1, visible=False),
        # annotations=annotations,
        # plot_bgcolor="white",
        showlegend=True,
        yaxis=dict(
            range=[
                # min - 1,5, # max + 1.5
                df_with_offsets["approbation"].min() - 0.5,
                df_with_offsets["approbation"].max() + 0.5,
            ],
        ),
    )

    # Title
    title = "<b>Classement des candidats à l'élection présidentielle 2027<br> à l'approbation</b> "

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
        height=1200,
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width/
    )
    return fig


def add_vertical_offset(df, candidate_column, date_column, value_column, offset=1.5):
    # Créer une copie du DataFrame avec les colonnes pertinentes
    result_df = df.copy()

    # Ajouter une colonne pour stocker le décalage
    result_df["y_offset"] = 0.0

    # Grouper par date
    for date, group in df.groupby(date_column):
        # Créer un dictionnaire pour stocker les positions utilisées à cette date
        used_positions = {}

        # Trier par valeur pour traiter d'abord les candidats avec des valeurs similaires
        for _, row in group.sort_values(value_column).iterrows():
            value = row[value_column]
            candidate = row[candidate_column]

            # Arrondir la valeur pour regrouper les positions proches
            rounded_value = round(value, 1)

            # Vérifier si cette position est déjà utilisée
            if rounded_value in used_positions:
                # Ajouter un décalage
                used_positions[rounded_value] += 0.25
                y_offset = (used_positions[rounded_value] - 1) * offset
                result_df.loc[
                    (result_df[date_column] == date) & (result_df[candidate_column] == candidate), "y_offset"
                ] = y_offset
            else:
                # Première utilisation de cette position
                used_positions[rounded_value] = 1

    return result_df


def plot_ranked_time_approval_profile(
    si: SurveysInterface,
    show_no_opinion: bool = True,
    on_rolling_data: bool = False,
) -> go.Figure:
    # Candidat list sorted the rank in the last poll
    si_most_recent = si.most_recent_survey
    si_most_recent.df = si_most_recent.df.sort_values(by="rang")
    titles_candidates = [f"{c} {rank2str(i+1)}" for i, c in enumerate(si_most_recent.candidates)]

    # size of the figure
    n_rows, n_cols = _generate_windows_size(len(si_most_recent.candidates))
    idx_rows, idx_cols = np.unravel_index([i for i in range(si_most_recent.nb_candidates)], (n_rows, n_cols))
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

    show_legend = False
    for row, col, c in zip(idx_rows, idx_cols, si_most_recent.candidates):
        fig = plot_time_approval_profile(
            si=si.select_candidate(c),
            fig=fig,
            show_legend=show_legend,
            no_layout=True,
            row=row,
            col=col,
        )
        fig.update_yaxes(range=[0, 50], row=row, col=col, title="Approbation (%)" if col == 1 else "")

    fig.update_layout(
        yaxis_range=(0, 50),
        width=1200,
        height=900 if n_rows > 1 else 450,
        legend_title_text="Approbation",
        autosize=True,
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.05),  # 50 % of the figure width/
        yaxis=dict(
            tickfont_size=15,
            title="Approbation (%)",  # candidat
            automargin=True,
        ),
        plot_bgcolor="white",
    )

    # Title
    pop_str = " - population: " + str(si.df["population"].iloc[0]) if "population" in si.df.columns else ""
    title = f"<b>Classement des candidats à l'approbation</b>{pop_str}"

    source_str = f"source: {si.sources_string}" if si.sources_string is not None else ""
    source_str += ", " if si.sponsors_string is not None else ""
    sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors_string is not None else ""
    subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {si.most_recent_date}.</i>"

    fig.update_layout(title=title + subtitle, title_x=0.5)
    fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    return fig


def plot_time_approval_profile(
    si: SurveyInterface,
    fig: go.Figure = None,
    show_legend: bool = True,
    show_logo: bool = True,
    no_layout: bool = False,
    row: int = None,
    col: int = None,
) -> go.Figure:

    if fig is None:
        fig = go.Figure()

    si.df.sort_values(by="end_date")

    COLORS = load_colors()
    candidate_color = COLORS.get(si.df.candidate.unique().tolist()[0], {"couleur": "black"})["couleur"]

    # convert to rgba from hex
    def hex_to_rgba(hex_color, alpha=0.5):
        """Convertit une couleur hexadécimale (#RRGGBB) en format RGBA."""
        # Supprimer les guillemets simples ou doubles si présents
        hex_color = hex_color.strip("'\"")

        # Enlever le # si présent
        hex_color = hex_color.lstrip("#")

        try:
            # Format standard #RRGGBB
            if len(hex_color) == 6:
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            # Format court #RGB
            elif len(hex_color) == 3:
                r, g, b = int(hex_color[0], 16) * 17, int(hex_color[1], 16) * 17, int(hex_color[2], 16) * 17
            # Autre format non reconnu, utiliser une couleur par défaut
            else:
                r, g, b = 0, 0, 0  # Noir
        except ValueError:
            # En cas d'erreur de conversion, utiliser une couleur par défaut
            r, g, b = 0, 0, 0  # Noir

        return f"rgba({r}, {g}, {b}, {alpha})"

    candidate_color_rgba = hex_to_rgba(candidate_color, alpha=0.5)

    fig.add_trace(
        go.Scatter(
            x=si.dates,
            y=si.df["approbation"],
            hoverinfo="x+y",
            mode="lines+markers",
            line=dict(width=0.5, color=candidate_color),
            stackgroup="one",
            fillcolor=candidate_color_rgba,
            name=si.df.candidate.unique().tolist()[0],
            showlegend=show_legend,
        ),
        row=row,
        col=col,
    )

    if not no_layout:
        fig.update_layout(
            yaxis_range=(0, 50),
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
            f"<b>Evolution des mentions à l'approbation"
            + f"<br> pour le candidat {si.df.candidate.unique().tolist()[0]}</b>"
        )
        source_str = f"source: {si.sources_string}" if si.sources is not None else ""
        source_str += ", " if si.sponsors is not None else ""
        sponsor_str = f"commanditaire: {si.sponsors_string}" if si.sponsors is not None else ""
        subtitle = f"<br><i>{source_str}{sponsor_str}, dernier sondage: {si.most_recent_date}.</i>"

        fig.update_layout(title=title + subtitle, title_x=0.5)

        if show_logo:
            fig = _add_image_to_fig(fig, x=1.00, y=1.05, sizex=0.10, sizey=0.10, xanchor="right")

    return fig
