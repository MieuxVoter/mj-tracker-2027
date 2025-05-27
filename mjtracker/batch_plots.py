from typing import Iterable

from .surveys_inferface import SurveysInterface
from .plots import (
    plot_merit_profiles,
    ranking_plot,
    comparison_ranking_plot,
    plot_time_merit_profile,
    plot_time_merit_profile_all_polls,
    plot_ranked_time_merit_profile,
    plot_comparison_intention,
    export_fig,
)
from .utils import (
    get_candidates,
)
from .misc.enums import PollingOrganizations, AggregationMode
from .smp_data import SMPData


def batch_merit_profile(si: SurveysInterface, args, auto_text: bool = False):
    """
    Plot merit profiles for all polls

    Parameters
    ----------
    si : SurveysInterface
        containing the data of the polls
    args : Namespace
        containing the arguments
    auto_text : bool
        If True, the intention of grade is automatically generated on the merit profile plot
    """

    for survey_id in si.surveys:
        si_survey = si.select_survey(survey_id)

        if args.merit_profiles:
            fig = plot_merit_profiles(
                df=si_survey.df,
                grades=si_survey.grades,
                auto_text=auto_text,
                source=si_survey.source,
                date=si_survey.end_date,
                sponsor=si_survey.sponsor,
                show_no_opinion=True,
            )
            filename = f"{survey_id}"
            print(filename)
            export_fig(fig, args, filename)


def batch_ranking(si: SurveysInterface, args, on_rolling_data: bool = False):
    for poll in PollingOrganizations:
        si_poll = si.select_polling_organization(poll)
        if si_poll.df.empty:
            continue

        if args.ranking_plot:
            fig = ranking_plot(
                si_poll.df,
                source=si_poll.sources_string,
                sponsor=si_poll.sponsors_string,
                show_grade_area=True,
                breaks_in_names=True,
                show_best_grade=False,
                on_rolling_data=on_rolling_data,
            )
            filename = f"ranking_plot_{poll.name}"
            print(filename)
            export_fig(fig, args, filename)

def batch_comparison_ranking(si: SurveysInterface, smp_data: SMPData, args, on_rolling_data: bool = False):
    for poll in PollingOrganizations:
        si_poll = si.select_polling_organization(poll)

        if args.comparison_ranking_plot:
            fig = comparison_ranking_plot(
                si_poll.df,
                smp_data=smp_data,
                source=si.sources_string,
                on_rolling_data=on_rolling_data
            )
            roll = "_roll" if on_rolling_data else ""
            filename = f"comparison_ranking_plot_{poll.name}{roll}"
            print(filename)
            export_fig(fig, args, filename)

# todo to pursue.
def batch_time_merit_profile(df, args, aggregation, polls: PollingOrganizations = PollingOrganizations):
    # check if polls is iterable
    if not isinstance(polls, Iterable):
        polls = [polls]
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue
        df_poll = df[df["institut"] == poll.value].copy() if poll != PollingOrganizations.ALL else df.copy()
        first_idx = df_poll.first_valid_index()
        source = poll.value
        label = source if poll != PollingOrganizations.ALL else poll.name
        sponsor = df_poll["commanditaire"].loc[first_idx] if poll != PollingOrganizations.ALL else None
        aggregation_label = f"_{aggregation.name}" if aggregation != AggregationMode.NO_AGGREGATION else ""

        for c in get_candidates(df):
            temp_df = df_poll[df_poll["candidate"] == c]
            if temp_df.empty:
                continue
            if args.time_merit_profile:
                fig = plot_time_merit_profile(temp_df, source=source, sponsor=sponsor)
                filename = f"time_merit_profile{aggregation_label}_{c}_{label}"
                print(filename)
                export_fig(fig, args, filename)

    for c in get_candidates(df):
        temp_df = df[df["candidate"] == c]
        if args.time_merit_profile:
            fig = plot_time_merit_profile_all_polls(temp_df, aggregation)
            filename = f"time_merit_profile_comparison{aggregation_label}_{c}"
            print(filename)
            export_fig(fig, args, filename)


def batch_ranked_time_merit_profile(
    df, args, aggregation, polls: PollingOrganizations = PollingOrganizations, on_rolling_data: bool = False
):
    # check if polls is iterable
    if not isinstance(polls, Iterable):
        polls = [polls]
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue
        df_poll = df[df["institut"] == poll.value].copy() if poll != PollingOrganizations.ALL else df.copy()

        if df_poll.empty:
            continue

        first_idx = df_poll.first_valid_index()
        source = poll.value
        label = source if poll != PollingOrganizations.ALL else poll.name
        sponsor = df_poll["commanditaire"].loc[first_idx] if poll != PollingOrganizations.ALL else None
        aggregation_label = f"_{aggregation.name}" if aggregation != AggregationMode.NO_AGGREGATION else ""
        roll = "_roll" if on_rolling_data else ""

        if args.ranked_time_merit_profile:
            fig = plot_ranked_time_merit_profile(
                df_poll, source=source, sponsor=sponsor, show_no_opinion=True, on_rolling_data=on_rolling_data
            )
            filename = f"ranked_time_merit_profile{aggregation_label}_{label}{roll}"
            print(filename)
            export_fig(fig, args, filename)


def batch_comparison_intention(
    df,
    smp_data: SMPData,
    args,
    aggregation,
    polls: PollingOrganizations = PollingOrganizations,
    on_rolling_data: bool = False,
):
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue
        df_poll = df[df["nom_institut"] == poll.value].copy() if poll != PollingOrganizations.ALL else df.copy()
        first_idx = df_poll.first_valid_index()
        source = poll.value
        label = source if poll != PollingOrganizations.ALL else poll.name
        sponsor = df_poll["commanditaire"].loc[first_idx] if poll != PollingOrganizations.ALL else None
        aggregation_label = f"_{aggregation.name}" if aggregation != AggregationMode.NO_AGGREGATION else ""

        if df_poll.empty:
            continue
        if args.comparison_intention:
            for c in get_candidates(df_poll):
                temp_df = df_poll[df_poll["candidat"] == c]
                fig = plot_comparison_intention(
                    temp_df,
                    smp_data=smp_data,
                    source=source,
                    sponsor=sponsor,
                    on_rolling_data=on_rolling_data,
                )
                filename = f"intention_{label}{aggregation_label}_{c}"
                print(filename)
                export_fig(fig, args, filename)


def batch_time_merit_profile_all(df, args, aggregation, on_rolling_data: bool = False):
    if aggregation == AggregationMode.NO_AGGREGATION:
        raise ValueError("Need to have an AggregationMode such as FOUR_MENTION to make it work.")

    poll = PollingOrganizations.ALL
    df_poll = df
    first_idx = df_poll.first_valid_index()
    source = poll.value
    label = poll.name
    sponsor = None
    aggregation_label = f"_{aggregation.name}"
    roll = "_roll" if on_rolling_data else ""

    for c in get_candidates(df):
        temp_df = df_poll[df_poll["candidat"] == c]
        if temp_df.empty:
            continue
        if args.time_merit_profile:
            fig = plot_time_merit_profile(temp_df, source=source, sponsor=sponsor, on_rolling_data=on_rolling_data)
            filename = f"time_merit_profile{aggregation_label}_{c}_{label}{roll}"
            print(filename)
            export_fig(fig, args, filename)
