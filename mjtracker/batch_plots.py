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
from .plots_v2 import plot_merit_profiles as pmp, ranking_plot as rkp
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
            fig = pmp(
                si=si_survey,
                auto_text=auto_text,
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
            fig = rkp(
                si_poll,
                show_grade_area=True,
                breaks_in_names=True,
                show_best_grade=False,
            )
            filename = f"ranking_plot_{poll.name}"
            print(filename)
            export_fig(fig, args, filename)


def batch_comparison_ranking(si: SurveysInterface, smp_data: SMPData, args, on_rolling_data: bool = False):
    for poll in PollingOrganizations:
        si_poll = si.select_polling_organization(poll)

        if args.comparison_ranking_plot:
            fig = comparison_ranking_plot(
                si_poll.df, smp_data=smp_data, source=si.sources_string, on_rolling_data=on_rolling_data
            )
            roll = "_roll" if on_rolling_data else ""
            filename = f"comparison_ranking_plot_{poll.name}{roll}"
            print(filename)
            export_fig(fig, args, filename)


def batch_time_merit_profile(
    si: SurveysInterface, args, aggregation, polls: PollingOrganizations = PollingOrganizations
):
    # check if polls is iterable
    if not isinstance(polls, Iterable):
        polls = [polls]
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue
        si_poll = si.select_polling_organization(poll)

        for candidate in si_poll.candidates:
            temp_df = si_poll.select_candidate(candidate).df

            if args.time_merit_profile:
                fig = plot_time_merit_profile(temp_df, source=si_poll.sources_string, sponsor=si_poll.sponsors_string)
                filename = f"time_merit_profile{aggregation.string_label}_{candidate}_{si_poll.sources_string}"
                print(filename)
                export_fig(fig, args, filename)

    for candidate in si.candidates:
        temp_df = si.select_candidate(candidate).df
        if args.time_merit_profile:
            fig = plot_time_merit_profile_all_polls(temp_df, aggregation)
            filename = f"time_merit_profile_comparison{aggregation.string_label}_{candidate}"
            print(filename)
            export_fig(fig, args, filename)


def batch_ranked_time_merit_profile(
    si: SurveysInterface,
    args,
    aggregation,
    polls: PollingOrganizations = PollingOrganizations,
    on_rolling_data: bool = False,
):
    if not isinstance(polls, Iterable):
        polls = [polls]
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue
        si_poll = si.select_polling_organization(poll)

        if si_poll.df.empty:
            continue

        roll = "_roll" if on_rolling_data else ""

        if args.ranked_time_merit_profile:
            fig = plot_ranked_time_merit_profile(
                si_poll.df,
                source=si_poll.sources_string,
                sponsor=si_poll.sponsors_string,
                show_no_opinion=True,
                on_rolling_data=on_rolling_data,
            )
            filename = f"ranked_time_merit_profile{aggregation.string_label}_{si_poll.sources_string}{roll}"
            print(filename)
            export_fig(fig, args, filename)


def batch_comparison_intention(
    si: SurveysInterface,
    smp_data: SMPData,
    args,
    aggregation,
    polls: PollingOrganizations = PollingOrganizations,
    on_rolling_data: bool = False,
):
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue

        si_poll = si.select_polling_organization(poll)

        if si_poll.df.empty:
            continue
        if args.comparison_intention:
            for candidate in si_poll.candidates:
                temp_df = si_poll.select_candidate(candidate).df
                fig = plot_comparison_intention(
                    temp_df,
                    smp_data=smp_data,
                    source=si_poll.sources_string,
                    sponsor=si_poll.sponsors_string,
                    on_rolling_data=on_rolling_data,
                )
                filename = f"intention_{aggregation.string_label}_{candidate}_{si_poll.sources_string}"
                print(filename)
                export_fig(fig, args, filename)


def batch_time_merit_profile_all(si: SurveysInterface, args, aggregation, on_rolling_data: bool = False):
    if aggregation == AggregationMode.NO_AGGREGATION:
        raise ValueError("Need to have an AggregationMode such as FOUR_MENTION to make it work.")

    roll = "_roll" if on_rolling_data else ""

    for candidate in si.candidates:
        si_candidate = si.select_candidate(candidate)
        temp_df = si_candidate.df
        if temp_df.empty:
            continue
        if args.time_merit_profile:
            fig = plot_time_merit_profile(
                temp_df, source=si.sources_string, sponsor=None, on_rolling_data=on_rolling_data
            )
            filename = f"time_merit_profile{aggregation.string_label}_{candidate}{roll}"
            print(filename)
            export_fig(fig, args, filename)
