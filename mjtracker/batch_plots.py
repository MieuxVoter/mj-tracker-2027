from typing import Iterable

from .surveys_inferface import SurveysInterface
from .plot_utils import export_fig
from .plots_v2 import (
    plot_merit_profiles as pmp,
    ranking_plot as rkp,
    plot_time_merit_profile as ptmp,
    plot_ranked_time_merit_profile as prtmp,
    plot_time_merit_profile_all_polls,
    plot_approval_profiles
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
            fig = pmp(
                si=si_survey,
                auto_text=auto_text,
                show_no_opinion=True,
            )
            filename = f"{survey_id}"
            print(filename)
            export_fig(fig, args, filename)

def batch_approval_profile(si: SurveysInterface, args, auto_text: bool = False):
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
            fig = plot_approval_profiles(
                si=si_survey,
                auto_text=auto_text,
                show_no_opinion=True,
            )
            filename = f"{survey_id}"
            print(filename)
            export_fig(fig, args, filename)

def batch_ranking(si: SurveysInterface, args, filtered: bool = False, show_grade_area: bool = False):
    for poll in PollingOrganizations:
        si_poll = si.select_polling_organization(poll)
        if si_poll.df.empty:
            continue

        if args.ranking_plot:
            fig = rkp(
                si_poll,
                show_grade_area=show_grade_area,
                breaks_in_names=True,
                show_best_grade=False,
            )
            filtered_str = "_filtered" if filtered else ""
            filename = f"ranking_plot_{poll.name}{filtered_str}"
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
            si_candidate = si_poll.select_candidate(candidate)

            if args.time_merit_profile:
                fig = ptmp(si_candidate)
                filename = f"time_merit_profile{aggregation.string_label}_{candidate}_{si_poll.sources_string}"
                print(filename)
                export_fig(fig, args, filename)

    for candidate in si.candidates:
        temp_df = si.select_candidate(candidate).df
        if args.time_merit_profile:
            fig = ptmp(temp_df, aggregation)
            filename = f"time_merit_profile_comparison{aggregation.string_label}_{candidate}"
            print(filename)
            export_fig(fig, args, filename)


def batch_ranked_time_merit_profile(
    si: SurveysInterface,
    args,
    aggregation,
    polls: PollingOrganizations = PollingOrganizations,
    filtered: bool = False,
):
    if not isinstance(polls, Iterable):
        polls = [polls]
    for poll in polls:
        if poll == PollingOrganizations.ALL and aggregation == AggregationMode.NO_AGGREGATION:
            continue
        si_poll = si.select_polling_organization(poll)

        if si_poll.df.empty:
            continue

        filtered_str = "_filtered" if filtered else ""

        if args.ranked_time_merit_profile:
            fig = prtmp(
                si_poll,
                show_no_opinion=True,
                on_rolling_data=filtered,
            )
            filename = f"ranked_time_merit_profile{aggregation.string_label}_{si_poll.sources_string}{filtered_str}"
            print(filename)
            export_fig(fig, args, filename)


def batch_time_merit_profile_all(si: SurveysInterface, args, aggregation, filtered: bool = False):
    if aggregation == AggregationMode.NO_AGGREGATION:
        raise ValueError("Need to have an AggregationMode such as FOUR_MENTION to make it work.")

    filtered_str = "_filtered" if filtered else ""

    for candidate in si.candidates:
        si_candidate = si.select_candidate(candidate)
        temp_df = si_candidate.df
        if temp_df.empty:
            continue
        if args.time_merit_profile:
            fig = plot_time_merit_profile_all_polls(si_candidate, aggregation)
            filename = f"time_merit_profile{aggregation.string_label}_{candidate}{filtered_str}"
            print(filename)
            export_fig(fig, args, filename)
