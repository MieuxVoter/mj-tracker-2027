from typing import Iterable

from pathlib import Path
import plotly.graph_objects as go
from .surveys_interface import SurveysInterface
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
from .plots_v2 import plot_merit_profiles as pmp, ranking_plot as rkp, plot_time_merit_profile as ptmp
from .misc.enums import PollingOrganizations, AggregationMode
from .smp_data import SMPData


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
