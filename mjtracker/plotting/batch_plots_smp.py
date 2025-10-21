from typing import Iterable

from pathlib import Path
import plotly.graph_objects as go
from ..core.surveys_interface import SurveysInterface

# Use validated SMP plotting functions
from .plots_smp import comparison_ranking_plot, plot_comparison_intention

from .plot_utils import export_fig
from ..misc.enums import PollingOrganizations, AggregationMode
from ..core.smp_data import SMPData


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
                # Skip if candidate has no SMP data
                if fig is None:
                    continue
                    
                filename = f"intention_{aggregation.string_label}_{candidate}_{si_poll.sources_string}"
                print(filename)
                export_fig(fig, args, filename)
