from pathlib import Path
import tap
from mjtracker.batch_figure import (
    batch_merit_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_comparison_ranking,
    batch_time_merit_profile_all,
    batch_ranked_time_merit_profile,
    batch_comparison_intention,
)
from mjtracker.load_surveys import load_surveys
from mjtracker.smp_data import SMPData
from mjtracker.misc.enums import Candidacy, AggregationMode, PollingOrganizations, UntilRound
from mjtracker import SurveysInterface


class Arguments(tap.Tap):
    merit_profiles: bool = True
    comparison_ranking_plot: bool = True
    ranking_plot: bool = True
    time_merit_profile: bool = True
    ranked_time_merit_profile: bool = True
    comparison_intention: bool = True
    test: bool = False
    show: bool = True
    html: bool = True
    png: bool = True
    json: bool = True
    csv: Path = Path("/home/pierre/Documents/Mieux_voter/database-mj-2027/mj-database-2027/mj2027.csv")
    dest: Path = Path("trackerapp/data/graphs/")


def main(args: Arguments):
    args.dest.mkdir(exist_ok=True, parents=True)
    aggregation_mode = AggregationMode.FOUR_MENTIONS

    si = SurveysInterface.load(
        args.csv,
        # candidates=Candidacy.ALL_CURRENT_CANDIDATES,
        # polling_organization=PollingOrganizations.IPSOS,
        # until_round=UntilRound.FIRST,
    )
    si.to_no_opinion_surveys()
    si.aggregate(aggregation_mode)
    filter = True
    if filter:
        si.filter()
    si.apply_mj()

    df = si.df
    batch_ranked_time_merit_profile(df, args, aggregation_mode, polls=PollingOrganizations.ALL, on_rolling_data=filter)


if __name__ == "__main__":
    args = Arguments()
    main(args)
