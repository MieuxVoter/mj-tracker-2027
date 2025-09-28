from pathlib import Path
import tap
from mjtracker.batch_plots import (
    batch_merit_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_ranked_time_merit_profile,
    batch_time_merit_profile_all,
)

# from mjtracker.smp_data import SMPData # not available yet.
from mjtracker.misc.enums import AggregationMode, PollingOrganizations, UntilRound
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
    html: bool = False
    png: bool = True
    json: bool = False
    svg: bool = False
    csv: Path = Path("/home/ppuchaud/Documents/perso/mieux_voter/mj-database-2027/mj2027.csv")
    dest: Path = Path("../trackerapp/data/graphs/mj")


def main(args: Arguments):
    args.dest.mkdir(exist_ok=True, parents=True)
    aggregation_mode = AggregationMode.NO_AGGREGATION

    # load from the database
    si = SurveysInterface.load(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
    )
    # remove no opinion data
    si.to_no_opinion_surveys()

    # aggregation to merge all database if possible.
    # si.aggregate(aggregation_mode)

    # filter the majority judgement data to get a smoother estimation of grades
    filtered = False

    # Apply the Majority Judgement rule
    si.apply_mj()

    # # generate all the graphs
    # batch_merit_profile(si, args, auto_text=False)
    # batch_ranking(si, args, filtered=filtered)
    # batch_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)
    batch_ranked_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS, filtered=filtered)
    # batch_time_merit_profile_all(si, args, aggregation_mode, filtered=filtered)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
