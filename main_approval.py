from pathlib import Path
import tap
from mjtracker.batch_plots import (
    batch_approval_profile,
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
    png: bool = False
    json: bool = True
    svg: bool = False
    csv: Path = Path("../mj-database-2027/mj2027.csv")
    dest: Path = Path("../trackerapp/data/graphs/")


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

    # filter the majority judgement data to get a smoother estimation of grades
    filtered = False
    if filtered:
        si.filter()

    # Apply the Majority Judgement rule
    si.apply_approval(up_to="plutôt satisfait")

    # # generate all the graphs
    batch_approval_profile(si, args, auto_text=True)
    # batch_ranking(si, args, filtered=filtered)
    # batch_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.ALL, filtered=filtered)
    # batch_ranked_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.ALL, filtered=filtered)
    # batch_time_merit_profile_all(si, args, aggregation_mode, filtered=filtered)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
