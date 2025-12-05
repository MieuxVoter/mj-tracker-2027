from pathlib import Path
import tap
from mjtracker.plotting import (
    batch_approval_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_ranked_time_merit_profile,
    batch_time_merit_profile_all,
    batch_time_approval_profiles,
    batch_ranked_time_approval_profile,
)

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
    show: bool = False
    html: bool = False
    png: bool = True
    json: bool = True
    svg: bool = False
    csv: str = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/refs/heads/main/mj2027.csv"
    dest: Path = Path("../trackerapp/data/graphs/approval")


def main(args: Arguments):
    args.dest.mkdir(exist_ok=True, parents=True)
    aggregation_mode = AggregationMode.NO_AGGREGATION

    # load from the database
    si = SurveysInterface.load_from_url(
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
    si.apply_approval()

    # # generate all the graphs
    batch_approval_profile(si, args, auto_text=True)
    batch_time_approval_profiles(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)
    batch_ranking(si, args, filtered=filtered, show_grade_area=False)
    batch_ranked_time_approval_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
