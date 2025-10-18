from pathlib import Path
import tap
from mjtracker.batch_plots import (
    batch_merit_profile,
    batch_approval_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_ranked_time_merit_profile,
    batch_time_approval_profiles,
    batch_ranked_time_approval_profile,
)
from mjtracker.export_compact_json import export_compact_json

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
    show: bool = False
    html: bool = False
    png: bool = True
    json: bool = True
    svg: bool = True
    csv: str = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/refs/heads/main/mj2027.csv"
    dest: Path = Path("../trackerapp/data/graphs/mj")


def main_mj(args: Arguments):
    args = Arguments().parse_args(known_only=True)
    args.dest = args.dest / "mj"
    args.dest.mkdir(exist_ok=True, parents=True)

    aggregation_mode = AggregationMode.NO_AGGREGATION

    # load from the database
    si = SurveysInterface.load_from_url(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
    )
    # remove no opinion data
    si.to_no_opinion_surveys()

    # Apply the Majority Judgement rule
    si.apply_mj()

    # Export JSON standard
    si.df.to_json(args.dest / "latest_survey_mj.json", orient="records")

    # Export JSON compact optimisé
    export_compact_json(si.df, args.dest / "latest_survey_mj_compact.json", voting_method="majority_judgment")

    # Export CSV
    si.df.to_csv(args.dest / "latest_survey_mj.csv", index=False)

    # generate all the graphs
    batch_merit_profile(si, args, auto_text=False)
    batch_ranking(si, args)
    batch_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)
    batch_ranked_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)


def main_approval(args: Arguments):
    args = Arguments().parse_args(known_only=True)
    args.dest = args.dest / "approval"
    args.dest.mkdir(exist_ok=True, parents=True)

    aggregation_mode = AggregationMode.NO_AGGREGATION

    # load from the database
    si = SurveysInterface.load(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
    )
    # remove no opinion data
    si.to_no_opinion_surveys()

    # Apply the Majority Judgement rule
    si.apply_approval(up_to="plutôt satisfait")

    # Export JSON standard
    si.df.to_json(args.dest / "latest_survey_approval.json", orient="records")

    # Export JSON compact optimisé
    export_compact_json(si.df, args.dest / "latest_survey_approval_compact.json", voting_method="approval")

    # Export CSV
    si.df.to_csv(args.dest / "latest_survey_approval.csv", index=False)

    # generate all the graphs
    batch_approval_profile(si, args, auto_text=True)
    batch_time_approval_profiles(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)
    batch_ranking(si, args, voting_str_title="à l'approbation", show_grade_area=False)
    batch_ranked_time_approval_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    args.dest.mkdir(exist_ok=True, parents=True)

    main_mj(args)
    main_approval(args)
