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
from mjtracker.batch_plots import batch_merit_profile as bmp, batch_ranking as br
from mjtracker.smp_data import SMPData
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
    html: bool = True
    png: bool = True
    json: bool = True
    csv: Path = Path("/home/pierre/Documents/Mieux_voter/database-mj-2027/mj-database-2027/mj2027.csv")
    dest: Path = Path("../trackerapp/data/graphs/")


def main(args: Arguments):
    args.dest.mkdir(exist_ok=True, parents=True)
    aggregation_mode = AggregationMode.FOUR_MENTIONS

    si = SurveysInterface.load(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
    )
    si.to_no_opinion_surveys()
    si.apply_mj()
    df = si.df

    # # generate merit profile figures
    bmp(si, args, auto_text=False)
    # batch_merit_profile(df, args)
    br(si, args, on_rolling_data=False)
    if not args.test:
        # generate ranking figures
        batch_ranking(df, args)
        #     # # generate comparison ranking figures
        #     batch_comparison_ranking(df, smp_data, args)
        #     # # # generate time merit profile figures
        batch_time_merit_profile(df, args, aggregation_mode, polls=PollingOrganizations.ALL)
        # # # generate ranked time merit profile figures
        batch_ranked_time_merit_profile(df, args, aggregation_mode, polls=PollingOrganizations.IPSOS)
    #     # batch_ranked_time_merit_profile(df, args, aggregation_mode, polls=PollingOrganizations.ALL)
    #     # # comparison uninominal per candidates
    # batch_comparison_intention(df, smp_data, args, aggregation)
    #
    #     # aggregation = AggregationMode.FOUR_MENTIONS
    #     # df = load_surveys(
    #     #     args.csv,
    #     #     no_opinion_mode=True,
    #     #     candidates=Candidacy.ALL_CURRENT_CANDIDATES_WITH_ENOUGH_DATA,
    #     #     aggregation=aggregation,
    #     #     polling_organization=PollingOrganizations.ALL,
    #     #     until_round=UntilRound.FIRST,
    #     #     rolling_data=True,
    #     # )
    #     # # df = apply_mj(df, rolling_mj=False)
    #     # df = apply_mj(df, rolling_mj=True)
    #     # batch_time_merit_profile_all(df, args, aggregation, on_rolling_data=False)
    #     # batch_time_merit_profile_all(df, args, aggregation, on_rolling_data=True)
    #     # batch_comparison_ranking(df, smp_data, args, on_rolling_data=True)
    #     batch_ranked_time_merit_profile(df, args, aggregation_mode, on_rolling_data=False)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
