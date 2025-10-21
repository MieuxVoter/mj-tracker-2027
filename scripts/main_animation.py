"""
Awful draft to test the animation of the merit profile. Don't think this will surprise anyone.
"""

from pathlib import Path
import tap

from mjtracker.plots import plot_animated_merit_profile

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
    # aggregation_mode = AggregationMode.FOUR_MENTIONS

    si = SurveysInterface.load(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
    )
    si.to_no_opinion_surveys()
    si.apply_mj()
    df_survey = si.df[si.df["poll_id"] == si.surveys[0]].copy()

    first_idx = df_survey.first_valid_index()
    source = df_survey["institut"].loc[first_idx]
    sponsor = df_survey["commanditaire"].loc[first_idx]
    date = df_survey["end_date"].loc[first_idx]
    nb_grades = df_survey["nombre_mentions"].unique()[0]
    grades = [f"mention{i}" for i in range(1, nb_grades + 1)]

    fig = plot_animated_merit_profile(
        df=df_survey,
        grades=grades,
        source=source,
        date=date,
        sponsor=sponsor,
        show_no_opinion=True,
    )
    filename = f"animation_test"
    print(filename)
    fig.show()
    # export as a gif for the animation
    # fig.write_image(f"{args.dest}/{filename}.gif")


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
