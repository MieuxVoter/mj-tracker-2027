from pathlib import Path
import tap
from mjtracker.batch_plots import (
    batch_merit_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_ranked_time_merit_profile,
    batch_time_merit_profile_all,
)
from mjtracker.color_utils import get_grade_color_palette
from mjtracker.plot_utils import load_colors

# from mjtracker.smp_data import SMPData # not available yet.
from mjtracker.misc.enums import AggregationMode, PollingOrganizations, UntilRound
from mjtracker import SurveysInterface

from sandbox.bezier_utils import plot_double_sided_sankey_bezier


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
    png: bool = False
    json: bool = True
    svg: bool = False
    csv: Path = Path("../mj-database-2027/mj2027.csv")
    # dest: Path = Path("../trackerapp/data/graphs/")
    dest: Path = Path("jmtracker.fr/plotly-standalone/graphs/jm")


def main(args: Arguments):
    args.dest.mkdir(exist_ok=True, parents=True)

    # load from the database
    si = SurveysInterface.load(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
    )
    # remove no opinion data
    si.to_no_opinion_surveys()
    si = si.select_survey("ipsos_202507")

    # Apply the Majority Judgement rule
    si.apply_mj()

    df = si.df.copy().sort_values(by="rang", ascending=True, na_position="last")

    candidates = df["candidate"].to_list()
    mentions = si.grades[::-1]  # reverse the order of mentions
    approbation_categories = df["tri_majoritaire"].to_list()

    import pandas as pd
    import numpy as np

    cumulative_df = pd.DataFrame()
    cumulative_df["candidate"] = df["candidate"]
    colheaders = [f"intention_mention_{i}" for i in range(1, si.nb_grades + 1)]

    cumulative_sum = np.zeros(len(df))
    for i, col in enumerate(colheaders):
        cumulative_sum += df[col]
        cumulative_df[f"cumulative_intention_mention_{i + 1}"] = cumulative_sum

    cols = list(cumulative_df.columns)[1:]
    proportion_map = {}
    for candidate in candidates:
        proportion = cumulative_df[cumulative_df["candidate"] == candidate]
        proportion_map[candidate] = proportion[cols].values.round(decimals=2) / 100.0
        proportion_map[candidate] = proportion_map[candidate].tolist()[0]

    best_grade = df["mention_majoritaire"].values.tolist()

    candidate_to_mention = {candidate: mentions.index(grade) for candidate, grade in zip(candidates, best_grade)}

    approbration = df["tri_majoritaire"].values.tolist()

    candidate_to_approbation = {
        candidate: 0 if approbation == "rejet" else 1 for candidate, approbation in zip(candidates, approbration)
    }

    fig = plot_double_sided_sankey_bezier(
        candidates,
        mentions,
        approbation_categories,
        candidate_to_mention,
        candidate_to_approbation,
        proportion_map,
    )
    fig.show()


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
