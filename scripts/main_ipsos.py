from pathlib import Path
import tap
from mjtracker.plotting.batch_plots import (
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
    svg: bool = True
    csv: str = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/refs/heads/main/mj2027.csv"
    dest: Path = Path("../trackerapp/data/graphs/mj")


def main(args: Arguments):
    args.dest.mkdir(exist_ok=True, parents=True)
    aggregation_mode = AggregationMode.NO_AGGREGATION

    # load from the database
    si = SurveysInterface.load_from_url(
        args.csv,
        polling_organization=PollingOrganizations.IPSOS,
        # candidates=
    )
    # remove no opinion data
    si.to_no_opinion_surveys()

    # aggregation to merge all database if possible.
    # si.aggregate(aggregation_mode)

    # filter the majority judgement data to get a smoother estimation of grades
    # filtered = True
    # if filtered:
    #     si.filter(rolling_period="90d")

    # Apply the Majority Judgement rule
    si.apply_mj()

    # generate all the graphs
    batch_merit_profile(si, args, auto_text=False)
    # only last one
    from mjtracker.plotting.plot_utils import export_fig
    from mjtracker.plotting.plots_v2 import (
        plot_merit_profiles as pmp,
    )

    survey_id = si.surveys[-1]
    si_survey = si.select_survey(survey_id)

    if args.merit_profiles:
        fig = pmp(
            si=si_survey,
            auto_text=False,
            show_no_opinion=False,
        )
        filename = f"{survey_id}"
        print(filename)
        export_fig(fig, args, filename)

    # batch_ranking(si, args, filtered=filtered)
    # batch_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS)
    # batch_ranked_time_merit_profile(si, args, aggregation_mode, polls=PollingOrganizations.IPSOS, filtered=filtered)
    # batch_time_merit_profile_all(si, args, aggregation_mode, filtered=filtered)

    df = si.df

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # === 1. Charger les données ===
    # Exemple :
    # df = pd.read_csv("data_jm.csv", sep=",")
    # ou
    # df = pd.read_excel("data_jm.xlsx")

    # === 2. Nettoyage et structuration ===
    # Normalisation des noms et dates
    df["name"] = df["name"].astype(str).str.strip()
    df["surname"] = df["surname"].astype(str).str.strip()
    df["candidate"] = (df["surname"] + " " + df["name"]).str.strip()
    df["poll_id"] = df["poll_id"].astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    # Suppression des doublons éventuels
    df = df.drop_duplicates(subset=["candidate_id", "poll_id"])

    # On garde uniquement les colonnes nécessaires
    core_cols = ["candidate_id", "candidate", "poll_id", "start_date", "rang", "mention_majoritaire"]
    df = df[core_cols].dropna(subset=["rang", "mention_majoritaire"])

    # === 3. Classements par vague ===
    classements = df.pivot_table(index="candidate", columns="poll_id", values="rang", aggfunc="first")

    # === 4. Corrélation intermois ===
    corr = classements.corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Corrélation de rangs (Jugement Majoritaire) entre vagues")
    plt.tight_layout()
    plt.show()

    # === 5. Stabilité des mentions ===
    mentions = df.pivot_table(index="candidate", columns="poll_id", values="mention_majoritaire", aggfunc="first")

    if mentions.shape[1] >= 2:
        mois1, mois2 = mentions.columns[0], mentions.columns[-1]
        stable_rate = (mentions[mois1] == mentions[mois2]).mean() * 100
        print(
            f"📊 {stable_rate:.1f}% des personnalités conservent la même mention majoritaire "
            f"entre {mois1} et {mois2}."
        )
    else:
        print("⚠️ Pas assez de vagues pour analyser la stabilité des mentions.")

    # === 6. Heatmap de transitions de mentions ===
    if mentions.shape[1] >= 2:
        transitions = pd.crosstab(mentions[mois1], mentions[mois2], normalize="index")

        plt.figure(figsize=(7, 5))
        sns.heatmap(transitions, annot=True, fmt=".1%", cmap="Blues")
        plt.title(f"Transitions de mentions JM : {mois1} → {mois2}")
        plt.xlabel(f"Mention en {mois2}")
        plt.ylabel(f"Mention en {mois1}")
        plt.tight_layout()
        plt.show()

    # === 7. Évolution du rang entre la première et la dernière vague ===
    if classements.shape[1] >= 2:
        delta = classements.iloc[:, -1] - classements.iloc[:, 0]
        delta = delta.dropna().sort_values()

        plt.figure(figsize=(8, 8))
        delta.plot(kind="barh")
        plt.axvline(0, color="black", lw=1)
        plt.title("Évolution du rang Jugement Majoritaire (dernier - premier sondage)")
        plt.xlabel("Variation de rang (négatif = amélioration)")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Pas assez de vagues pour visualiser les variations de rang.")


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
