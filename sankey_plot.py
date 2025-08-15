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
    show: bool = False
    html: bool = False
    png: bool = False
    json: bool = True
    svg: bool = False
    csv: Path = Path("/mj2027.csv")
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

    df = si.df
    # Try a sankey plotly that maps "candidate" to their "mention_majoritaire"
    import plotly.graph_objects as go

    # Créer des listes de noms uniques pour les candidats et les mentions
    candidates = df["candidate"].unique().tolist()
    mentions = df["mention_majoritaire"].unique().tolist()

    # Préparer les étiquettes des nœuds
    labels = candidates + mentions

    # Créer le mapping de chaque candidat et mention vers un index
    candidate_to_idx = {name: i for i, name in enumerate(candidates)}
    mention_to_idx = {mention: i + len(candidates) for i, mention in enumerate(mentions)}

    # Préparer les données source, target et value
    sources = [candidate_to_idx[candidate] for candidate in df["candidate"]]
    targets = [mention_to_idx[mention] for mention in df["mention_majoritaire"]]
    values = [5 for _ in df["candidate"]]

    # Créer la figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="blue"),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )

    fig.update_layout(title_text="Diagramme Sankey : Candidats et Mentions Majoritaires", font_size=10)
    # fig.show()

    # Créer des listes de noms uniques pour les candidats et les mentions
    candidates = df["candidate"].unique().tolist()
    mentions = df["mention_majoritaire"].unique().tolist()

    # Créer les catégories d'approbation/rejet pour chaque mention
    approbation_rejet_categories = []
    for mention in mentions:
        approbation_rejet_categories.append(f"{mention} - approbation")
        approbation_rejet_categories.append(f"{mention} - rejet")

    # Préparer les étiquettes des nœuds
    labels = candidates + mentions + approbation_rejet_categories

    # Créer le mapping de chaque catégorie vers un index
    candidate_to_idx = {name: i for i, name in enumerate(candidates)}
    mention_to_idx = {mention: i + len(candidates) for i, mention in enumerate(mentions)}
    apprej_to_idx = {cat: i + len(candidates) + len(mentions) for i, cat in enumerate(approbation_rejet_categories)}

    # Préparer les données pour le premier niveau (candidat -> mention)
    sources_level1 = []
    targets_level1 = []
    values_level1 = []

    # Préparer les données pour le deuxième niveau (mention -> approbation/rejet)
    sources_level2 = []
    targets_level2 = []
    values_level2 = []

    # Calculer le nombre de votes pour chaque combinaison
    for candidate in candidates:
        candidate_data = df[df["candidate"] == candidate]

        for mention in mentions:
            mention_data = candidate_data[candidate_data["mention_majoritaire"] == mention]
            count = len(mention_data)

            if count > 0:
                # Ajouter le lien candidat -> mention
                sources_level1.append(candidate_to_idx[candidate])
                targets_level1.append(mention_to_idx[mention])
                values_level1.append(count)

                # Compter les approbations et rejets pour cette mention
                approbation_count = len(mention_data[mention_data["tri_majoritaire"] == "approbation"])
                rejet_count = len(mention_data[mention_data["tri_majoritaire"] == "rejet"])

                if approbation_count > 0:
                    sources_level2.append(mention_to_idx[mention])
                    targets_level2.append(apprej_to_idx[f"{mention} - approbation"])
                    values_level2.append(approbation_count)

                if rejet_count > 0:
                    sources_level2.append(mention_to_idx[mention])
                    targets_level2.append(apprej_to_idx[f"{mention} - rejet"])
                    values_level2.append(rejet_count)

    # Combiner les données des deux niveaux
    sources = sources_level1 + sources_level2
    targets = targets_level1 + targets_level2
    values = values_level1 + values_level2

    # Définir les couleurs pour les liens
    colors = ["rgba(31, 119, 180, 0.4)"] * len(sources_level1)

    # Ajouter les couleurs pour les liens de niveau 2
    for i in range(len(sources_level2)):
        # Vérifier si c'est une approbation ou un rejet en regardant le nom de la cible
        target_name = labels[targets[i + len(sources_level1)]]
        if "approbation" in target_name:
            colors.append("rgba(44, 160, 44, 0.4)")  # Vert pour approbation
        else:
            colors.append("rgba(214, 39, 40, 0.4)")  # Rouge pour rejet

    # Créer des couleurs pour les nœuds
    node_colors = (
        ["rgba(31, 119, 180, 0.8)" for _ in candidates]  # Bleu pour les candidats
        + ["rgba(255, 127, 14, 0.8)" for _ in mentions]  # Orange pour les mentions
        + [
            "rgba(44, 160, 44, 0.8)" if "approbation" in cat else "rgba(214, 39, 40, 0.8)"
            for cat in approbation_rejet_categories
        ]  # Vert pour approbation, Rouge pour rejet
    )

    # Créer la figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(source=sources, target=targets, value=values, color=colors),
            )
        ]
    )

    fig.update_layout(title_text="Diagramme Sankey : Candidats → Mentions → Approbation/Rejet", font_size=10)
    fig.show()


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
