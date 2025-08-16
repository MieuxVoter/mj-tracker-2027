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

    df = si.df
    # Try a sankey plotly that maps "candidate" to their "mention_majoritaire"
    import plotly.graph_objects as go

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

    # Récupérer les couleurs pour les grades/mentions
    grade_colors = get_grade_color_palette(si.nb_grades)

    # Créer des listes de noms uniques pour les candidats et les mentions
    candidates = df["candidate"].unique().tolist()
    mentions = df["mention_majoritaire"].unique().tolist()

    # Créer le mapping de mention à index de couleur
    # Assure que les mentions sont associées aux bonnes couleurs de la palette
    mention_to_color_idx = {mention: i for i, mention in enumerate(si.grades) if mention in mentions}

    # Définir les couleurs pour les liens du premier niveau (candidat → mention)
    COLORS = load_colors()
    link_colors_level1 = []
    for candidate in candidates:
        candidate_data = df[df["candidate"] == candidate]
        link_colors_level1 += [COLORS.get(candidate, {"couleur": "black"})["couleur"]      ]




    # Définir les couleurs pour les liens du second niveau (mention → approbation/rejet)
    link_colors_level2 = []
    for i in range(len(sources_level2)):
        source_index = sources_level2[i]
        mention_index = source_index - len(candidates)
        mention = mentions[mention_index]

        if "approbation" in labels[targets_level2[i]]:
            link_colors_level2.append("rgba(44, 160, 44, 0.4)")  # Vert pour approbation
        else:
            link_colors_level2.append("rgba(214, 39, 40, 0.4)")  # Rouge pour rejet

    # Créer des couleurs pour les nœuds
    node_colors = (
            ["rgba(31, 119, 180, 0.8)" for _ in candidates]  # Bleu pour les candidats
            + [grade_colors[mention_to_color_idx[mention]] for mention in
               mentions]  # Utiliser les couleurs de la palette pour les mentions
            + [
                "rgba(44, 160, 44, 0.8)" if "approbation" in cat else "rgba(214, 39, 40, 0.8)"
                for cat in approbation_rejet_categories
            ]  # Vert pour approbation, Rouge pour rejet
    )

    # Combiner les données des deux niveaux
    sources = sources_level1 + sources_level2
    targets = targets_level1 + targets_level2
    values = values_level1 + values_level2

    # Combiner les couleurs des liens
    link_colors = link_colors_level1 + link_colors_level2

    def rgb_to_hex(rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    node_colors = [rgb_to_hex(c) if isinstance(c, tuple) else c for c in node_colors]
    link_colors = [rgb_to_hex(c) if isinstance(c, tuple) else c for c in link_colors]

    # Créer la figure avec les couleurs appropriées

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",  # keeps nodes aligned
                node=dict(
                    pad=30,  # more padding for space between nodes
                    thickness=30,  # make nodes wider so labels are clearly on them
                    line=dict(color="black", width=1),  # node borders
                    label=labels,
                    color=node_colors
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors
                ),
            )
        ]
    )

    # Example node positions
    x_pos = [0, 0.3, 0.6]  # Candidate, mention, app/rej
    y_pos = [i * 0.05 for i in range(len(candidates))]  # spread vertically

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[""] * len(labels),  # Remove default labels
            x=[0] * len(candidates) + [0.5] * len(mentions) + [1] * len(approbation_rejet_categories),
            y=y_pos + [None] * (len(labels) - len(candidates)),
            color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    ))

    # Add annotations for candidates
    for i, cand in enumerate(candidates):
        fig.add_annotation(
            x=-0.02,  # slightly left of the node
            y=y_pos[i],
            xref="paper", yref="paper",
            text=cand,
            showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right"
        )

    fig.show()


    fig.update_layout(
        title_text="Diagramme Sankey : Candidats → Mentions → Approbation/Rejet",
        font_size=10
    )
    fig.show()


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    main(args)
