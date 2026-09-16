"""
Génère les figures SMP consommées par le site.

Le site charge des **figures Plotly au format JSON** (quelques dizaines de Ko) et
les rend avec un plotly.js partagé, au lieu de l'ancien HTML autonome de 4,9 Mo
qui embarquait toute la bibliothèque à chaque page.

Sorties dans ``<dest>/smp/`` :

- ``r1_<methode>.json``   : premier tour, une figure par méthode de lissage ;
- ``r2_<hypothese>.json`` : second tour, une figure par duel encore d'actualité ;
- ``manifest.json``       : liste des méthodes et des duels. C'est lui qui pilote
  les boutons du site — ajouter une méthode ou un duel ne demande aucune retouche
  du HTML ;
- ``all_candidates_2027.{html,png}`` : conservés pour les consommateurs existants
  (``main_export.py``, ``trackerapp``) et pour la version sans JavaScript.
"""

import json
from pathlib import Path
from typing import Optional

import tap
from plotly.graph_objs import Figure
import plotly.io as pio

from mjtracker.core.smp_data import SMPData, SOURCE_URL
from mjtracker.core.smoothing import DEFAULT_METHOD, SMOOTHERS


class Arguments(tap.Tap):
    test: bool = False
    show: bool = False
    html: bool = True
    png: bool = False
    json: bool = False
    svg: bool = False
    csv: str = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/refs/heads/main/mj2027.csv"
    source: Optional[str] = None  # fichier local de sondages, sinon la source GitHub par défaut
    dest: Path = Path("../trackerapp/data/graphs")


def _round_floats(value, digits: int = 2):
    """
    Arrondit récursivement les flottants d'une structure JSON.

    Plotly sérialise en pleine précision (17 chiffres pour une valeur qui en vaut
    trois), ce qui pèse environ 40 % du fichier pour rien : on affiche des dixièmes
    de point.
    """
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_floats(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    return value


def _write_figure(fig: Figure, path: Path) -> int:
    """Écrit une figure en JSON allégé et retourne sa taille en octets."""
    payload = _round_floats(json.loads(fig.to_json()))
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def _build_first_round(source: str, smp_dest: Path) -> tuple[list, SMPData]:
    """Une figure par méthode de lissage. Retourne (entrées du manifeste, SMPData par défaut)."""
    from mjtracker.plotting.plots_smp_intentions import plot_aggregated_intentions

    entries = []
    default_smp = None

    for method, (label, _) in SMOOTHERS.items():
        print(f"  · lissage {method} ({label})...")
        smp = SMPData(source_file=source, method=method)
        fig = plot_aggregated_intentions(smp, candidates_to_highlight=None)

        size = _write_figure(fig, smp_dest / f"r1_{method}.json")
        entries.append({"id": method, "label": label})
        print(f"    ✓ r1_{method}.json ({size // 1024} Ko)")

        if method == DEFAULT_METHOD:
            default_smp = smp

    if default_smp is None:
        raise RuntimeError(f"La méthode par défaut {DEFAULT_METHOD!r} est absente du registre SMOOTHERS.")

    return entries, default_smp


def _fetch_source_once(source: Optional[str], workdir: Path) -> str:
    """
    Télécharge la source une seule fois et retourne un chemin local.

    Chaque ``SMPData`` relit la source ; avec quatre méthodes, cela ferait quatre
    téléchargements identiques à chaque exécution — soit près de 200 par jour avec
    le cron de 30 minutes, et quatre occasions d'échouer au lieu d'une.
    """
    if source is not None:
        return source

    import requests

    print(f"  · téléchargement de la source ({SOURCE_URL.rsplit('/', 1)[-1]})...")
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()

    local = workdir / "presidentielle2027.source.json"
    local.write_bytes(response.content)
    print(f"    ✓ {len(response.content) // 1024} Ko en cache local")

    return str(local)


def _short_label(label: str) -> str:
    """
    « Marine Le Pen vs Édouard Philippe » -> « Le Pen vs Philippe ».

    On retire le seul prénom, c'est-à-dire le premier mot : les particules
    (« Le Pen », « de Villepin ») sont ainsi conservées.
    """
    return " vs ".join(nom.split(" ", 1)[-1] for nom in label.split(" vs "))


def _build_second_round(smp: SMPData, smp_dest: Path) -> list:
    """Une figure par duel encore d'actualité."""
    from mjtracker.plotting.plots_smp_second_round import plot_second_round_duel

    duels = smp.get_second_round(only_active=True)
    entries = []

    for hypothese, duel in duels.items():
        fig = plot_second_round_duel(duel, method=smp.method, hypothese=hypothese)
        size = _write_figure(fig, smp_dest / f"r2_{hypothese}.json")
        entries.append(
            {
                "id": hypothese,
                "label": duel["label"],
                "label_court": _short_label(duel["label"]),
                "candidats": sorted(duel["candidats"].keys()),
                "n_sondages": duel["n_sondages"],
                "derniere_date": duel["derniere_date"],
            }
        )
        print(f"    ✓ r2_{hypothese}.json ({size // 1024} Ko) — {duel['label']}")

    return entries


def _write_legacy_exports(args: Arguments, fig_all: Figure, smp_dest: Path) -> None:
    """Exports historiques, conservés pour les consommateurs existants."""
    if args.html:
        output_html = smp_dest / "all_candidates_2027.html"
        fig_all.write_html(str(output_html))
        print(f"  ✓ {output_html.name} (repli sans JavaScript)")

    if args.png:
        fig_export = Figure(fig_all)
        for trace in fig_export.data:
            trace.update(visible=True)

        img_bytes = pio.to_image(fig_all, format="png", width=2800, height=1600, scale=2, engine="kaleido")
        (smp_dest / "all_candidates_2027.png").write_bytes(img_bytes)
        print("  ✓ all_candidates_2027.png")

    if args.svg:
        output_svg = smp_dest / "all_candidates_2027.svg"
        fig_all.write_image(str(output_svg), width=1400, height=800)
        print(f"  ✓ {output_svg.name}")


def main_smp(args: Arguments):
    """Generate SMP intention plots in dedicated /smp folder."""
    print("\n=== Generating SMP intention plots ===")

    smp_dest = args.dest / "smp"
    smp_dest.mkdir(exist_ok=True, parents=True)

    print("\n[0/3] Source")
    source = _fetch_source_once(args.source, args.dest)

    print("\n[1/3] Premier tour")
    methods, smp = _build_first_round(source, smp_dest)

    print("\n[2/3] Second tour")
    duels = _build_second_round(smp, smp_dest)

    print("\n[3/3] Manifeste et exports historiques")

    # Candidats du plus fort au plus faible : sur petit écran, le site n'en affiche
    # qu'une partie, sinon les vingt courbes forment une pelote illisible.
    from mjtracker.plotting.plots_smp_intentions import _get_last_candidates_values_and_rank_them

    classement = _get_last_candidates_values_and_rank_them(smp.get_ranks())
    candidats = [
        {"nom": nom, "valeur": round(float(valeur), 1)}
        for nom, (valeur, _) in sorted(classement.items(), key=lambda kv: kv[1][1])
    ]

    manifest = {
        "mise_a_jour": smp.aggregated_data["mise_a_jour"],
        "dernier_sondage": smp.aggregated_data["dernier_sondage"],
        "methode_par_defaut": DEFAULT_METHOD,
        "methodes": methods,
        "duels": duels,
        "candidats": candidats,
    }
    (smp_dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ manifest.json ({len(methods)} méthodes, {len(duels)} duels)")

    from mjtracker.plotting.plots_smp_intentions import plot_aggregated_intentions

    fig_all = plot_aggregated_intentions(smp, candidates_to_highlight=None)
    if args.show:
        fig_all.show()

    _write_legacy_exports(args, fig_all, smp_dest)

    print(f"\n✓ Terminé : {smp_dest.resolve()}")


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    args.dest.mkdir(exist_ok=True, parents=True)
    main_smp(args)
