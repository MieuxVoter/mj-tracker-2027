"""
Script to analyze ELABE polls for the left segment population.
Produces the classical analysis: merit profiles, rankings, time profiles.
Compares Approval voting vs Majority Judgment rankings.
"""

from pathlib import Path
import copy
import tap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from mjtracker.plotting.batch_plots import (
    batch_merit_profile,
    batch_approval_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_ranked_time_merit_profile,
    batch_time_approval_profiles,
    batch_ranked_time_approval_profile,
)
from mjtracker.export.export_compact_json import export_compact_json

from mjtracker.misc.enums import AggregationMode, PollingOrganizations
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
    svg: bool = False
    csv: str = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/main/mj2027_left.csv"
    dest: Path = Path("../trackerapp/data/graphs/left")
    reorder_grades_as_central: bool = True  # Move 'sans opinion' to middle grade
    filter_primaire_candidates: bool = True  # Only include left primaire candidates


# ELABE grade order with "sans opinion" as central grade
ELABE_GRADES_CENTRAL_OPINION = [
    "une image très positive",
    "une image plutôt positive",
    "sans opinion",  # moved to middle
    "une image plutôt négative",
    "une image très négative",
]

# Candidats déclarés et pressentis de la primaire de la gauche 2027
LEFT_PRIMAIRE_CANDIDATES = [
    # Candidats déclarés à la primaire
    "François Ruffin",
    "Clémentine Autain",
    "Marine Tondelier",
    # Pressentis / potentiels
    "Olivier Faure",
    "Ségolène Royal",
    "Jérôme Guedj",
    # "François Hollande",
    # Hors primaire mais figures majeures de la gauche
    "Raphaël Glucksmann",
    "Jean-Luc Mélenchon",
    # "Fabien Roussel",
]


def filter_candidates(si: SurveysInterface, candidates: list[str]) -> SurveysInterface:
    """Filter the surveys to only include specified candidates."""
    # Get exact matches from the data
    available = set(si.df["candidate"].unique())
    selected = [c for c in candidates if c in available]
    missing = [c for c in candidates if c not in available]

    if missing:
        print(f"  ⚠ Candidats non trouvés: {', '.join(missing)}")

    print(f"  → Filtrage sur {len(selected)} candidats: {', '.join(selected)}")

    # Filter dataframe
    si.df = si.df[si.df["candidate"].isin(selected)].copy()
    return si


def main_mj(args: Arguments):
    """Generate Majority Judgment analysis for left segment."""
    print("\n=== Analyse Jugement Majoritaire - Segment Gauche ===")

    dest = args.dest / "mj"
    dest.mkdir(exist_ok=True, parents=True)

    aggregation_mode = AggregationMode.NO_AGGREGATION

    # Load from the database (left segment, ELABE)
    si = SurveysInterface.load_from_url(
        args.csv,
        polling_organization=PollingOrganizations.ELABE,
    )
    print(f"  ✓ {len(si.df)} enregistrements chargés")
    print(f"  ✓ {si.nb_surveys} sondages: {', '.join(si.surveys)}")

    # Filter to only include primaire candidates
    if args.filter_primaire_candidates:
        si = filter_candidates(si, LEFT_PRIMAIRE_CANDIDATES)

    # Optionally reorder grades to move "sans opinion" to the middle
    if args.reorder_grades_as_central:
        print("  → Reordering grades: 'sans opinion' as central grade")
        si.reorder_grades(ELABE_GRADES_CENTRAL_OPINION)

    # Remove no opinion data
    if not args.reorder_grades_as_central:
        si.to_no_opinion_surveys()

    # Apply the Majority Judgement rule
    si.apply_mj()

    # Export JSON standard
    si.df.to_json(dest / "latest_survey_mj_left.json", orient="records")
    print(f"  ✓ Exported: {dest / 'latest_survey_mj_left.json'}")

    # Export JSON compact optimisé (filter out None ranks)
    df_valid = si.df.dropna(subset=["rang"])
    if len(df_valid) > 0:
        export_compact_json(df_valid, dest / "latest_survey_mj_left_compact.json", voting_method="majority_judgment")
        print(f"  ✓ Exported: {dest / 'latest_survey_mj_left_compact.json'}")
    else:
        print("  ⚠ Skipped compact JSON export (no valid ranks)")

    # Export CSV
    si.df.to_csv(dest / "latest_survey_mj_left.csv", index=False)
    print(f"  ✓ Exported: {dest / 'latest_survey_mj_left.csv'}")

    # Create modified args for this subdirectory
    mj_args = copy.copy(args)
    mj_args.dest = dest

    # Generate all the graphs
    print("\n  Génération des graphiques JM...")
    batch_merit_profile(si, mj_args, auto_text=False)
    batch_ranking(si, mj_args)
    # batch_time_merit_profile(si, mj_args, aggregation_mode, polls=PollingOrganizations.ELABE)
    batch_ranked_time_merit_profile(si, mj_args, aggregation_mode, polls=PollingOrganizations.ELABE)

    print("  ✓ Analyse JM terminée")
    return si


def main_approval(args: Arguments):
    """Generate Approval voting analysis for left segment."""
    print("\n=== Analyse Approbation - Segment Gauche ===")

    dest = args.dest / "approval"
    dest.mkdir(exist_ok=True, parents=True)

    aggregation_mode = AggregationMode.NO_AGGREGATION

    # Load from the database (left segment, ELABE)
    si = SurveysInterface.load_from_url(
        args.csv,
        polling_organization=PollingOrganizations.ELABE,
    )
    print(f"  ✓ {len(si.df)} enregistrements chargés")

    # Filter to only include primaire candidates
    if args.filter_primaire_candidates:
        si = filter_candidates(si, LEFT_PRIMAIRE_CANDIDATES)

    # Remove no opinion data
    si.to_no_opinion_surveys()

    # Apply the Approval rule (positive + very positive for ELABE polls)
    si.apply_approval(up_to="une image plutôt positive")

    # Export JSON standard
    si.df.to_json(dest / "latest_survey_approval_left.json", orient="records")
    print(f"  ✓ Exported: {dest / 'latest_survey_approval_left.json'}")

    # Export JSON compact optimisé (filter out None ranks)
    df_valid = si.df.dropna(subset=["rang"])
    if len(df_valid) > 0:
        export_compact_json(df_valid, dest / "latest_survey_approval_left_compact.json", voting_method="approval")
        print(f"  ✓ Exported: {dest / 'latest_survey_approval_left_compact.json'}")
    else:
        print("  ⚠ Skipped compact JSON export (no valid ranks)")

    # Export CSV
    si.df.to_csv(dest / "latest_survey_approval_left.csv", index=False)
    print(f"  ✓ Exported: {dest / 'latest_survey_approval_left.csv'}")

    # Create modified args for this subdirectory
    approval_args = copy.copy(args)
    approval_args.dest = dest

    # Generate all the graphs
    print("\n  Génération des graphiques Approbation...")
    batch_approval_profile(si, approval_args, auto_text=True)
    batch_time_approval_profiles(si, approval_args, aggregation_mode, polls=PollingOrganizations.ELABE)
    batch_ranking(si, approval_args, voting_str_title="à l'approbation", show_grade_area=False)
    batch_ranked_time_approval_profile(si, approval_args, aggregation_mode, polls=PollingOrganizations.ELABE)

    print("  ✓ Analyse Approbation terminée")
    return si


def main(args: Arguments):
    """Main function: run full analysis for left segment with ELABE polls."""
    print("\n" + "=" * 60)
    print("ANALYSE ELABE - SEGMENT GAUCHE")
    print("=" * 60)

    args.dest.mkdir(exist_ok=True, parents=True)

    # Run MJ analysis
    si_mj = main_mj(args)

    # Run Approval analysis
    si_approval = main_approval(args)

    print("\n" + "=" * 60)
    print("✓ ANALYSE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    main(args)
