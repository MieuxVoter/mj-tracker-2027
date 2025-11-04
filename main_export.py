from pathlib import Path
import tap
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

from mjtracker.core.smp_data import SMPData
from mjtracker.misc.enums import AggregationMode, PollingOrganizations, UntilRound, Candidacy
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
    html: bool = True  # Enable HTML export for SMP
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


def main_smp(args: Arguments):
    """Generate SMP intention plots in dedicated /smp folder."""
    print("\n=== Generating SMP intention plots ===")

    # Create SMP output directory
    smp_dest = args.dest.parent / "smp"
    smp_dest.mkdir(exist_ok=True, parents=True)

    # Load SMP data
    smp = SMPData()
    print(f"✓ SMPData loaded: {len(smp.df_raw)} records")

    # Generate aggregated intentions plot (like sandbox/plot_intentions_2027.py)
    from mjtracker.plotting.plots_smp_intentions import plot_aggregated_intentions
    from mjtracker.plotting.plot_utils import export_fig

    # All candidates plot
    print("  Generating aggregated intentions (all candidates)...")
    fig_all = plot_aggregated_intentions(smp, candidates_to_highlight=None)

    # Export as HTML and PNG
    if args.html:
        output_html = smp_dest / "all_candidates_2027.html"
        fig_all.write_html(str(output_html))
        print(f"  ✓ Saved: {output_html}")

    if args.png:
        output_png = smp_dest / "all_candidates_2027.png"
        fig_all.write_image(str(output_png), width=1400, height=800, scale=2)
        print(f"  ✓ Saved: {output_png}")

    if args.svg:
        output_svg = smp_dest / "all_candidates_2027.svg"
        fig_all.write_image(str(output_svg), width=1400, height=800)
        print(f"  ✓ Saved: {output_svg}")

    if args.json:
        import json
        from plotly.io import write_json

        output_json = smp_dest / "all_candidates_2027.json"
        write_json(fig_all, output_json)
        print(f"  ✓ Saved: {output_json}")

    # Top candidates plot (highlighted)
    print("  Generating aggregated intentions (top candidates highlighted)...")
    top_candidates = [
        "Jordan Bardella",
        "Édouard Philippe",
        "Jean-Luc Mélenchon",
        "Bruno Retailleau",
        "Raphaël Glucksmann",
        "Marine Le Pen",
        "Gabriel Attal",
        "François Bayrou",
    ]
    fig_top = plot_aggregated_intentions(smp, candidates_to_highlight=top_candidates)

    if args.html:
        output_html = smp_dest / "top_candidates_2027.html"
        fig_top.write_html(str(output_html))
        print(f"  ✓ Saved: {output_html}")

    if args.png:
        output_png = smp_dest / "top_candidates_2027.png"
        fig_top.write_image(str(output_png), width=1400, height=800, scale=2)
        print(f"  ✓ Saved: {output_png}")

    if args.svg:
        output_svg = smp_dest / "top_candidates_2027.svg"
        fig_top.write_image(str(output_svg), width=1400, height=800)
        print(f"  ✓ Saved: {output_svg}")

    if args.json:
        from plotly.io import write_json

        output_json = smp_dest / "top_candidates_2027.json"
        write_json(fig_top, output_json)
        print(f"  ✓ Saved: {output_json}")

    print("✓ SMP intention plots generated")


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    args.dest.mkdir(exist_ok=True, parents=True)

    # main_mj(args)
    # main_approval(args)
    main_smp(args)
