from pathlib import Path
import tap

from mjtracker.core.smp_data import SMPData
from plotly.graph_objs import Figure
import plotly.io as pio

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
    json: bool = False
    svg: bool = False
    csv: str = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/refs/heads/main/mj2027.csv"
    dest: Path = Path("../trackerapp/data/graphs")


def main_smp(args: Arguments):
    """Generate SMP intention plots in dedicated /smp folder."""
    print("\n=== Generating SMP intention plots ===")

    # Create SMP output directory
    smp_dest = args.dest / "smp"
    smp_dest.mkdir(exist_ok=True, parents=True)

    # Load SMP data
    smp = SMPData()
    print(f"✓ SMPData loaded: {len(smp.df_raw)} records")

    from mjtracker.plotting.plots_smp_intentions import plot_aggregated_intentions
    from mjtracker.plotting.plot_utils import export_fig

    # All candidates plot
    print("  Generating aggregated intentions (all candidates)...")
    fig_all = plot_aggregated_intentions(smp, candidates_to_highlight=None)
    fig_all.show()

    # Export as HTML and PNG
    if args.html:
        output_html = smp_dest / "all_candidates_2027.html"
        fig_all.write_html(str(output_html))
        print(f"  ✓ Saved: {output_html}")

    if args.png:
        # output_png = smp_dest / "all_candidates_2027.png"
        # export_fig(fig_all, args, "all_candidates_2027")
        # print(f"  ✓ Saved: {output_png}")
        # Clone the figure to avoid modifying the original
        fig_export = Figure(fig_all)
        
        # Ensure all traces are visible and properly configured for static export
        fig_export.update_layout(
            width=2800,
            height=1600,
            margin=dict(l=80, r=200, t=100, b=80),
        )
        
        # Force update all traces to ensure visibility
        for trace in fig_export.data:
            trace.update(visible=True)
        
        output_png = smp_dest / "all_candidates_2027.png"
        # fig_export.write_image(str(output_png), width=2800, height=1600, scale=1)
        # print(f"  ✓ Saved: {output_png}")

        # Try using to_image directly
        img_bytes = pio.to_image(fig_all, format="png", width=2800, height=1600, scale=2, engine="kaleido")
        
        with open(output_png, "wb") as f:
            f.write(img_bytes)

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



if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    args.dest.mkdir(exist_ok=True, parents=True)
    main_smp(args)
