"""Debug script to test plotting with all features."""

from mjtracker.core.smp_data import SMPData
from mjtracker.plotting.plots_smp_intentions import plot_aggregated_intentions

print("Loading data...")
smp = SMPData()

print("\n=== Creating test plot ===")
fig = plot_aggregated_intentions(smp, candidates_to_highlight=None)

# Check number of traces
print(f"\nNumber of traces in figure: {len(fig.data)}")

# Analyze traces
trace_types = {}
for i, trace in enumerate(fig.data):
    trace_type = type(trace).__name__
    mode = getattr(trace, "mode", "N/A")
    fill = getattr(trace, "fill", "N/A")
    name = getattr(trace, "name", "N/A")

    key = f"{trace_type}_{mode}_{fill}"
    trace_types[key] = trace_types.get(key, 0) + 1

    if i < 5:  # Print first 5 traces for debugging
        print(f"Trace {i}: {trace_type}, mode={mode}, fill={fill}, name={name}")

print("\n=== Trace type summary ===")
for key, count in sorted(trace_types.items()):
    print(f"{key}: {count}")

# Check layout
print(f"\n=== Layout ===")
print(f"xaxis type: {fig.layout.xaxis.type if hasattr(fig.layout, 'xaxis') else 'N/A'}")
print(f"xaxis title: {fig.layout.xaxis.title.text if hasattr(fig.layout.xaxis, 'title') else 'N/A'}")

# Save for inspection
output_file = "test_debug_plot.html"
fig.write_html(output_file)
print(f"\n✓ Saved to {output_file}")
