"""Plotting utilities and batch generation functions."""

# Note: imports are lazy to avoid loading heavy dependencies at module import time
# Use explicit imports: from mjtracker.plotting.plots import plot_merit_profiles
# Or use the functions directly: from mjtracker.plotting import batch_plots

__all__ = [
    # Module names available
    "plots",
    "plots_v2",
    "plots_smp",
    "plot_utils",
    "color_utils",
    "batch_plots",
    "batch_plots_smp",
    "batch_figure",
]

from .batch_plots import (
    batch_approval_profile,
    batch_ranking,
    batch_time_merit_profile,
    batch_ranked_time_merit_profile,
    batch_time_merit_profile_all,
    batch_time_approval_profiles,
    batch_ranked_time_approval_profile,
)
