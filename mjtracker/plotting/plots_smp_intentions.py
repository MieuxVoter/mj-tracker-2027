"""
SMP Intentions plotting - Based on sandbox/plot_intentions_2027.py

Functions for plotting aggregated intention data for 2027 presidential election.
"""

import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from ..plotting.plot_utils import load_colors
from ..core.smp_data import SMPData


def rank2str(rank):
    """Convert rank number to string (e.g., 1 -> '1er', 2 -> '2ème')."""
    if rank == 1:
        return "1er"
    elif rank == 2:
        return "2ème"
    elif rank == 3:
        return "3ème"
    else:
        return f"{int(rank)}ème"


def _extended_name_annotations(
    df,
    candidate=None,
    show_rank=True,
    show_intention=True,
    breaks_in_names=True,
):
    """Create extended name annotations for candidates."""
    text = candidate if candidate else ""
    
    annotations = []
    
    if show_rank and "rang" in df.columns:
        rank = df["rang"].iloc[-1]
        annotations.append(f"{rank2str(rank)}")
    
    if show_intention and "valeur" in df.columns:
        intention = df["valeur"].iloc[-1]
        annotations.append(f"{intention:.1f}%")
    
    if annotations:
        text += " (" + ", ".join(annotations) + ")"
    
    return text


def plot_intention(
    df,
    col_intention: str,
    fig: go.Figure = None,
    colored: bool = True,
    row: int = None,
    col: int = None,
    max_gap_days: int = 60,
    connect_gap_days: int = 150,
) -> go.Figure:
    """
    Plot intention curve for a single candidate with gap detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with candidate data (must contain single candidate)
    col_intention : str
        Column name for intention values
    fig : go.Figure
        Plotly figure to add traces to
    colored : bool
        Use colored lines and annotations
    row : int
        Subplot row number
    col : int
        Subplot column number
    max_gap_days : int
        Maximum gap in days to keep points in same segment (default: 60 days)
    connect_gap_days : int
        Maximum gap to connect segments with dashed line (default: 150 days)
        
    Returns
    -------
    go.Figure
        Updated figure
    """
    if fig is None:
        fig = go.Figure()
    
    candidate = df["candidat"].unique()[0]
    colors = load_colors()
    color = colors.get(candidate, {}).get("couleur", "#d3d3d3") if colored else "#d3d3d3"
    opacity = 1 if colored else 0.3
    width = 3 if colored else 1
    
    # Convert dates to datetime for gap detection
    df_copy = df.copy()
    df_copy["fin_enquete"] = pd.to_datetime(df_copy["fin_enquete"])
    df_copy = df_copy.sort_values("fin_enquete")
    
    # Group by date and calculate mean when multiple polls on same day
    agg_dict = {
        col_intention: "mean",
        "candidat": "first",
    }
    if "rang" in df_copy.columns:
        agg_dict["rang"] = "first"
    if "erreur_sup" in df_copy.columns:
        agg_dict["erreur_sup"] = "mean"
    if "erreur_inf" in df_copy.columns:
        agg_dict["erreur_inf"] = "mean"
        
    df_copy = df_copy.groupby("fin_enquete").agg(agg_dict).reset_index()
    
    # Detect gaps and split into segments
    segments = []
    current_segment = []
    gaps_to_connect = []
    
    for idx, row_data in df_copy.iterrows():
        if not current_segment:
            current_segment.append(row_data)
        else:
            time_gap = (row_data["fin_enquete"] - current_segment[-1]["fin_enquete"]).days
            if time_gap <= max_gap_days:
                current_segment.append(row_data)
            else:
                if len(current_segment) > 0:
                    segments.append(pd.DataFrame(current_segment))
                    if time_gap <= connect_gap_days:
                        gaps_to_connect.append(len(segments) - 1)
                current_segment = [row_data]
    
    if len(current_segment) > 0:
        segments.append(pd.DataFrame(current_segment))
    
    # Plot each segment
    for i, segment_df in enumerate(segments):
        if len(segment_df) >= 2 or (i == len(segments) - 1 and not segment_df.empty):
            mode = "lines+markers" if colored else "markers"
            
            fig.add_trace(
                go.Scatter(
                    x=segment_df["fin_enquete"],
                    y=segment_df[col_intention],
                    mode=mode,
                    line=dict(color=color, width=width) if colored else None,
                    marker=dict(color=color, size=6, opacity=0.8),
                    hovertemplate=f"<b>{candidate}</b><br>" +
                                  "Date: %{x|%Y-%m-%d}<br>" +
                                  "Intention: %{y:.1f}%<br>" +
                                  "<extra></extra>",
                    name=candidate,
                    showlegend=False,
                    legendgroup=None,
                ),
                row=row,
                col=col,
            )
            
            # Add dashed connection line to next segment if gap is within connect_gap_days
            if i < len(segments) - 1 and colored:
                next_segment = segments[i + 1]
                time_gap = (next_segment["fin_enquete"].iloc[0] - segment_df["fin_enquete"].iloc[-1]).days
                if time_gap <= connect_gap_days:
                    fig.add_trace(
                        go.Scatter(
                            x=[segment_df["fin_enquete"].iloc[-1], next_segment["fin_enquete"].iloc[0]],
                            y=[segment_df[col_intention].iloc[-1], next_segment[col_intention].iloc[0]],
                            mode="lines",
                            line=dict(color=color, width=width, dash="dot"),
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=None,
                        ),
                        row=row,
                        col=col,
                    )
    
    # Last point marker
    if segments and "rang" in df.columns:
        last_segment = segments[-1]
        rank = last_segment["rang"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=last_segment["fin_enquete"].iloc[-1:],
                y=last_segment[col_intention].iloc[-1:],
                mode="markers",
                name=candidate,
                marker=dict(color=color, opacity=opacity, size=8),
                legendgroup=None,
                showlegend=False,
                text=[f"{rank2str(rank)}"],
            ),
            row=row,
            col=col,
        )
    
    # Error bands
    if colored and "erreur_sup" in df.columns and "erreur_inf" in df.columns:
        c = px.colors.hex_to_rgb(color)
        c_alpha = f"rgba({c[0]},{c[1]},{c[2]},0.2)"
        
        for segment_df in segments:
            if len(segment_df) >= 2:
                x_date = segment_df["fin_enquete"].tolist()
                y_upper = segment_df["erreur_sup"].tolist()
                y_lower = segment_df["erreur_inf"].tolist()

                fig.add_scatter(
                    x=x_date + x_date[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=c_alpha,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=candidate,
                    legendgroup=None,
                    row=row,
                    col=col,
                )

        # Annotation on last segment
        if segments:
            last_segment = segments[-1]
            candidate_text = _extended_name_annotations(
                last_segment, candidate=candidate, show_rank=True, show_intention=True, breaks_in_names=True
            )
            
            fig.add_annotation(
                x=pd.to_datetime(last_segment["fin_enquete"].iloc[-1]),
                y=last_segment[col_intention].iloc[-1],
                xanchor="left",
                xshift=10,
                yanchor="middle",
                text=f"<b>{candidate_text}</b>",
                font=dict(family="Arial", size=12, color=color),
                showarrow=False,
                row=row,
                col=col,
            )

    return fig


def plot_intention_data(
    df,
    col_intention: str,
    fig: go.Figure = None,
    colored: bool = True,
    row: int = None,
    col: int = None,
    max_gap_days: int = 60,
) -> go.Figure:
    """Plot raw intention data points (scatter) with connecting lines within segments."""
    if fig is None:
        fig = go.Figure()
    
    candidate = df["candidat"].unique()[0]
    colors = load_colors()
    color = colors.get(candidate, {}).get("couleur", "lightgray") if colored else "lightgray"
    opacity = 0.6 if colored else 0.3
    
    # Convert dates and sort
    df_copy = df.copy()
    df_copy["fin_enquete"] = pd.to_datetime(df_copy["fin_enquete"])
    df_copy = df_copy.sort_values("fin_enquete")
    
    # Detect gaps and split into segments (same logic as plot_intention)
    segments = []
    current_segment = []
    
    for idx, row_data in df_copy.iterrows():
        if not current_segment:
            current_segment.append(row_data)
        else:
            time_gap = (row_data["fin_enquete"] - current_segment[-1]["fin_enquete"]).days
            if time_gap <= max_gap_days:
                current_segment.append(row_data)
            else:
                if len(current_segment) > 0:
                    segments.append(pd.DataFrame(current_segment))
                current_segment = [row_data]
    
    if len(current_segment) > 0:
        segments.append(pd.DataFrame(current_segment))
    
    # Plot each segment with lines+markers
    for segment_df in segments:
        if len(segment_df) >= 2:
            # Draw thin line connecting raw data points within segment
            if colored:
                fig.add_trace(
                    go.Scatter(
                        x=segment_df["fin_enquete"],
                        y=segment_df[col_intention],
                        mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        hoverinfo="skip",
                        name=candidate,
                        showlegend=False,
                        legendgroup=None,
                    ),
                    row=row,
                    col=col,
                )
        
        # Draw markers for all points (even single points)
        fig.add_trace(
            go.Scatter(
                x=segment_df["fin_enquete"],
                y=segment_df[col_intention],
                mode="markers",
                marker=dict(color=color, opacity=opacity, size=5),
                hovertemplate=f"<b>{candidate}</b><br>" +
                              "Date: %{x|%Y-%m-%d}<br>" +
                              "Intention: %{y:.1f}%<br>" +
                              "<extra></extra>",
                name=candidate,
                showlegend=False,
                legendgroup=None,
            ),
            row=row,
            col=col,
        )

    return fig


def plot_aggregated_intentions(
    smp_data: SMPData,
    candidates_to_highlight=None,
    date_range=None,
) -> go.Figure:
    """
    Plot aggregated intentions for all candidates.
    
    Parameters
    ----------
    smp_data : SMPData
        Data object with polling data
    candidates_to_highlight : list, optional
        List of candidate names to highlight (others will be greyed out)
    date_range : tuple, optional
        (start_date, end_date) to filter data
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    df_ranks = smp_data.get_ranks()
    df_intentions = smp_data.get_intentions()
    
    # Filter by date range if provided
    if date_range:
        df_ranks = df_ranks[
            (df_ranks["fin_enquete"] >= date_range[0]) &
            (df_ranks["fin_enquete"] <= date_range[1])
        ]
        df_intentions = df_intentions[
            (df_intentions["fin_enquete"] >= date_range[0]) &
            (df_intentions["fin_enquete"] <= date_range[1])
        ]
    
    fig = go.Figure()
    
    # Plot all candidates
    all_candidates = df_ranks["candidat"].unique()
    
    for candidate in all_candidates:
        df_cand_ranks = df_ranks[df_ranks["candidat"] == candidate]
        df_cand_intentions = df_intentions[df_intentions["candidat"] == candidate]
        
        # Determine if this candidate should be highlighted
        colored = (candidates_to_highlight is None) or (candidate in candidates_to_highlight)
        
        # Plot rolling average
        if not df_cand_ranks.empty:
            fig = plot_intention(df_cand_ranks, col_intention="valeur", fig=fig, colored=colored)
        
        # Plot raw data points
        if not df_cand_intentions.empty:
            fig = plot_intention_data(df_cand_intentions, col_intention="intentions", fig=fig, colored=colored)
    
    # Update layout
    fig.update_yaxes(title="Intention de vote (%)", range=[0, max(40, df_ranks["valeur"].max() + 5)])
    fig.update_xaxes(title="Date de fin d'enquête", type="date")
    
    title = "<b>Intentions de vote agrégées (tous scénarios confondus) - Élection présidentielle 2027</b><br>"
    title += f"<sub>Source: presidentielle2027.json | Moyenne mobile sur 14 jours</sub>"
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        width=1400,
        height=800,
        hovermode="closest",
        template="plotly_white",
        images=[
            dict(
                source="https://raw.githubusercontent.com/MieuxVoter/mj-tracker-2027/refs/heads/main/icons/logo.svg",
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                sizex=0.1,
                sizey=0.1,
                xanchor="left",
                yanchor="top",
                layer="above"
            )
        ]
    )
    
    return fig
