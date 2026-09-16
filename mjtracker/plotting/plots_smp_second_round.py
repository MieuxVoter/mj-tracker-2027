"""
Tracé des hypothèses de second tour (duels).

Un duel n'est pas un graphique d'intentions comme un autre : il n'a que deux
courbes, elles somment à 100, et la seule chose qui compte est de savoir laquelle
passe au-dessus de 50 %. D'où les différences avec le graphique de premier tour :
une ligne de seuil à 50 %, une vraie légende (deux séries, inutile d'aller chercher
des étiquettes en marge) et un axe y resserré autour des valeurs observées.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ..core.smoothing import METHOD_SUBTITLES
from .plots_smp_intentions import (
    DEFAULT_MAX_GAP_DAYS,
    DEFAULT_RAW_MARKER_SIZE,
    _detect_time_segments,
)

# Deux couleurs de parti peuvent être quasi identiques (Le Pen et Bardella
# partagent le même bleu). Sous ce seuil de distance RVB, le second candidat
# bascule sur une couleur de contraste, sinon le duel est illisible.
MIN_COLOR_DISTANCE = 120.0
CONTRAST_COLOR = "#eb6834"

MAJORITY_LINE = 50.0
BAND_OPACITY = 0.18


def _color_distance(first: str, second: str) -> float:
    a = px.colors.hex_to_rgb(first)
    b = px.colors.hex_to_rgb(second)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _duel_colors(duel: Dict[str, Any], noms: List[str]) -> Dict[str, str]:
    """Une couleur par candidat, en garantissant qu'elles restent distinguables."""
    colors = {nom: duel["candidats"][nom].get("couleur", "#808080") for nom in noms}

    if _color_distance(colors[noms[0]], colors[noms[1]]) < MIN_COLOR_DISTANCE:
        colors[noms[1]] = CONTRAST_COLOR

    return colors


def _smoothed_frame(candidat_data: Dict[str, Any]) -> pd.DataFrame:
    """Série lissée d'un candidat, au format attendu par les helpers de tracé."""
    moy = candidat_data["intentions_moy"]
    df = pd.DataFrame(
        {
            "fin_enquete": pd.to_datetime(moy["end_date"]),
            "valeur": moy["valeur"],
            "erreur_inf": moy["erreur_inf"],
            "erreur_sup": moy["erreur_sup"],
            "erreur_inf_spread": moy["erreur_inf_spread"],
            "erreur_sup_spread": moy["erreur_sup_spread"],
        }
    )
    return df.sort_values("fin_enquete")


def _raw_frame(candidat_data: Dict[str, Any]) -> pd.DataFrame:
    raw = candidat_data["intentions"]
    df = pd.DataFrame(
        {
            "fin_enquete": pd.to_datetime(raw["fin_enquete"]),
            "intentions": raw["valeur"],
            "institut": raw["institut"],
            "commanditaire": raw["commanditaire"],
        }
    )
    return df.sort_values("fin_enquete")


def _add_raw_markers(fig: go.Figure, df_raw: pd.DataFrame, candidate: str, color: str) -> None:
    """Sondages bruts, avec le détail de l'institut au survol."""
    hover = [
        f"<b>{candidate}</b><br>{row['fin_enquete']:%d/%m/%Y}<br>{row['intentions']:.0f} %<br>{row['institut']}"
        for _, row in df_raw.iterrows()
    ]

    fig.add_trace(
        go.Scatter(
            x=df_raw["fin_enquete"],
            y=df_raw["intentions"],
            mode="markers",
            marker=dict(color=color, size=DEFAULT_RAW_MARKER_SIZE + 3, opacity=0.45, line=dict(width=0)),
            name=candidate,
            legendgroup=candidate,
            showlegend=False,
            hovertext=hover,
            hoverinfo="text",
            meta=dict(role="raw", candidat=candidate),
        )
    )


def _add_confidence_band(fig: go.Figure, segments: List[pd.DataFrame], candidate: str, color: str) -> None:
    """
    Une seule bande : l'intervalle à 95 % de l'estimation.

    Le premier tour en empile deux (dispersion entre instituts *et* précision de
    l'estimation). Sur un duel cela ferait quatre zones translucides pour deux
    courbes — illisible. On garde la précision de l'estimation, qui est ce qui
    permet de juger si l'écart au seuil de 50 % est significatif.
    """
    rgb = px.colors.hex_to_rgb(color)
    fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{BAND_OPACITY})"

    for segment in segments:
        if len(segment) < 2 or segment[["erreur_sup", "erreur_inf"]].isna().any().any():
            continue

        dates = segment["fin_enquete"].tolist()
        fig.add_trace(
            go.Scatter(
                x=dates + dates[::-1],
                y=segment["erreur_sup"].tolist() + segment["erreur_inf"].tolist()[::-1],
                fill="toself",
                fillcolor=fill,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=candidate,
                meta=dict(role="band", candidat=candidate),
            )
        )


def _add_smoothed_curve(fig: go.Figure, df_fit: pd.DataFrame, candidate: str, color: str) -> None:
    """Courbe lissée, découpée aux trous de sondage comme au premier tour."""
    segments = _detect_time_segments(df_fit, DEFAULT_MAX_GAP_DAYS)

    for index, segment in enumerate(segments):
        fig.add_trace(
            go.Scatter(
                x=segment["fin_enquete"],
                y=segment["valeur"],
                # `lines+markers` et non `lines` : une hypothèse ne reposant que sur
                # un seul sondage se réduit à un point, qu'une ligne n'afficherait pas.
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(color=color, size=4),
                name=candidate,
                legendgroup=candidate,
                showlegend=(index == 0),
                hovertemplate="%{y:.1f} %<extra>" + candidate + "</extra>",
                meta=dict(role="curve", candidat=candidate),
            )
        )


def plot_second_round_duel(
    duel: Dict[str, Any],
    method: str = "rolling",
    hypothese: Optional[str] = None,
) -> go.Figure:
    """
    Trace une hypothèse de second tour.

    Parameters
    ----------
    duel : dict
        Une entrée de ``SMPData.get_second_round()`` : ``label``, ``actif``,
        ``n_sondages``, ``derniere_date`` et ``candidats``.
    method : str
        Clé de la méthode de lissage, pour le sous-titre.
    hypothese : str, optional
        Identifiant ``H2_x``, affiché en sous-titre à titre de traçabilité.

    Returns
    -------
    go.Figure
    """
    noms = sorted(duel["candidats"].keys())
    colors = _duel_colors(duel, noms)

    fig = go.Figure()
    all_values = []

    for nom in noms:
        data = duel["candidats"][nom]
        color = colors[nom]

        df_fit = _smoothed_frame(data)
        df_raw = _raw_frame(data)
        all_values.extend(df_raw["intentions"].tolist())

        segments = _detect_time_segments(df_fit, DEFAULT_MAX_GAP_DAYS)
        _add_confidence_band(fig, segments, nom, color)
        _add_raw_markers(fig, df_raw, nom, color)
        _add_smoothed_curve(fig, df_fit, nom, color)

    # Seuil de victoire : sans lui le graphique ne dit pas qui gagne.
    fig.add_hline(
        y=MAJORITY_LINE,
        line=dict(color="rgba(11,11,11,0.45)", width=1, dash="dash"),
        annotation_text="50 %",
        annotation_position="top left",
        annotation_font=dict(size=11, color="rgba(11,11,11,0.6)"),
    )

    # Date du jour, comme sur le graphique de premier tour : elle montre d'un coup
    # d'œil à quel point la dernière estimation est ancienne.
    today = pd.Timestamp.today().normalize()
    fig.add_trace(
        go.Scatter(
            x=[today, today],
            y=[0, 100],
            mode="lines",
            line=dict(color="rgba(11,11,11,0.55)", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(role="today"),
        )
    )
    fig.add_annotation(
        x=today,
        yref="paper",
        y=1.0,
        text="aujourd'hui",
        showarrow=False,
        xanchor="right",
        xshift=-4,
        yanchor="top",
        font=dict(size=10, color="rgba(11,11,11,0.6)"),
    )

    low = min(all_values) if all_values else 30.0
    high = max(all_values) if all_values else 70.0
    # L'axe encadre toujours le seuil de 50 %, sinon on perd la référence, et reste
    # serré autour des valeurs observées : sur un duel, deux points d'écart comptent.
    y_range = [max(0.0, min(low, MAJORITY_LINE) - 3), min(100.0, max(high, MAJORITY_LINE) + 3)]

    subtitle = f"{duel['n_sondages']} sondage{'s' if duel['n_sondages'] > 1 else ''}"
    subtitle += f" · dernier le {pd.to_datetime(duel['derniere_date']):%d/%m/%Y}"
    subtitle += f" · {METHOD_SUBTITLES.get(method, '')}"
    if hypothese:
        subtitle += f" · {hypothese}"

    fig.update_layout(
        title=dict(
            text=f"<b>Second tour — {duel['label']}</b><br><sub>{subtitle}</sub>",
            x=0.5,
            xanchor="center",
        ),
        width=None,
        height=None,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=110, b=50),
    )
    fig.update_yaxes(title="Intention de vote (%)", range=y_range)
    fig.update_xaxes(title="Date de fin d'enquête", type="date")

    return fig
