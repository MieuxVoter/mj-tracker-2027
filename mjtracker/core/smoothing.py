"""
Méthodes de lissage des séries d'intentions de vote.

Toutes les fonctions de ce module partagent le **même contrat** que
``weighted_resample_and_rolling`` (dans ``smp_data``), la méthode historique :

    smoother(df_temp, ...) -> (mean, ci, spread)

trois ``pd.Series`` indexées par date, exprimées en points de pourcentage :

- ``mean``   : l'estimation lissée du soutien ;
- ``ci``     : demi-largeur de l'intervalle à 95 % de *l'estimation* (précision) ;
- ``spread`` : dispersion pondérée entre sondages au voisinage (désaccord des instituts).

Elles sont donc interchangeables dans ``SMPData._treatement`` via le registre
``SMOOTHERS`` en bas de fichier.

Choix de la méthode
-------------------
Un backtest hors échantillon (778 prédictions : chaque méthode doit prédire le
sondage suivant à partir des seuls sondages antérieurs) donne, en erreur de
prédiction hors bruit d'échantillonnage et en rugosité de la courbe :

    Kalman          2,84 pts   0,002      <- meilleur sur les deux critères
    Noyau gaussien  3,08 pts   0,010
    Moyenne mobile  3,09 pts   0,013      <- méthode historique
    LOESS           3,11 pts   0,038

La spline pénalisée a été écartée : elle ne s'extrapole pas (au-delà du dernier
sondage, une spline cubique part à la dérive), or c'est précisément la valeur de
bout de courbe qui est affichée.

Attention : aucune de ces méthodes n'atteint la couverture nominale de 95 %
(mesurée entre 58 et 62 %). Il manque ~3 points d'écart-type d'effets d'institut
et d'hétérogénéité des scénarios, que le lissage ne peut pas corriger.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_SAMPLE = 1000

# Pas de la grille d'évaluation. Un pas quotidien multiplierait par sept le poids
# des figures exportées sans rien apporter de visible sur deux ans et demi.
DEFAULT_STEP_DAYS = 7

# Au-delà de cette distance au sondage le plus proche, on ne trace plus : sans ce
# masquage on dessinerait une certitude qu'aucun sondage ne porte. C'est la même
# règle que la détection de segments côté tracé (DEFAULT_MAX_GAP_DAYS).
DEFAULT_MAX_GAP_DAYS = 21

# Bande passante utilisée pour estimer la dispersion locale entre instituts.
SPREAD_BANDWIDTH_DAYS = 14.0

SeriesTriplet = Tuple[pd.Series, pd.Series, pd.Series]


def _prepare(df_temp: pd.DataFrame, default_sample: int = DEFAULT_SAMPLE):
    """
    Extrait (dates, x en jours, y en %, variance d'échantillonnage) d'un candidat.

    La variance binomiale de chaque sondage est ``p(100 - p) / n`` en points de
    pourcentage — identique à celle utilisée par la méthode historique.
    """
    df = df_temp.copy()
    df.index = pd.to_datetime(df["end_date"])
    df = df.sort_index()

    y = pd.to_numeric(df["intentions"], errors="coerce")
    n = pd.to_numeric(df.get("echantillon"), errors="coerce").fillna(default_sample).clip(lower=1)

    keep = y.notna()
    y, n, dates = y[keep], n[keep], df.index[keep]

    y = y.to_numpy(dtype=float)
    n = n.to_numpy(dtype=float)
    var = np.clip(y * (100.0 - y) / n, 1e-6, None)
    x = (dates - dates[0]).total_seconds().to_numpy() / 86400.0 if len(dates) else np.array([])

    return dates, x, y, var


def _build_grid(
    dates: pd.DatetimeIndex,
    x: np.ndarray,
    anchor: Optional[pd.Timestamp],
    step_days: int,
) -> np.ndarray:
    """
    Grille d'évaluation régulière, **alignée sur une origine commune**.

    L'ancrage est ce qui permet à tous les candidats de partager les mêmes dates :
    sans lui, chaque courbe tomberait sur ses propres dates et le classement par
    date calculé en aval n'aurait plus de sens.
    """
    offset = 0.0
    if anchor is not None:
        offset = (dates[0] - pd.Timestamp(anchor)).total_seconds() / 86400.0
        offset = offset % step_days

    grid = np.arange(-offset, x.max() + step_days, step_days)
    grid = grid[(grid >= x.min() - 1e-9) & (grid <= x.max() + 1e-9)]

    # La dernière date de sondage est toujours évaluée : c'est la valeur affichée
    # en bout de courbe, elle ne doit pas dépendre du pas de grille.
    if grid.size == 0 or not np.isclose(grid[-1], x.max()):
        grid = np.r_[grid, x.max()]

    return np.unique(grid)


def _local_spread(x: np.ndarray, y: np.ndarray, grid: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Dispersion des sondages autour de la courbe, au voisinage de chaque point de grille."""
    out = np.full(grid.size, np.nan)
    for i, t in enumerate(grid):
        w = np.exp(-0.5 * ((x - t) / SPREAD_BANDWIDTH_DAYS) ** 2)
        total = w.sum()
        if total <= 0 or not np.isfinite(fitted[i]):
            continue
        out[i] = np.sqrt(np.sum(w * (y - fitted[i]) ** 2) / total)
    return out


def _mask_far_from_data(x: np.ndarray, grid: np.ndarray, max_gap_days: float) -> np.ndarray:
    """True là où la grille est trop loin de tout sondage."""
    return np.min(np.abs(grid[:, None] - x[None, :]), axis=1) > max_gap_days


def _finalize(
    dates: pd.DatetimeIndex,
    grid: np.ndarray,
    mean: np.ndarray,
    ci: np.ndarray,
    spread: np.ndarray,
) -> SeriesTriplet:
    """Convertit les tableaux en séries datées et retire les points non estimés."""
    index = pd.DatetimeIndex(dates[0] + pd.to_timedelta(grid, unit="D"))
    valid = np.isfinite(mean)

    mean_s = pd.Series(mean[valid], index=index[valid])
    ci_s = pd.Series(np.nan_to_num(ci[valid], nan=0.0), index=index[valid])
    spread_s = pd.Series(np.nan_to_num(spread[valid], nan=0.0), index=index[valid])

    return mean_s, ci_s, spread_s


def _empty_triplet() -> SeriesTriplet:
    empty = pd.Series(dtype=float)
    return empty, empty, empty


def loess_smooth(
    df_temp: pd.DataFrame,
    h_days: float = 30.0,
    anchor: Optional[pd.Timestamp] = None,
    step_days: int = DEFAULT_STEP_DAYS,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
    default_sample: int = DEFAULT_SAMPLE,
) -> SeriesTriplet:
    """
    LOESS à fenêtre **temporelle** fixe : régression locale linéaire, noyau tricube.

    La fenêtre est en jours et non en fraction de points (le ``frac`` de
    ``statsmodels``) : les sondages sont trop irrégulièrement espacés pour qu'une
    fraction de points corresponde à une durée stable.

    Le poids de chaque sondage est le produit (tricube dans le temps) × (inverse de
    sa variance) : proche *et* précis pèse le plus. Le degré 1 évite l'aplatissement
    aux bords, ce qui compte ici puisque la dernière valeur est celle qu'on lit.
    """
    dates, x, y, var = _prepare(df_temp, default_sample)
    if x.size == 0:
        return _empty_triplet()

    grid = _build_grid(dates, x, anchor, step_days)
    mean = np.full(grid.size, np.nan)
    ci = np.full(grid.size, np.nan)

    for i, t in enumerate(grid):
        u = np.abs(x - t) / h_days
        win = u < 1.0
        if win.sum() < 1:
            continue

        w = (1.0 - u[win] ** 3) ** 3 / var[win]
        xs = x[win] - t
        ys, vs = y[win], var[win]

        # Garde-fou indispensable : dans une grappe de sondages quasi simultanés la
        # pente locale n'est pas identifiable et le local-linéaire extrapole
        # n'importe quoi (jusqu'à -20 % observé). On retombe alors au degré 0.
        w_mean_x = np.sum(w * xs) / np.sum(w)
        spread_x = np.sqrt(np.sum(w * (xs - w_mean_x) ** 2) / np.sum(w))
        degree = 1 if (np.unique(xs).size >= 3 and spread_x > 0.15 * h_days) else 0

        design = np.vander(xs, degree + 1, increasing=True)
        weighted = design.T * w
        normal = weighted @ design
        if degree > 0 and np.linalg.cond(normal) > 1e8:
            design = np.ones((xs.size, 1))
            weighted = design.T * w
            normal = weighted @ design

        try:
            weights = np.linalg.solve(normal, weighted)[0]
        except np.linalg.LinAlgError:
            continue

        mean[i] = float(weights @ ys)
        ci[i] = 1.96 * float(np.sqrt(np.sum(weights**2 * vs)))

    mean[_mask_far_from_data(x, grid, max_gap_days)] = np.nan
    return _finalize(dates, grid, mean, ci, _local_spread(x, y, grid, mean))


def gaussian_kernel_smooth(
    df_temp: pd.DataFrame,
    h_days: float = 10.0,
    anchor: Optional[pd.Timestamp] = None,
    step_days: int = DEFAULT_STEP_DAYS,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
    default_sample: int = DEFAULT_SAMPLE,
) -> SeriesTriplet:
    """
    Noyau gaussien centré (Nadaraya-Watson) : moyenne locale pondérée.

    Plus régulier que LOESS et sans risque de sur-extrapolation, mais biaisé dans
    les pentes et aux extrémités — il tire la courbe vers la moyenne locale.
    """
    dates, x, y, var = _prepare(df_temp, default_sample)
    if x.size == 0:
        return _empty_triplet()

    grid = _build_grid(dates, x, anchor, step_days)
    mean = np.full(grid.size, np.nan)
    ci = np.full(grid.size, np.nan)

    for i, t in enumerate(grid):
        d = np.abs(x - t)
        near = d < 3.0 * h_days
        if not near.any():
            continue
        w = np.exp(-0.5 * (d[near] / h_days) ** 2) / var[near]
        weights = w / w.sum()
        mean[i] = float(weights @ y[near])
        ci[i] = 1.96 * float(np.sqrt(np.sum(weights**2 * var[near])))

    mean[_mask_far_from_data(x, grid, max_gap_days)] = np.nan
    return _finalize(dates, grid, mean, ci, _local_spread(x, y, grid, mean))


def _daily_inverse_variance(x: np.ndarray, y: np.ndarray, var: np.ndarray):
    """Agrège en une observation par jour (plusieurs instituts peuvent publier le même jour)."""
    days = np.unique(x)
    values = np.empty(days.size)
    variances = np.empty(days.size)
    for i, t in enumerate(days):
        m = x == t
        w = 1.0 / var[m]
        values[i] = np.sum(w * y[m]) / np.sum(w)
        variances[i] = 1.0 / np.sum(w)
    return days, values, variances


def kalman_smooth(
    df_temp: pd.DataFrame,
    sigma_day: float = 0.10,
    anchor: Optional[pd.Timestamp] = None,
    step_days: int = DEFAULT_STEP_DAYS,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
    default_sample: int = DEFAULT_SAMPLE,
) -> SeriesTriplet:
    """
    Modèle à niveau local : filtre de Kalman puis lisseur RTS.

    Le vrai niveau de soutien est supposé suivre une marche aléatoire, dont chaque
    sondage est une mesure bruitée de variance connue. C'est le modèle le plus
    adapté au problème : aucune fenêtre arbitraire, pondération automatique par la
    précision de chaque sondage, et une incertitude qui s'élargit d'elle-même
    lorsque les sondages se raréfient.

    ``sigma_day`` est le seul réglage : l'écart-type du mouvement réel de l'opinion
    par jour, en points. Plus il est grand, plus la courbe colle aux sondages.
    """
    dates, x, y, var = _prepare(df_temp, default_sample)
    if x.size == 0:
        return _empty_triplet()

    grid = _build_grid(dates, x, anchor, step_days)

    # Le filtre tourne au pas quotidien, indépendamment du pas d'affichage : la
    # dynamique de la marche aléatoire ne doit pas dépendre de la grille choisie.
    days = np.arange(np.floor(x.min()), np.ceil(x.max()) + 1.0)
    obs_x, obs_y, obs_var = _daily_inverse_variance(x, y, var)
    observations = {int(round(t)): (obs_y[i], obs_var[i]) for i, t in enumerate(obs_x)}

    q = sigma_day**2
    n = days.size
    a_pred = np.empty(n)
    p_pred = np.empty(n)
    a_filt = np.empty(n)
    p_filt = np.empty(n)

    a, p = float(obs_y[0]), 100.0  # a priori volontairement large
    for i, t in enumerate(days):
        if i > 0:
            p += q  # un jour de marche aléatoire
        a_pred[i], p_pred[i] = a, p

        key = int(round(t))
        if key in observations:
            value, r = observations[key]
            gain = p / (p + r)
            a += gain * (value - a)
            p *= 1.0 - gain
        a_filt[i], p_filt[i] = a, p

    # Lisseur RTS : rétro-propagation de l'information du futur vers le passé.
    a_smooth, p_smooth = a_filt.copy(), p_filt.copy()
    for i in range(n - 2, -1, -1):
        c = p_filt[i] / p_pred[i + 1]
        a_smooth[i] = a_filt[i] + c * (a_smooth[i + 1] - a_pred[i + 1])
        p_smooth[i] = p_filt[i] + c**2 * (p_smooth[i + 1] - p_pred[i + 1])

    mean = np.interp(grid, days, a_smooth)
    ci = 1.96 * np.sqrt(np.clip(np.interp(grid, days, p_smooth), 0.0, None))

    mean[_mask_far_from_data(x, grid, max_gap_days)] = np.nan
    return _finalize(dates, grid, mean, ci, _local_spread(x, y, grid, mean))


def rolling_smooth(
    df_temp: pd.DataFrame,
    window: str = "14d",
    anchor: Optional[pd.Timestamp] = None,
    default_sample: int = DEFAULT_SAMPLE,
    **_ignored,
) -> SeriesTriplet:
    """
    Méthode historique : moyenne mobile traînante pondérée par l'inverse-variance.

    Simple enveloppe autour de ``weighted_resample_and_rolling`` pour lui donner la
    signature commune du registre. ``anchor`` est ignoré : cette méthode est évaluée
    aux dates de sondage, pas sur une grille.
    """
    from .smp_data import weighted_resample_and_rolling

    return weighted_resample_and_rolling(df_temp, window=window, default_sample=default_sample)


# Registre unique consommé par SMPData, les scripts d'export et le site.
# L'ordre est celui des boutons affichés.
SMOOTHERS: dict[str, tuple[str, Callable[..., SeriesTriplet]]] = {
    "kalman": ("Kalman", kalman_smooth),
    "loess": ("LOESS 30 j", loess_smooth),
    "kernel": ("Noyau gaussien 10 j", gaussian_kernel_smooth),
    "rolling": ("Moyenne mobile 14 j", rolling_smooth),
}

DEFAULT_METHOD = "kalman"

# Sous-titre affiché sur la figure pour chaque méthode.
METHOD_SUBTITLES = {
    "kalman": "Modèle à niveau local (filtre de Kalman)",
    "loess": "Régression locale LOESS (fenêtre 30 jours)",
    "kernel": "Noyau gaussien (bande passante 10 jours)",
    "rolling": "Moyenne mobile sur 14 jours",
}
