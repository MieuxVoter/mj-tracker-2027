"""
Single-Member Plurality Voting Data loader for 2027 presidential election.
Supports JSON format from presidentielle2027.json.
"""

import json

import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from ..constants import CANDIDATS

SOURCE_URL = "https://raw.githubusercontent.com/MieuxVoter/presidentielle2027/refs/heads/main/presidentielle2027.json"


class SMPData:
    """
    Single-Member Plurality Voting Data loader for presidential election 2027.

    Loads data from presidentielle2027.json (or custom source) and creates
    aggregated intentions with rolling averages.

    Attributes
    ----------
    source : str
        Path or URL to the data source (JSON format).
    df_raw : pd.DataFrame
        The raw data from the JSON file, flattened to DataFrame.
    df_treated : pd.DataFrame
        The dataframe with the data treated (moving average, etc.).
    aggregated_data : dict
        The aggregated data structure with rolling averages.
    output_file : Path
        Path where aggregated JSON data is saved.

    Methods
    -------
    get_ranks() -> pd.DataFrame
        Load the uninomial ranks into a nice dataframe.
    get_intentions() -> pd.DataFrame
        Load the uninomial intentions into a nice dataframe.
    save_aggregated_data(output_file: str = None)
        Save aggregated data to JSON file.

    Examples
    --------
    >>> smp = SMPData()  # Load from default source
    >>> df_ranks = smp.get_ranks()
    >>> df_intentions = smp.get_intentions()
    """

    def __init__(
        self,
        min_date: str = "2024-01-01",
        rolling_window: str = "14d",
        output_dir: Optional[Path] = None,
        source_file: Optional[str] = None,
    ):
        """
        Initialize the SMPData loader.

        Parameters
        ----------
        source_file : str, optional
            Path or URL to the JSON file. If None, uses presidentielle2027.json
            from GitHub repository.
        min_date : str, default="2024-01-01"
            Minimum date to include in the dataset (format: YYYY-MM-DD).
        rolling_window : str, default="10d"
            Window size for rolling average calculation (pandas offset string).
        output_dir : Path, optional
            Directory where to save aggregated JSON. If None, uses parent directory.
        """
        # Set default source
        if source_file is None:
            source_file = SOURCE_URL

        self.source = source_file
        self.min_date = min_date
        self.rolling_window = rolling_window

        # Set output directory
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent  # Project root
        self.output_file = output_dir / "intentionsCandidatsMoyenneMobile14Jours_2027.json"

        print(f"Loading SMP data from {self.source}")

        # Load data
        self.df_raw = self._load_data()
        self.df_treated = None
        self.aggregated_data = None

        # Process data
        self._treatement()

    def _load_data(self) -> pd.DataFrame:
        """
        Load and flatten JSON data into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Flattened dataframe with one row per candidate per poll.
        """
        # Load JSON data
        if self.source.startswith("http"):
            import requests

            response = requests.get(self.source, timeout=10)  # 10 second timeout
            response.raise_for_status()
            data = response.json()
        else:
            with open(self.source, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Flatten the JSON structure to DataFrame
        rows = []
        for poll in data:
            print("poll", poll)
            poll_id = poll.get("poll_id", "")
            fin_enquete = poll.get("fin_enquete", "")
            debut_enquete = poll.get("debut_enquete", "")
            institut = poll.get("institut", "")
            commanditaire = poll.get("commanditaire", "")
            echantillon = poll.get("echantillon", None)
            tour = poll.get("tour", "")

            for candidat_data in poll.get("candidats", []):
                row = {
                    "poll_id": poll_id,
                    "fin_enquete": fin_enquete,
                    "debut_enquete": debut_enquete,
                    "end_date": fin_enquete,  # Alias for compatibility
                    "institut": institut,
                    "commanditaire": commanditaire,
                    "echantillon": echantillon,
                    "tour": tour,
                    "candidat": candidat_data.get("candidat", ""),
                    "intentions": candidat_data.get("intentions", None),
                    "retrait_candidature": candidat_data.get("retrait_candidature", ""),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _treatement(self):
        """
        Process raw dataframe and calculate rolling averages.

        Creates aggregated data structure with:
        - Rolling averages (10-day window by default)
        - Standard deviations
        - Error margins
        - Latest intentions per candidate
        """
        df = self.df_raw.copy()

        # Filter for first round only
        df = df[df["tour"] == "1er Tour"]

        # Convert to datetime to avoid comparison errors with strings or NaNs
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        min_date_ts = pd.to_datetime(self.min_date)

        # Sort by end date
        df = df.sort_values(by="end_date")

        # Filter by minimum date
        df = df[df["end_date"] >= min_date_ts]

        # Create aggregated data structure
        dict_candidats = {}

        count = 0  # for debugging plots

        for candidat in CANDIDATS.keys():
            print(f"Processing candidate: {candidat}")

            df_temp = df[df["candidat"] == candidat].copy()

            # Filter out withdrawn candidates
            withdrawn = False
            if "retrait_candidature" in df_temp.columns:
                for val in df_temp["retrait_candidature"].dropna():
                    if str(val).strip() and str(val).strip().lower() not in ["nan", "none"]:
                        withdrawn = True
                        break
            if withdrawn:
                print(f"  ! Candidate {candidat} has withdrawn. Skipping.")
                continue

            count += 1  # for debugging plots
            if df_temp.empty:
                continue

            df_temp_rolling, df_temp_rolling_ci, df_temp_rolling_spread = weighted_resample_and_rolling(
                df_temp, window=self.rolling_window
            )

            df_temp.index = pd.to_datetime(df_temp["end_date"])

            # Inner band: 95% confidence interval of the estimate (sampling uncertainty)
            erreur_inf = (df_temp_rolling.values - df_temp_rolling_ci.values).tolist()
            erreur_sup = (df_temp_rolling.values + df_temp_rolling_ci.values).tolist()

            # Outer band: weighted between-poll dispersion (how much pollsters disagree)
            erreur_inf_spread = (df_temp_rolling.values - df_temp_rolling_spread.values).tolist()
            erreur_sup_spread = (df_temp_rolling.values + df_temp_rolling_spread.values).tolist()

            # plot the difference between df_temp_rolling and df_temp_rolling_old for debugging
            # import matplotlib.pyplot as plt
            #
            # if count == 1:
            #     plt.figure(figsize=(10, 5))
            # plt.plot(df_temp.index, df_temp["intentions"], "o", label="Raw Intentions")
            # plt.plot(df_temp_rolling.index, df_temp_rolling, "-o", label="New Rolling Mean", ms=1)
            #
            # # margins
            # plt.fill_between(
            #     df_temp_rolling.index,
            #     erreur_inf,
            #     erreur_sup,
            #     color="blue",
            #     alpha=0.2,
            #     label="New Error Margin",
            # )
            #
            # plt.title(f"Rolling Mean Comparison for {candidat}")
            # plt.xlabel("Date")
            # plt.ylabel("Intentions")
            # plt.legend()
            #
            # if count == 4:
            #     plt.show()

            # Build data structure for this candidate
            dict_candidats[candidat] = {
                "intentions_moy_14d": {
                    "end_date": df_temp_rolling.index.strftime("%Y-%m-%d").to_list(),
                    "valeur": df_temp_rolling.values.tolist(),
                    "ci": df_temp_rolling_ci.values.tolist(),
                    "std": df_temp_rolling_spread.values.tolist(),
                    "erreur_inf": erreur_inf,
                    "erreur_sup": erreur_sup,
                    "erreur_inf_spread": erreur_inf_spread,
                    "erreur_sup_spread": erreur_sup_spread,
                },
                "intentions": {
                    "fin_enquete": df_temp.index.strftime("%Y-%m-%d").to_list(),
                    "valeur": df_temp.intentions.to_list(),
                    "commanditaire": df_temp["commanditaire"].to_list(),
                    "institut": df_temp["institut"].to_list(),
                },
                "derniers_sondages": [],
                "couleur": CANDIDATS[candidat]["couleur"],
            }

        # Create final aggregated structure
        self.aggregated_data = {
            "dernier_sondage": df["fin_enquete"].max(),
            "mise_a_jour": datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M"),
            "candidats": dict_candidats,
        }

        # Save to JSON file
        self.save_aggregated_data()

        # Load back as df_treated for compatibility
        self.df_treated = pd.read_json(self.output_file)

    def save_aggregated_data(self, output_file: Optional[str] = None):
        """
        Save aggregated data to JSON file.

        Parameters
        ----------
        output_file : str, optional
            Path to output file. If None, uses self.output_file.
        """
        if output_file is None:
            output_file = self.output_file
        else:
            output_file = Path(output_file)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.aggregated_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Aggregated SMP data saved to {output_file}")

    def _read_aggregated_data(self) -> Dict[str, Any]:
        """
        Read saved aggregated JSON data.

        Returns
        -------
        dict
            The aggregated data structure.
        """
        if self.aggregated_data is not None:
            return self.aggregated_data

        with open(self.output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_ranks(self) -> pd.DataFrame:
        """
        Load candidate rankings based on rolling average intentions.

        Calculates ranks for each date based on the rolling average values.
        Fills missing dates between first and last appearance of each candidate.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - candidat: Candidate name
            - fin_enquete: Survey end date
            - valeur: Rolling average intention value
            - rang: Rank (1 = highest intention)
            - erreur_sup: Upper error margin
            - erreur_inf: Lower error margin
        """
        data = self._read_aggregated_data()

        # Create dataframe from aggregated data
        df_rank_smp = pd.DataFrame(
            columns=[
                "candidat",
                "fin_enquete",
                "valeur",
                "rang",
                "erreur_sup",
                "erreur_inf",
                "erreur_sup_spread",
                "erreur_inf_spread",
            ]
        )

        for candidat, candidat_data in data.get("candidats", {}).items():
            dict_moy = candidat_data.get("intentions_moy_14d", {})
            n_pts = len(dict_moy.get("end_date", []))
            spread_sup = dict_moy.get("erreur_sup_spread", [None] * n_pts)
            spread_inf = dict_moy.get("erreur_inf_spread", [None] * n_pts)

            for d, v, inf, sup, inf_s, sup_s in zip(
                dict_moy.get("end_date", []),
                dict_moy.get("valeur", []),
                dict_moy.get("erreur_inf", []),
                dict_moy.get("erreur_sup", []),
                spread_inf,
                spread_sup,
            ):
                row_to_add = {
                    "candidat": candidat,
                    "fin_enquete": d,
                    "valeur": v,
                    "rang": None,
                    "erreur_sup": sup,
                    "erreur_inf": inf,
                    "erreur_sup_spread": sup_s,
                    "erreur_inf_spread": inf_s,
                }
                df_rank_smp = pd.concat([df_rank_smp, pd.DataFrame([row_to_add])], ignore_index=True)

        # NOTE: We intentionally do NOT forward-fill missing dates here. A previous
        # version stamped each candidate's last known value across every date where
        # *any other* candidate had a poll (a zero-order hold). That fabricated long
        # flat plateaus — e.g. a single Ruffin poll drawn as a constant line for two
        # months — implying support was *known* and *unchanged* when it was simply
        # unmeasured. Gaps are now left empty and handled at plot time by segment /
        # gap detection (see DEFAULT_MAX_GAP_DAYS in plots_smp_intentions).

        # Remove duplicates (keep last entry per candidate per date)
        df_rank_smp = df_rank_smp.sort_values(by=["fin_enquete", "candidat"])
        df_rank_smp = df_rank_smp.drop_duplicates(subset=["fin_enquete", "candidat"], keep="last")

        # Compute ranks: sort by date and value (descending)
        df_rank_smp = df_rank_smp.sort_values(by=["fin_enquete", "valeur"], ascending=(True, False))

        dates = df_rank_smp["fin_enquete"].unique()
        for d in dates:
            df_date = df_rank_smp[df_rank_smp["fin_enquete"] == d]
            nb_candidates = len(df_date)
            index_row = df_date.index
            df_rank_smp.loc[index_row, "rang"] = [i + 1 for i in range(nb_candidates)]

        # Add aliases for compatibility with plotting functions
        df_rank_smp["end_date"] = df_rank_smp["fin_enquete"]
        df_rank_smp["candidate"] = df_rank_smp["candidat"]

        # Convert dates to datetime for proper plotting
        df_rank_smp["fin_enquete"] = pd.to_datetime(df_rank_smp["fin_enquete"])
        df_rank_smp["end_date"] = pd.to_datetime(df_rank_smp["end_date"])

        return df_rank_smp

    def get_intentions(self) -> pd.DataFrame:
        """
        Load raw intention data for each candidate.

        Returns the non-aggregated, original intention values from polls.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - candidat: Candidate name
            - fin_enquete: Survey end date
            - intentions: Raw intention value from poll
        """
        data = self._read_aggregated_data()

        # Create dataframe
        df_smp_data = pd.DataFrame(columns=["candidat", "fin_enquete", "intentions"])

        for candidat, candidat_data in data.get("candidats", {}).items():
            intentions_data = candidat_data.get("intentions", {})

            for d, i in zip(intentions_data.get("fin_enquete", []), intentions_data.get("valeur", [])):
                row_to_add = {
                    "candidat": candidat,
                    "fin_enquete": d,
                    "intentions": i,
                }
                df_smp_data = pd.concat([df_smp_data, pd.DataFrame([row_to_add])], ignore_index=True)

        # Add aliases for compatibility with plotting functions
        df_smp_data["end_date"] = df_smp_data["fin_enquete"]
        df_smp_data["candidate"] = df_smp_data["candidat"]

        # Convert dates to datetime for proper plotting
        df_smp_data["fin_enquete"] = pd.to_datetime(df_smp_data["fin_enquete"])
        df_smp_data["end_date"] = pd.to_datetime(df_smp_data["end_date"])

        return df_smp_data


def weighted_resample_and_rolling(df_temp, window="14d", default_sample=1000):
    """
    Inverse-variance weighted, time-aware rolling aggregation of poll intentions.

    Each poll ``i`` reports an intention ``p_i`` (in %) measured on a sample of
    size ``n_i``. The binomial sampling variance of that estimate is
    ``var_i = p_i (1 - p_i) / n_i``. Combining polls in a trailing time ``window``
    with inverse-variance weights ``w_i = 1 / var_i`` yields the minimum-variance
    unbiased linear estimate of the underlying support, and its standard error has
    a closed form. Estimates are produced *at the actual poll dates* and connected
    with straight segments at plot time, which avoids the S-shaped "plateau" bias a
    daily Gaussian-kernel smoother introduces around sparse polls and cluster edges.

    Parameters
    ----------
    df_temp : pd.DataFrame
        Rows for a single candidate, with ``end_date``, ``intentions`` (%), and
        ``echantillon`` (sample size) columns.
    window : str, default="14d"
        Trailing time window (pandas offset string).
    default_sample : int, default=1000
        Sample size assumed when ``echantillon`` is missing.

    Returns
    -------
    mean : pd.Series
        Inverse-variance weighted rolling mean (%), indexed by date.
    ci : pd.Series
        95% half-width of the *estimate* (sampling uncertainty, %). Never
        collapses to zero: a lone small poll still carries its own sampling error.
    spread : pd.Series
        Weighted between-poll standard deviation (%), i.e. how much the pollsters
        in the window disagree. Goes to zero when a single poll is in the window.
    """
    df = df_temp.copy()
    df.index = pd.to_datetime(df["end_date"])
    df = df.sort_index()

    p = pd.to_numeric(df["intentions"], errors="coerce") / 100.0
    n = pd.to_numeric(df.get("echantillon"), errors="coerce").fillna(default_sample).clip(lower=1)

    # Binomial sampling variance of each poll; floored to avoid division by zero
    # at the p = 0 / p = 1 boundaries.
    var_i = (p * (1.0 - p) / n).clip(lower=1e-9)
    w = 1.0 / var_i

    frame = pd.DataFrame({"w": w, "wp": w * p, "wp2": w * p * p}).dropna()
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    roll = frame.rolling(window)
    s_w = roll["w"].sum()
    s_wp = roll["wp"].sum()
    s_wp2 = roll["wp2"].sum()

    mean = s_wp / s_w
    # Standard error of the inverse-variance weighted mean: Var(mean) = 1 / sum(w).
    ci = 1.96 * (1.0 / s_w) ** 0.5
    # Weighted between-poll variance (dispersion of the poll cloud itself).
    between_var = (s_wp2 / s_w - mean**2).clip(lower=0.0)
    spread = between_var**0.5

    # Convert back to percentage points.
    return mean * 100.0, ci * 100.0, spread * 100.0
