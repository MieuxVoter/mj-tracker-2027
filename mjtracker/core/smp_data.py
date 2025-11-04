"""
Single-Member Plurality Voting Data loader for 2027 presidential election.
Supports JSON format from presidentielle2027.json.
"""

import json
import pandas as pd
import datetime
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

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
        rolling_window: str = "10d",
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
            response = requests.get(self.source)
            response.raise_for_status()
            data = response.json()
        else:
            with open(self.source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Flatten the JSON structure to DataFrame
        rows = []
        for poll in data:
            poll_id = poll.get('poll_id', '')
            fin_enquete = poll.get('fin_enquete', '')
            debut_enquete = poll.get('debut_enquete', '')
            institut = poll.get('institut', '')
            commanditaire = poll.get('commanditaire', '')
            echantillon = poll.get('echantillon', None)
            tour = poll.get('tour', '')
            
            for candidat_data in poll.get('candidats', []):
                row = {
                    'poll_id': poll_id,
                    'fin_enquete': fin_enquete,
                    'debut_enquete': debut_enquete,
                    'end_date': fin_enquete,  # Alias for compatibility
                    'institut': institut,
                    'commanditaire': commanditaire,
                    'echantillon': echantillon,
                    'tour': tour,
                    'candidat': candidat_data.get('candidat', ''),
                    'intentions': candidat_data.get('intentions', None),
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
        
        # Sort by end date
        df = df.sort_values(by="end_date")
        
        # Filter by minimum date
        df = df[df["end_date"] >= self.min_date]
        
        # Create aggregated data structure
        dict_candidats = {}
        derniere_intention = pd.DataFrame()
        
        for candidat in CANDIDATS.keys():
            df_temp = df[df["candidat"] == candidat].copy()
            
            if df_temp.empty:
                continue
            
            # Set index to datetime for rolling operations
            df_temp.index = pd.to_datetime(df_temp["end_date"])
            
            # Calculate rolling averages (no lookahead bias)
            df_temp_rolling = (
                df_temp[["intentions"]]
                .rolling(self.rolling_window, min_periods=1)
                .mean()
            )
            
            df_temp_rolling_std = (
                df_temp[["intentions"]]
                .rolling(self.rolling_window, min_periods=1)
                .std()
            )
            
            # Fill NaN std with 0
            df_temp_rolling_std = df_temp_rolling_std.fillna(0)
            
            # Calculate error margins (±1 std)
            erreur_inf = (df_temp_rolling.intentions - df_temp_rolling_std.intentions).tolist()
            erreur_sup = (df_temp_rolling.intentions + df_temp_rolling_std.intentions).tolist()
            
            # Store last intention
            if not df_temp_rolling.empty:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    derniere_intention = pd.concat([
                        derniere_intention,
                        pd.DataFrame({
                            "candidat": [candidat],
                            "intentions": [df_temp_rolling.intentions.tolist()[-1]]
                        })
                    ], ignore_index=True)
            
            # Build data structure for this candidate
            dict_candidats[candidat] = {
                "intentions_moy_14d": {
                    "end_date": df_temp_rolling.index.strftime("%Y-%m-%d").to_list(),
                    "valeur": df_temp_rolling.intentions.to_list(),
                    "std": df_temp_rolling_std.intentions.to_list(),
                    "erreur_inf": erreur_inf,
                    "erreur_sup": erreur_sup,
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
        
        with open(output_file, "w", encoding='utf-8') as f:
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
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
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
            columns=["candidat", "fin_enquete", "valeur", "rang", "erreur_sup", "erreur_inf"]
        )
        
        for candidat, candidat_data in data.get("candidats", {}).items():
            dict_moy = candidat_data.get("intentions_moy_14d", {})
            
            for d, v, sup, inf in zip(
                dict_moy.get("end_date", []),
                dict_moy.get("valeur", []),
                dict_moy.get("erreur_inf", []),
                dict_moy.get("erreur_sup", []),
            ):
                row_to_add = {
                    "candidat": candidat,
                    "fin_enquete": d,
                    "valeur": v,
                    "rang": None,
                    "erreur_sup": sup,
                    "erreur_inf": inf,
                }
                df_rank_smp = pd.concat(
                    [df_rank_smp, pd.DataFrame([row_to_add])],
                    ignore_index=True
                )

        # Fill dates without values for some candidates (interpolation)
        for c in df_rank_smp["candidat"].unique():
            temp_df = df_rank_smp[df_rank_smp["candidat"] == c]
            date_min = temp_df["fin_enquete"].min()
            date_max = temp_df["fin_enquete"].max()
            
            for d in df_rank_smp["fin_enquete"].unique():
                if (d > date_min) and (d < date_max) and temp_df[temp_df["fin_enquete"] == d].empty:
                    idx = temp_df["fin_enquete"].searchsorted(d)
                    if idx > 0:
                        v = temp_df["valeur"].iloc[idx - 1]
                        sup = temp_df["erreur_sup"].iloc[idx - 1]
                        inf = temp_df["erreur_inf"].iloc[idx - 1]
                        row_to_add = {
                            "candidat": c,
                            "fin_enquete": d,
                            "valeur": v,
                            "rang": None,
                            "erreur_sup": sup,
                            "erreur_inf": inf,
                        }
                        df_rank_smp = pd.concat(
                            [df_rank_smp, pd.DataFrame([row_to_add])],
                            ignore_index=True
                        )

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
            
            for d, i in zip(
                intentions_data.get("fin_enquete", []),
                intentions_data.get("valeur", [])
            ):
                row_to_add = {
                    "candidat": candidat,
                    "fin_enquete": d,
                    "intentions": i,
                }
                df_smp_data = pd.concat(
                    [df_smp_data, pd.DataFrame([row_to_add])],
                    ignore_index=True
                )

        # Add aliases for compatibility with plotting functions
        df_smp_data["end_date"] = df_smp_data["fin_enquete"]
        df_smp_data["candidate"] = df_smp_data["candidat"]
        
        # Convert dates to datetime for proper plotting
        df_smp_data["fin_enquete"] = pd.to_datetime(df_smp_data["fin_enquete"])
        df_smp_data["end_date"] = pd.to_datetime(df_smp_data["end_date"])

        return df_smp_data
