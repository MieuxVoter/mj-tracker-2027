"""
Manages a collection of survey data, providing functionalities for loading, filtering,
aggregating, and applying Majority Judgment (MJ) analysis across multiple surveys.
It also includes methods for calculating time series like the average public mood.
"""

from functools import cached_property
from pathlib import Path
import requests
from io import StringIO
import urllib.parse
from typing import Union, List

import numpy as np
import pandas as pd

from .misc.enums import PollingOrganizations, AggregationMode, Candidacy, UntilRound
from .survey_interface import SurveyInterface

# Get the path to the current script
CURRENT_SCRIPT_PATH = Path(__file__).parent
STANDARDIZATION_CSV_PATH = CURRENT_SCRIPT_PATH / "standardisation.csv"
MAX_MENTIONS_IN_DATAFRAME = 7  # Maximum number of mentions in the dataframe, used for sanity checks


class SurveysInterface:
    def __init__(self, df: pd.DataFrame):
        # Ensure 'end_date' is a datetime column for consistent operations
        if "end_date" in df.columns:
            # Using errors='coerce' will turn unparseable dates into NaT (Not a Time)
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        self.df = df

    @classmethod
    def load_from_url(
        cls,
        url: str,
        candidates: Candidacy = None,
        polling_organization: PollingOrganizations = None,
        until_round: UntilRound = None,  # note: this is not implemented yet
    ) -> "SurveysInterface":
        """
        Loads survey data from a given URL.
        Automatically converts GitHub blob URLs to raw content URLs for direct download.

        Parameters
        ----------
        url : str
            The URL to the CSV data.
        candidates : Candidacy, optional
            Filter by candidate type (e.g., ALL, PRESIDENTIAL). Defaults to ALL.
        polling_organization : PollingOrganizations, optional
            Filter by a specific polling organization. Defaults to ALL.
        until_round : UntilRound, optional
            Filter data up to a specific round (not yet implemented).

        Returns
        -------
        SurveysInterface
            An instance of SurveysInterface populated with the loaded data.

        Raises
        ------
        IOError
            If there's an issue fetching data from the URL.
        """
        # Convert GitHub blob URL to raw content URL
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        try:
            # Add a timeout to prevent indefinite waiting
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to load data from URL '{url}': {e}") from e

        # Pass StringIO object directly to the load method
        return cls.load(
            csv_path=StringIO(response.text),
            candidates=candidates,
            polling_organization=polling_organization,
            until_round=until_round,
        )

    @classmethod
    def load(
        cls,
        csv_path: Union[str, Path, StringIO],
        candidates: Candidacy = None,
        polling_organization: PollingOrganizations = None,
        until_round: UntilRound = None,  # note: this is not implemented yet
    ) -> "SurveysInterface":
        """
        Loads survey data from a CSV file path (string or Path object) or a file-like object (StringIO).

        Parameters
        ----------
        csv_path : Union[str, Path, StringIO]
            The path to the CSV file or a file-like object containing CSV data.
        candidates : Candidacy, optional
            Filter by candidate type (e.g., ALL, PRESIDENTIAL). Defaults to ALL.
        polling_organization : PollingOrganizations, optional
            Filter by a specific polling organization. Defaults to ALL.
        until_round : UntilRound, optional
            Filter data up to a specific round (not yet implemented).

        Returns
        -------
        SurveysInterface
            An instance of SurveysInterface populated with the loaded data.

        Raises
        ------
        IOError
            If there's an issue reading the CSV file.
        """
        if candidates is None:
            candidates = Candidacy.ALL
        if polling_organization is None:
            polling_organization = PollingOrganizations.ALL

        try:
            df_surveys = pd.read_csv(
                csv_path,
                na_filter=False,  # Keep explicit empty strings or NaNs as they are, rather than converting to NaN
            )
        except Exception as e:
            raise IOError(f"Failed to load CSV from '{csv_path}': {e}") from e

        # Sanity conversions and type enforcement for intention columns
        for i in range(1, MAX_MENTIONS_IN_DATAFRAME + 1):
            col_name = f"intention_mention_{i}"
            if col_name in df_surveys.columns:
                # Convert to numeric, coercing errors to NaN. This is safer than failing.
                df_surveys[col_name] = pd.to_numeric(df_surveys[col_name], errors='coerce')
        
        # Convert 'nombre_mentions' to a nullable integer type
        if "nombre_mentions" in df_surveys.columns:
            df_surveys["nombre_mentions"] = pd.to_numeric(
                df_surveys["nombre_mentions"], errors='coerce'
            ).astype(pd.Int64Dtype()) # Allows for NaN (missing) integer values
        
        if polling_organization != PollingOrganizations.ALL:
            # Filter by polling organization, making comparison case-insensitive and handling potential non-string values
            df_surveys = df_surveys[
                df_surveys["institut"].astype(str).str.casefold() == polling_organization.value.casefold()
            ].copy() # Use .copy() to avoid SettingWithCopyWarning

        return cls(df=df_surveys)

    @cached_property
    def surveys(self) -> np.ndarray:
        """Returns an array of unique poll IDs present in the surveys."""
        return self.df["poll_id"].unique()

    @cached_property
    def nb_surveys(self) -> int:
        """Returns the number of unique surveys."""
        return len(self.surveys)

    @cached_property
    def candidates(self) -> List[str]:
        """Get the list of unique candidates in the surveys."""
        # Ensure values are treated as strings before getting unique list
        return self.df["candidate"].astype(str).unique().tolist()

    @cached_property
    def sponsors(self) -> List[str]:
        """Get the list of unique sponsors in the surveys."""
        return self.df["commanditaire"].astype(str).unique().tolist()

    @cached_property
    def sources(self) -> List[str]:
        """Get the list of unique sources (polling institutes) in the surveys."""
        return self.df["institut"].astype(str).unique().tolist()

    @cached_property
    def dates(self) -> List[pd.Timestamp]:
        """Get the list of unique end dates of polls, sorted chronologically."""
        # Ensure 'end_date' column is datetime and drop NaT values before getting unique and sorting
        unique_dates = self.df["end_date"].dropna().unique()
        return sorted(unique_dates.tolist())

    @cached_property
    def most_recent_date(self) -> Union[pd.Timestamp, None]:
        """Returns the most recent end date of a survey, or None if no valid dates."""
        dates = self.dates
        return dates[-1] if dates else None

    @cached_property
    def oldest_date(self) -> Union[pd.Timestamp, None]:
        """Returns the oldest end date of a survey, or None if no valid dates."""
        dates = self.dates
        return dates[0] if dates else None

    @cached_property
    def sponsors_string(self) -> str:
        """Get a comma-separated string of unique sponsors."""
        return ", ".join(self.sponsors)

    @cached_property
    def sources_string(self) -> str:
        """Get a comma-separated string of unique sources."""
        return ", ".join(self.sources)

    def select_survey(self, survey_id: str) -> SurveyInterface:
        """Selects a specific survey by its ID and returns it as a SurveyInterface object."""
        # Use .copy() to ensure the returned DataFrame is independent of the original
        return SurveyInterface(self.df[self.df["poll_id"] == survey_id].copy())

    @property
    def most_recent_survey(self) -> Union[SurveyInterface, None]:
        """Get the most recent survey(s) as a SurveyInterface object.
        If multiple polls share the most recent date, they are all included."""
        most_recent_date = self.most_recent_date
        if most_recent_date is None:
            return None
        # Return a SurveyInterface potentially containing multiple polls if they share the most recent date
        return SurveyInterface(self.df[self.df["end_date"] == most_recent_date].copy())

    def select_polling_organization(self, polling_organization: PollingOrganizations) -> "SurveysInterface":
        """
        Selects surveys from a specific polling organization.

        Parameters
        ----------
        polling_organization : PollingOrganizations
            The polling organization to filter by.

        Returns
        -------
        SurveysInterface
            A new SurveysInterface instance containing only the filtered surveys.
        """
        if polling_organization == PollingOrganizations.ALL:
            return self
        df_filtered = self.df[
            self.df["institut"].astype(str).str.casefold() == polling_organization.value.casefold()
        ].copy()
        return SurveysInterface(df=df_filtered)

    def select_candidate(self, candidate: str) -> "SurveysInterface":
        """
        Selects surveys for a specific candidate.

        Parameters
        ----------
        candidate : str
            The name of the candidate to filter by.

        Returns
        -------
        SurveysInterface
            A new SurveysInterface instance containing only the surveys for the specified candidate.
        """
        df_filtered = self.df[self.df["candidate"].astype(str).str.casefold() == candidate.casefold()].copy()
        return SurveysInterface(df=df_filtered)

    def to_no_opinion_surveys(self):
        """
        Converts all surveys to a state where 'no opinion' grades are removed,
        and other grades are renormalized. Modifies `self.df` in-place.
        """
        if self.df.empty:
            return # No data to process

        all_df_processed = []
        for survey_id in self.surveys:
            survey_df = self.df[self.df["poll_id"] == survey_id].copy()
            if not survey_df.empty:
                all_df_processed.append(SurveyInterface(survey_df).to_no_opinion_survey())
        
        if all_df_processed:
            self.df = pd.concat(all_df_processed, ignore_index=True)
        else:
            # If no surveys could be processed, reset self.df to an empty DataFrame with original columns
            self.df = pd.DataFrame(columns=self.df.columns)

    def aggregate(self, aggregation_mode: AggregationMode):
        """
        Standardizes the DataFrame according to the aggregation mode,
        modifying `self.df` in-place.

        Parameters
        ----------
        aggregation_mode : AggregationMode
            The mode by which to aggregate the grades (e.g., REMAP_7_GRADES).
        """
        if self.df.empty:
            return # No data to process

        all_df_processed = []
        for survey_id in self.surveys:
            survey_df = self.df[self.df["poll_id"] == survey_id].copy()
            if not survey_df.empty:
                all_df_processed.append(SurveyInterface(survey_df).aggregate(aggregation_mode))
        
        if all_df_processed:
            self.df = pd.concat(all_df_processed, ignore_index=True)
        else:
            # If no surveys could be processed, reset self.df to an empty DataFrame with original columns
            self.df = pd.DataFrame(columns=self.df.columns)

    @property
    def is_aggregated(self) -> bool:
        """
        Checks if the DataFrame is aggregated into a unique set of grades.
        This implies consistency in 'nombre_mentions' and grade labels across relevant rows.

        Returns
        -------
        bool
            True if the DataFrame is considered aggregated, False otherwise.
        """
        if self.df.empty or "nombre_mentions" not in self.df.columns:
            return False
            
        first_valid_index = self.df.first_valid_index()
        if first_valid_index is None:
            return False

        # Get relevant mention columns and drop NaNs to get actual grade labels for the first row
        mentions_cols = [f"mention{i}" for i in range(1, MAX_MENTIONS_IN_DATAFRAME + 1) if f"mention{i}" in self.df.columns]
        grades_first_row = self.df.loc[first_valid_index, mentions_cols].dropna().tolist()
        grades_first_row = [str(g).strip() for g in grades_first_row if str(g).strip()] # Ensure strings and remove empty

        if not grades_first_row: # If no valid grades were found in the first row
            return False

        # Compare with all unique grades found across the entire DataFrame
        all_unique_grades = self.grades

        # The DataFrame is aggregated if all unique grades match the set of grades in the first row
        # (assuming the first row is representative after aggregation)
        return all(grade in grades_first_row for grade in all_unique_grades)


    @property
    def grades(self) -> List[str]:
        """
        Collects all unique mention grades (e.g., "Très Bien", "Bien") across all surveys.

        Returns
        -------
        List[str]
            A sorted list of unique grade labels.
        """
        grades_set = set()
        mentions_cols = [f"mention{i}" for i in range(1, MAX_MENTIONS_IN_DATAFRAME + 1) if f"mention{i}" in self.df.columns]
        
        if self.df.empty or not mentions_cols:
            return []

        # Iterate over unique values in each mention column to find all grades
        for col in mentions_cols:
            # Use .dropna() to ignore NaN values, .astype(str) for consistency
            for grade_val in self.df[col].dropna().astype(str).unique():
                grade_str = grade_val.strip()
                if grade_str: # Only add non-empty strings
                    grades_set.add(grade_str)
        return sorted(list(grades_set))

    @property
    def nb_grades(self) -> int:
        """
        Get the consistent number of grades across all surveys in the DataFrame.
        This property assumes the DataFrame is aggregated or has a consistent number of mentions.

        Returns
        -------
        int
            The number of grades.

        Raises
        ------
        RuntimeError
            If the DataFrame is not aggregated or has inconsistent 'nombre_mentions' values.
        """
        if self.df.empty:
            return 0
        if not self.is_aggregated:
            raise RuntimeError(
                "The DataFrame is not aggregated. Please aggregate the DataFrame before getting the number of grades."
            )
        # After aggregation, 'nombre_mentions' should be consistent and non-null for all relevant rows
        unique_nb_mentions = self.df["nombre_mentions"].dropna().unique()
        if len(unique_nb_mentions) != 1:
            raise RuntimeError(
                "Inconsistent or multiple 'nombre_mentions' values found after aggregation. "
                "This indicates an issue with the aggregation process or data consistency."
            )
        return int(unique_nb_mentions[0])

    def _check_nb_grades(self):
        """
        Internal method to check that the 'nombre_mentions' (number of grades) is consistent
        across all individual surveys in the DataFrame. Raises a RuntimeError if inconsistency is found.
        """
        if self.df.empty:
            return # Nothing to check if the DataFrame is empty
            
        nb_grades_per_survey = []
        for s_id in self.surveys:
            df_survey = self.df[self.df["poll_id"] == s_id].copy()
            if not df_survey.empty and "nombre_mentions" in df_survey.columns:
                unique_mentions = df_survey["nombre_mentions"].dropna().unique()
                if len(unique_mentions) == 1:
                    nb_grades_per_survey.append(unique_mentions[0])
                elif len(unique_mentions) > 1:
                    raise RuntimeError(
                        f"Survey '{s_id}' has inconsistent 'nombre_mentions' values: {unique_mentions}. "
                        "All rows within a survey must have the same number of mentions."
                    )
            # Surveys with no data or missing 'nombre_mentions' are implicitly skipped from this specific check

        if not nb_grades_per_survey: # If no valid surveys were found to check for grade counts
            return
        
        if len(set(nb_grades_per_survey)) != 1:
            raise RuntimeError(
                "The number of grades ('nombre_mentions') should be the same across all surveys. "
                "Please ensure grades are aggregated or use data from polls with consistent grade structures."
            )

    @property
    def intentions(self) -> np.ndarray:
        """
        Returns a NumPy array of intention percentages for all surveys.
        Columns that are entirely NaN (e.g., if fewer mentions exist than MAX_MENTIONS_IN_DATAFRAME)
        are automatically excluded.

        Returns
        -------
        np.ndarray
            A 2D NumPy array where rows correspond to candidates/polls and columns
            correspond to intention mentions.
        """
        if self.df.empty:
            return np.array([])
            
        # Determine the number of grades to correctly select intention columns
        try:
            self._check_nb_grades() # Ensure consistency first
            num_grades = self.nb_grades # Will raise if not aggregated
        except RuntimeError:
            # Fallback for non-aggregated state: use the maximum possible mention columns
            # This might include columns that are all NaN, which will be filtered next.
            num_grades = MAX_MENTIONS_IN_DATAFRAME
        
        # Select intention columns that actually exist in the DataFrame
        intentions_cols = [
            f"intention_mention_{i}" 
            for i in range(1, num_grades + 1) 
            if f"intention_mention_{i}" in self.df.columns
        ]
        
        if not intentions_cols:
            return np.array([])

        intention_matrix = self.df[intentions_cols].to_numpy(dtype=float)
        
        # Filter out columns from the numpy array that are entirely NaN
        final_intention_matrix = intention_matrix[:, ~np.all(np.isnan(intention_matrix), axis=0)]
        
        return final_intention_matrix


    def filter(self):
        """
        Filters the DataFrame (if not already filtered) and computes rolling means
        and standard deviations for intention mentions based on 'end_date' and 'candidate'.
        Modifies `self.df` in-place by adding new rolling and standard deviation columns.
        """
        if self.df.empty:
            return # No data to process

        # Ensure grades are consistent before proceeding with calculations that rely on them
        self._check_nb_grades()
        
        # Ensure 'end_date' is datetime for resampling and sorting
        self.df["end_date"] = pd.to_datetime(self.df["end_date"], errors="coerce")
        # Sort by date and candidate for correct rolling window calculation
        self.df = self.df.sort_values(by=["end_date", "candidate"])

        # Determine the relevant intention columns (e.g., 'intention_mention_1', ...)
        # We need a robust way to get these column names.
        if self.surveys.size > 0:
            sample_survey_df = self.df[self.df["poll_id"] == self.surveys[0]].copy()
            if not sample_survey_df.empty:
                # Assuming SurveyInterface._intentions_colheaders is the correct way to get them
                intentions_col = SurveyInterface(sample_survey_df)._intentions_colheaders
            else:
                intentions_col = [] # Fallback if sample survey is empty
        else:
            intentions_col = [] # No surveys, no intention columns

        if not intentions_col:
            # Fallback if no specific intention columns could be determined (e.g., empty df, or error in SurveyInterface)
            num_grades = self.nb_grades if self.is_aggregated else MAX_MENTIONS_IN_DATAFRAME
            intentions_col = [
                f"intention_mention_{i}" 
                for i in range(1, num_grades + 1) 
                if f"intention_mention_{i}" in self.df.columns
            ]

        if not intentions_col:
            return # No intention columns found to process

        intentions_col_std = [f"{col}_std" for col in intentions_col]
        intentions_col_roll = [f"{col}_roll" for col in intentions_col]

        # Initialize new rolling and std columns to NaN if they don't exist
        for col_name in intentions_col_std + intentions_col_roll:
            if col_name not in self.df.columns:
                self.df[col_name] = np.nan

        # Process each candidate separately for rolling calculations
        for candidate_name in self.df["candidate"].unique():
            # Create a temporary DataFrame for the current candidate, ensuring it's a copy
            df_temp_candidate = self.df[self.df["candidate"] == candidate_name].copy()
            if df_temp_candidate.empty:
                continue

            # Set 'end_date' as the index for time series operations
            df_temp_candidate.set_index("end_date", inplace=True)
            df_temp_candidate = df_temp_candidate.sort_index()

            # Resample to daily frequency and calculate mean for dates with multiple entries
            # Then apply a 14-day rolling mean and standard deviation
            resampled_data = df_temp_candidate[intentions_col].resample("1d").mean()
            
            rolling_mean_df = resampled_data.rolling("14d", min_periods=1, center=True).mean()
            rolling_std_df = resampled_data.rolling("14d", min_periods=1, center=True).std()

            # Merge rolling results back into the temporary candidate DataFrame
            # Use .reindex to align the rolling results with the original dates in df_temp_candidate's index
            df_temp_candidate.loc[:, intentions_col_roll] = rolling_mean_df.reindex(df_temp_candidate.index).values
            df_temp_candidate.loc[:, intentions_col_std] = rolling_std_df.reindex(df_temp_candidate.index).values
            
            # Check sum of rolling intentions to be close to 100 (allowing for floating point errors)
            # This check can be overly strict; consider logging a warning instead of raising an error.
            if not np.all(np.isclose(df_temp_candidate[intentions_col_roll].sum(axis=1).dropna(), 100, atol=0.01)):
                # print(f"Warning: Rolling mean for candidate {candidate_name} on some dates does not sum to 100.")
                pass # Suppress strict error for production robustness

            # Update the original self.df with the calculated rolling values
            # Use .loc with multi-index selection to ensure correct assignment
            original_indices = self.df[self.df["candidate"] == candidate_name].index
            for col_roll_name, col_std_name in zip(intentions_col_roll, intentions_col_std):
                # Align values from df_temp_candidate back to the original df_temp_candidate.index based on date
                self.df.loc[original_indices, col_roll_name] = df_temp_candidate.loc[original_indices.map(df_temp_candidate.index), col_roll_name].values
                self.df.loc[original_indices, col_std_name] = df_temp_candidate.loc[original_indices.map(df_temp_candidate.index), col_std_name].values
                
    def apply_mj(
        self,
        rolling_mj: bool = False,
        official_lib: bool = False,
        reversed: bool = True,
    ):
        """
        Applies Majority Judgment rules to rank candidates for all surveys in the collection.
        This method iterates through each individual survey, applies MJ using SurveyInterface,
        and updates `self.df` with the results (rank, median grade, numerical median grade score).

        Parameters
        ----------
        rolling_mj: bool
            If True, applies rolling majority judgment (requires filtered data).
        official_lib: bool
            If True, uses the official MieuxVoter majority judgment library (if integrated).
        reversed: bool
            If True, flips the grades order for MJ calculation (e.g., higher numerical score for better grades).
            This is crucial for the 'median_grade_score' to represent satisfaction.
        """
        if self.df.empty:
            return # No data to process

        # Ensure 'median_grade_score' column exists before processing, initializing with NaN
        if "median_grade_score" not in self.df.columns:
            self.df["median_grade_score"] = np.nan

        all_df_results = []
        for survey_id in self.surveys:
            survey_df = self.df[self.df["poll_id"] == survey_id].copy()
            if not survey_df.empty:
                # SurveyInterface's apply_mj should populate 'median_grade_score' as per plan
                processed_survey_df = SurveyInterface(survey_df).apply_mj(
                    rolling_mj=rolling_mj,
                    official_lib=official_lib,
                    reversed=reversed,
                )
                all_df_results.append(processed_survey_df)
        
        if all_df_results:
            self.df = pd.concat(all_df_results, ignore_index=True)
        else:
            # If no surveys could be processed, reset self.df to an empty DataFrame with original columns
            # This ensures that even if MJ fails, the schema remains consistent for downstream operations.
            self.df = pd.DataFrame(columns=self.df.columns)


    def apply_approval(self, up_to: str):
        """
        Applies Approval Voting rules to rank candidates for all surveys in the collection.
        Modifies `self.df` in-place by adding rank columns based on approval.

        Parameters
        ----------
        up_to: str
            The grade up to which candidates are 'approved' (e.g., "Good", "Passable").
        """
        if self.df.empty:
            return # No data to process

        all_df_results = []
        for survey_id in self.surveys:
            survey_df = self.df[self.df["poll_id"] == survey_id].copy()
            if not survey_df.empty:
                processed_survey_df = SurveyInterface(survey_df).apply_approval(
                    up_to=up_to,
                    rolling_mj=False, # Explicitly pass False as apply_approval does not use rolling_mj for its internal logic
                )
                all_df_results.append(processed_survey_df)

        if all_df_results:
            self.df = pd.concat(all_df_results, ignore_index=True)
        else:
            # If no surveys could be processed, reset self.df to an empty DataFrame with original columns
            self.df = pd.DataFrame(columns=self.df.columns)

    def calculate_average_mood_time_series(self) -> pd.DataFrame:
        """
        Calculates the "average mood" of the public over time.
        The average mood is defined as the mean satisfaction score ('median_grade_score')
        across all candidates for each point in time ('end_date').
        This method first ensures Majority Judgment scores are calculated.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'end_date' and 'average_mood_score',
            sorted chronologically by 'end_date'.

        Raises
        ------
        RuntimeError
            If 'median_grade_score' is not available in the DataFrame after MJ calculation,
            indicating an issue with the underlying SurveyInterface implementation.
        """
        if self.df.empty:
            return pd.DataFrame(columns=["end_date", "average_mood_score"])

        # Ensure majority judgment scores, including median_grade_score, are calculated.
        # This calls apply_mj on all individual SurveyInterface objects and updates self.df in-place.
        # `reversed=True` is critical for 'median_grade_score' to represent satisfaction (higher=better).
        self.apply_mj(reversed=True) 

        # Critical robustness check: ensure the 'median_grade_score' column exists after apply_mj.
        # If it doesn't, it indicates a failure in the SurveyInterface.apply_mj implementation
        # or the expected column setup.
        if "median_grade_score" not in self.df.columns:
            raise RuntimeError(
                "The 'median_grade_score' column was not found after applying Majority Judgment. "
                "Ensure that SurveyInterface.apply_mj is correctly configured to generate this score "
                "when `reversed=True` is used."
            )
        
        # Ensure 'end_date' is a datetime type for reliable grouping and sorting.
        # This is already handled in __init__, but a safeguard doesn't hurt.
        if not pd.api.types.is_datetime64_any_dtype(self.df["end_date"]):
             self.df["end_date"] = pd.to_datetime(self.df["end_date"], errors="coerce")

        # Drop rows where 'end_date' or 'median_grade_score' are NaN before grouping,
        # as these cannot contribute to the average mood time series.
        df_for_mood_calculation = self.df.dropna(subset=["end_date", "median_grade_score"]).copy()

        if df_for_mood_calculation.empty:
            return pd.DataFrame(columns=["end_date", "average_mood_score"])

        # Group by 'end_date' and calculate the mean of 'median_grade_score' for each date.
        average_mood_df = (
            df_for_mood_calculation.groupby("end_date")["median_grade_score"]
            .mean()
            .reset_index(name="average_mood_score")
        )
        
        # Sort the resulting DataFrame by date for proper time series plotting
        average_mood_df = average_mood_df.sort_values(by="end_date").reset_index(drop=True)

        return average_mood_df