"""
This module is not finished yet. But it will be very much beneficial for the whole project.
"""

from functools import cached_property
from pathlib import Path
import requests
from io import StringIO
import urllib.parse

import numpy as np
import pandas as pd

from ..misc.enums import PollingOrganizations, AggregationMode, Candidacy, UntilRound
from .survey_interface import SurveyInterface

# Get the path to the current script
CURRENT_SCRIPT_PATH = Path(__file__).parent.parent
STANDARDIZATION_CSV_PATH = CURRENT_SCRIPT_PATH / "standardisation.csv"
MAX_MENTIONS_IN_DATAFRAME = 7  # Maximum number of mentions in the dataframe, used for sanity checks


class SurveysInterface:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def load_from_url(
        cls,
        url: str = None,
        candidates: Candidacy = None,
        polling_organization: PollingOrganizations = None,
        until_round: UntilRound = None,  # note: this is not implemented yet
    ):
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        response = requests.get(url)
        response.raise_for_status()

        return cls.load(
            csv_path=StringIO(response.text),
            candidates=candidates,
            polling_organization=polling_organization,
            until_round=until_round,
        )

    @classmethod
    def load(
        cls,
        csv_path: str,
        candidates: Candidacy = None,
        polling_organization: PollingOrganizations = None,
        until_round: UntilRound = None,  # note: this is not implemented yet
    ):

        if candidates is None:
            candidates = Candidacy.ALL
        if polling_organization is None:
            polling_organization = PollingOrganizations.ALL

        df_surveys = pd.read_csv(
            csv_path,
            na_filter=False,
        )
        # df_surveys.reset_index(drop=True, inplace=True)

        # sanity conversions - force float64 to avoid dtype warnings later
        for i in range(7):
            df_surveys[f"intention_mention_{i + 1}"] = pd.to_numeric(
                df_surveys[f"intention_mention_{i + 1}"], errors="coerce"
            ).astype("float64")
        # convert mention number to integer
        df_surveys["nombre_mentions"] = pd.to_numeric(df_surveys["nombre_mentions"])

        if polling_organization != PollingOrganizations.ALL:
            df_surveys = df_surveys[df_surveys["institut"] == polling_organization.value]

        return cls(df=df_surveys)

    @cached_property
    def surveys(self):
        return self.df["poll_id"].unique()

    @cached_property
    def nb_surveys(self):
        return len(self.surveys)

    @cached_property
    def candidates(self) -> list[str]:
        """Get the list of candidates in the surveys."""
        return self.df["candidate"].unique()

    @cached_property
    def sponsors(self) -> list[str]:
        """Get the list of sponsors in the surveys."""
        return self.df["commanditaire"].unique()

    @cached_property
    def sources(self) -> list[str]:
        """Get the list of sources in the surveys."""
        return self.df["institut"].unique()

    @cached_property
    def dates(self) -> list[str]:
        """Get the list of end dates of polls."""
        dates = self.df["end_date"].unique().tolist()
        return sorted(dates)

    @cached_property
    def most_recent_date(self) -> str:
        return self.dates[-1]

    @cached_property
    def oldest_date(self) -> str:
        return self.dates[0]

    @cached_property
    def sponsors_string(self) -> str:
        """Get the string of sponsors in the surveys."""
        return ", ".join(self.sponsors)

    @cached_property
    def sources_string(self) -> str:
        """Get the string of sources in the surveys."""
        return ", ".join(self.sources)

    def select_survey(self, survey_id) -> SurveyInterface:
        return SurveyInterface(self.df[self.df["poll_id"] == survey_id].copy())

    @property
    def most_recent_survey(self) -> SurveyInterface:
        """Get the most recent survey."""
        most_recent_date = self.most_recent_date
        return SurveyInterface(self.df[self.df["end_date"] == most_recent_date].copy())

    def select_polling_organization(self, polling_organization: PollingOrganizations) -> "SurveysInterface":
        """Select surveys from a specific polling organization."""
        if polling_organization == PollingOrganizations.ALL:
            return self
        df_filtered = self.df[self.df["institut"] == polling_organization.value].copy()
        return SurveysInterface(df=df_filtered)

    def select_candidate(self, candidate: str) -> "SurveysInterface":
        """Select surveys for a specific candidate."""
        df_filtered = self.df[self.df["candidate"] == candidate].copy()
        return SurveysInterface(df=df_filtered)

    def to_no_opinion_surveys(self):
        """Convert all surveys to surveys removing the no opinion grades, and renormalizing the other grades"""
        all_df = [
            SurveyInterface(self.df[self.df["poll_id"] == survey_id].copy()).to_no_opinion_survey()
            for survey_id in self.surveys
        ]
        self.df = pd.concat([df for df in all_df], ignore_index=True)

    def aggregate(self, aggregation_mode: AggregationMode):
        """Standardize the dataframe according to the aggregation mode"""
        all_df = [
            SurveyInterface(self.df[self.df["poll_id"] == survey_id].copy()).aggregate(aggregation_mode)
            for survey_id in self.surveys
        ]
        self.df = pd.concat([df for df in all_df], ignore_index=True)

    @property
    def is_aggregated(self) -> bool:
        """Check if the dataframe is aggregated into a unique set of grades."""
        grades_first_row = self.df.loc[
            self.df.first_valid_index(), [f"mention{i}" for i in range(1, MAX_MENTIONS_IN_DATAFRAME + 1)]
        ].tolist()
        grades_first_row = [grade for grade in grades_first_row if grade not in {"nan", ""}]

        return all(grade in grades_first_row for grade in self.grades)

    @property
    def grades(self):
        grades = []
        seen = set()
        for id in self.surveys:
            for grade in self.select_survey(id).grades:
                if grade not in seen:
                    seen.add(grade)
                    grades.append(grade)
        return grades

    @property
    def nb_grades(self) -> int:
        """Get the number of grades in the dataframe."""
        if not self.is_aggregated:
            raise RuntimeError(
                "The dataframe is not aggregated. Please aggregate the dataframe before getting the number of grades."
            )
        return int(self.df["nombre_mentions"].unique()[0])

    def _check_nb_grades(self):
        """Check that the number of grades is the same for all surveys."""
        nb_grades = []
        for s in self.surveys:
            df_survey = self.df[self.df["poll_id"] == s].copy()
            nb_grades.append(df_survey["nombre_mentions"].unique()[0])
            if len(list(set(nb_grades))) != 1:
                raise RuntimeError(
                    "The number of grade should be the same for all surveys. Please aggregate grades"
                    "or use data from the same kind of polls"
                )

    @property
    def intentions(self):
        intention_matrix = self.df[[f"intention_mention_{i}" for i in range(1, self.nb_grades + 1)]].to_numpy()
        isnan_matrix = np.isnan(intention_matrix)
        final_intention_matrix = np.ndarray((len(self.df), 0))
        for intentioncol, nancol in zip(intention_matrix.T, isnan_matrix.T):
            if not np.all(nancol):
                final_intention_matrix = np.concatenate((final_intention_matrix, intentioncol[:, None]), axis=1)

        return final_intention_matrix

    def filter(self):
        """
        Filter the dataframe according to the candidates and polling organization
        """
        self._check_nb_grades()
        # new cols to store the data (rolling mean, std)
        intentions_col = SurveyInterface(self.df[self.df["poll_id"] == self.surveys[0]])._intentions_colheaders
        intentions_col_std = [f"{col}_std" for col in intentions_col]
        intentions_col_roll = [f"{col}_roll" for col in intentions_col]

        self.df[intentions_col_std] = None
        self.df[intentions_col_roll] = None
        # self.df[sans_opinion_roll] = np.nan
        self.df = self.df.sort_values(by="end_date")
        # mean by candidates
        for c in self.df["candidate"].unique():
            df_temp = self.df[self.df["candidate"] == c]
            df_temp.index = pd.to_datetime(df_temp["end_date"])
            df_temp = df_temp.sort_index()

            # Resample("1d").mean() helps to handle multiple surveys on the same dates
            df_temp[intentions_col_roll] = (
                df_temp[intentions_col].resample("1d").mean().rolling("14d", min_periods=1, center=True).mean()
            )
            df_temp[intentions_col_std] = (
                df_temp[intentions_col].resample("1d").mean().rolling("14d", min_periods=1, center=True).std()
            )

            if not df_temp[(df_temp[intentions_col_roll].sum(axis=1) - 100).round(3) != 0].empty:
                raise RuntimeError("Rolling mean conducted to less than 100 sum of intentions of vote")

            # refilling the original dataframe
            df_temp.index = self.df[self.df["candidate"] == c].index
            row_indexer = self.df[self.df["candidate"] == c].index
            for col, col_std in zip(intentions_col_roll, intentions_col_std):
                self.df.loc[row_indexer, col] = df_temp.loc[:, col]
                self.df.loc[row_indexer, col_std] = df_temp.loc[:, col]

    def apply_mj(
        self,
        rolling_mj: bool = False,
        official_lib: bool = False,
        reversed: bool = True,
    ):
        """
        Reindexing candidates in the dataFrame following majority judgment rules

        Parameters
        ----------
        rolling_mj: bool
            if we apply rolling majority judgment
        official_lib: bool
            if we use the official majority judgment lib from MieuxVoter
        reversed: bool
            if we want to flip the grades order in the data
        Returns
        -------
        Return the DataFrame df with the rank within majority judgment rules for all studies
        """
        # Compute the rank for each survey
        col_rank = "rang_glissant" if rolling_mj else "rang"
        col_median_grade = "mention_majoritaire_glissante" if rolling_mj else "mention_majoritaire"
        self.df[col_rank] = None
        self.df[col_median_grade] = None

        all_df = [
            SurveyInterface(self.df[self.df["poll_id"] == survey_id].copy()).apply_mj(
                rolling_mj=rolling_mj,
                official_lib=official_lib,
                reversed=reversed,
            )
            for survey_id in self.surveys
        ]
        self.df = pd.concat([df for df in all_df], ignore_index=True)

    def apply_approval(self, up_to: str):

        # Compute the rank for each survey
        col_rank = "rang"
        self.df[col_rank] = None

        all_df = [
            SurveyInterface(self.df[self.df["poll_id"] == survey_id].copy()).apply_approval(
                up_to=up_to,
                rolling_mj=False,
            )
            for survey_id in self.surveys
        ]
        self.df = pd.concat([df for df in all_df], ignore_index=True)
