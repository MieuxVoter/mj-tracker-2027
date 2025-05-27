"""
This module is not finished yet. But it will be very much beneficial for the whole project.
"""

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .misc.enums import AggregationMode
from .constants import NO_OPINION_GRADES
from .utils import check_sum_intentions
from .interface_mj import interface_to_official_lib
from .libs.majority_judgment_2 import majority_judgment as mj


# Get the path to the current script
CURRENT_SCRIPT_PATH = Path(__file__).parent
STANDARDIZATION_CSV_PATH = CURRENT_SCRIPT_PATH / "standardisation.csv"


class SurveyInterface:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        if self.nb_surveys != 1:
            raise ValueError(
                f"SurveyInterface only works with one survey" f"but {self.nb_surveys} surveys were provided."
            )
        self._sanity_check_on_intentions()

    def _sanity_check_on_intentions(self):
        cols = self._intentions_colheaders
        total_percentages_among_all_candidates = self.df[cols].sum(axis=1).round(5).unique()
        if len(total_percentages_among_all_candidates) != 1:
            id_survey = self.df["poll_id"][self.df.first_valid_index()]

            for i in range(len(total_percentages_among_all_candidates)):
                if total_percentages_among_all_candidates[i] != 100:
                    select = self.df[cols].sum(axis=1).round(5) == total_percentages_among_all_candidates[i]
                    print(
                        f"candidate {self.df['candidate'][select]} has {total_percentages_among_all_candidates[i]}% "
                        f"of intentions in survey {id_survey}"
                    )

            raise ValueError(f"the number of grades is not equal for each candidate in {id_survey}")

    @cached_property
    def surveys(self):
        return list(self.df["poll_id"].unique())

    @cached_property
    def nb_surveys(self):
        return len(self.surveys)

    @cached_property
    def nb_grades(self) -> int:
        return int(self.df["nombre_mentions"].unique()[0])

    @cached_property
    def nb_candidates(self) -> int:
        return len(self.df["candidate"].unique())

    @cached_property
    def source(self) -> str:
        return self.df["institut"].loc[self.df.first_valid_index()]

    @cached_property
    def sponsor(self) -> str:
        return self.df["commanditaire"].loc[self.df.first_valid_index()]

    @cached_property
    def end_date(self) -> str:
        return self.df["end_date"].loc[self.df.first_valid_index()]

    @cached_property
    def grades(self) -> list[str]:
        """Returns the list of grades used in the survey. taken from the first row of the dataframe."""
        return self.df.iloc[0][self._grades_colheaders].tolist()

    @cached_property
    def _intentions_colheaders(self):
        return [f"intention_mention_{i}" for i in range(1, self.nb_grades + 1)]

    @cached_property
    def _intentions_colheaders_idx(self):
        return [self.df.columns.get_loc(f"intention_mention_{i}") for i in range(1, self.nb_grades + 1)]

    @cached_property
    def _grades_colheaders(self):
        return [f"mention{i}" for i in range(1, self.nb_grades + 1)]

    @cached_property
    def _grades_colheaders_idx(self):
        return [self.df.columns.get_loc(f"mention{i}") for i in range(1, self.nb_grades + 1)]

    @cached_property
    def _grades_no_opinion_colheaders(self):
        col_idx = np.array(self._no_opinion_colheader[1]) + 1
        return [f"mention{i}" for i in col_idx]

    @cached_property
    def _grades_no_opinion_colheaders_idx(self):
        no_opinion_grades = self._grades_no_opinion_colheaders
        return [self.df.columns.get_loc(g) for g in no_opinion_grades]

    @cached_property
    def _intentions_without_no_opinion_colheaders(self):
        no_opinion_col_idx = np.array(self._no_opinion_colheader[1]) + 1
        return [f"intention_mention_{i}" for i in range(1, self.nb_grades + 1) if i not in no_opinion_col_idx]

    @cached_property
    def _intention_no_opinion_colheader(self):
        col_idx = np.array(self._no_opinion_colheader[1]) + 1
        return [f"intention_mention_{i}" for i in col_idx]

    @cached_property
    def _no_opinion_colheader(self):
        """
        Returns the column headers of the grades that are considered as no opinion grades.

        Returns
        -------
        the_undecided_grades: list
            list of column headers that are considered as no opinion grades.
        the_undecided_grade_col_idx: list
            list of the indexes of the grades that are considered as no opinion grades.
            if index 4, means intention_mention_5 and mention5

        """
        grade_colheaders = self._grades_colheaders

        the_undecided_grades = []
        the_undecided_grade_col_idx = []

        for no_opinion_grade in NO_OPINION_GRADES:
            bool_with_undecided = self.df[grade_colheaders].iloc[0, :].str.contains(no_opinion_grade)
            if bool_with_undecided.any():
                the_undecided_grades.append(bool_with_undecided[bool_with_undecided].index)
                the_undecided_grade_col_idx.append(np.where(bool_with_undecided)[0][0])

        return the_undecided_grades, the_undecided_grade_col_idx

    @cached_property
    def has_no_opinion_grade(self) -> bool:
        """Check if the no opinion grades are present in the dataframe"""
        # check if the no opinion grades are present in the dataframe
        no_opinion_colheaders, _ = self._no_opinion_colheader
        return len(no_opinion_colheaders) > 0

    @cached_property
    def is_no_opinion_last(self) -> bool:
        """Check if the no opinion grades are the last ones in the dataframe"""
        # check if the no opinion grades are the last ones in the dataframe
        if not self.has_no_opinion_grade:
            raise ValueError("No no opinion grades found in the dataframe")

        no_opinion_colheaders, _ = self._no_opinion_colheader
        grade_colheaders = self._grades_colheaders
        return no_opinion_colheaders[-1][0] == grade_colheaders[-1]

    @cached_property
    def total_intentions(self):
        return self.df[self._intentions_colheaders].sum(axis=1)

    @cached_property
    def total_intentions_no_opinion(self):
        return self.df[self._intention_no_opinion_colheader].sum(axis=1)

    @cached_property
    def total_intentions_without_no_opinion(self):
        return self.df[self._intentions_without_no_opinion_colheaders].sum(axis=1)

    def mj_data_to_dict(self):
        """Returns a sliced DataFrame containing only the columns relevant for Majority Judgment data."""
        colheaders = ["candidate"]
        colheaders.extend(self._intentions_colheaders)
        df_intentions = self.df[colheaders].copy()
        return {
            df_intentions["candidate"].iloc[i]: [df_intentions.iloc[i, j + 1] for j in range(self.nb_grades)]
            for i in range(self.nb_candidates)
        }

    def to_no_opinion_survey(self):
        """
        This function will return a new SurveyInterface object with the no opinion grades removed.
        Adjusting the percentages of the other grades to make the sum of each row equal to 100%.
        """
        df_copy = self.df.copy()
        the_undecided_grades, the_undecided_grade_col_idx = self._no_opinion_colheader

        for grade in the_undecided_grades:
            df_copy.loc[df_copy.index, grade] = "nan"

        # reduced the number of declared grades
        index_no = df_copy.columns.get_loc("nombre_mentions")
        df_copy.loc[df_copy.index, "nombre_mentions"] = int(self.df.iloc[0, index_no] - len(the_undecided_grades))

        tot_undecided = self.total_intentions_no_opinion
        tot_decided = self.total_intentions_without_no_opinion

        for decided_col in self._intentions_without_no_opinion_colheaders:
            col_idx = df_copy.columns.get_loc(decided_col)
            normalizing_coef = 1 + tot_undecided / tot_decided
            df_copy.iloc[:, col_idx] = df_copy.iloc[:, col_idx] * normalizing_coef

        if (df_copy[self._intentions_without_no_opinion_colheaders].sum(axis=1).round(5) != 100).any():
            raise ValueError(
                "Sum of intention mentions is not 100, but {df_copy[['intention_mention_1', 'intention_mention_2', 'intention_mention_4', 'intention_mention_5']].sum(axis=1).round(5)}"
            )

        # the no opinion col to zero and store it somwehere else
        if "sans_opinion" not in df_copy.columns:
            df_copy["sans_opinion"] = 0
        df_copy["sans_opinion"] += df_copy[self._intention_no_opinion_colheader].sum(
            axis=1
        )  # adds up with the previous values
        df_copy[self._intention_no_opinion_colheader] = 0

        # remove the no opinion grades from the dataframe
        if self.has_no_opinion_grade and not self.is_no_opinion_last:
            offset = len(self._intention_no_opinion_colheader)

            intention_colheaders_idx = self._intentions_colheaders_idx
            last_no_opinion_idx = list(self.df.columns).index(self._intention_no_opinion_colheader[-1])
            idx_after_last_no_opinion = [i for i in intention_colheaders_idx if i > last_no_opinion_idx]

            for i in idx_after_last_no_opinion:
                df_copy.iloc[:, i - offset] = df_copy.iloc[:, i]
            df_copy.iloc[:, idx_after_last_no_opinion[-1]] = 0

            grade_colheaders_idx = self._grades_colheaders_idx
            last_no_opinion_idx = self._grades_no_opinion_colheaders_idx[-1]
            idx_after_last_no_opinion = [i for i in grade_colheaders_idx if i > last_no_opinion_idx]

            for i in idx_after_last_no_opinion:
                df_copy.iloc[:, i - offset] = df_copy.iloc[:, i]
            df_copy.iloc[:, idx_after_last_no_opinion[-1]] = 0

        check_sum_intentions(df_copy)
        return df_copy

    def aggregate(self, aggregation_mode: AggregationMode):
        map_grades = aggregation_mode.map

        # grades of the current survey
        cols_grades = self._grades_colheaders
        cols_grades_idx = [self.df.columns.get_loc(c) for c in cols_grades]

        cols_intentions = self._intentions_colheaders
        cols_intentions_idx = [self.df.columns.get_loc(c) for c in cols_intentions]
        df_grades_only = self.df[cols_grades].loc[self.df.first_valid_index()]

        # Refill the new_survey dataframe
        new_df_survey = self.df.copy()
        new_df_survey.iloc[:, cols_intentions_idx] = 0
        new_df_survey.iloc[:, cols_grades_idx] = "nan"

        # Add the numbers together
        new_grades = []
        old_grades = self.df.iloc[0, cols_grades_idx].to_list()

        for old_grade in old_grades:
            new_grades += [map_grades.get(old_grade, None)]

        for i, ng in enumerate(new_grades):
            # todo: fix the fact that elabe has four but there might be a conversion to sans opinion
            new_df_survey.iloc[:, cols_grades_idx[i]] = ng
            potential_grades = aggregation_mode.potential_grades(ng)
            # find if all the potential grades are in this survey
            for pg in potential_grades:
                pg_in_grades_survey = pg == df_grades_only
                if pg_in_grades_survey.any():
                    idx = np.where(pg_in_grades_survey)[0][0]
                    new_df_survey.iloc[:, cols_intentions_idx[i]] += self.df.iloc[:, cols_intentions_idx[idx]]

        check_sum_intentions(new_df_survey)

        if "sans opinion" in new_grades:
            assert len(new_grades) == aggregation_mode.nb + 1
            new_df_survey.loc[new_df_survey.index, "nombre_mentions"] = len(new_grades)
            return SurveyInterface(new_df_survey).to_no_opinion_survey()
        else:
            new_df_survey.loc[new_df_survey.index, "nombre_mentions"] = aggregation_mode.nb
            return new_df_survey

    def apply_mj(
        self,
        rolling_mj: bool = False,
        official_lib: bool = False,
        reversed: bool = True,
    ):
        df_with_rank = self._sort_candidates_mj(
            col_rank="rang_glissant" if rolling_mj else "rang",
            col_median_grade="mention_majoritaire_glissante" if rolling_mj else "mention_majoritaire",
            official_lib=official_lib,
            reversed=reversed,
        )
        return df_with_rank

    def _sort_candidates_mj(
        self,
        col_rank: str = None,
        col_median_grade: str = None,
        official_lib: bool = False,
        reversed: bool = True,
    ):
        """
        Reindexing candidates in the dataFrame following majority judgment rules

        Parameters
        ----------
        df: DataFrame
            contains all the data of vote / survey
        col_rank: str
            rank col to considered (ex: rang or rang_glissant)
        col_median_grade: str
            rank col to considered (ex: mention_majoritaire or mention_majoritaire_glissante)
        official_lib: bool
            if we use the official majority judgment lib from MieuxVoter
        Returns
        -------
        Return the DataFrame df sorted with the rank within majority judgment rules.
        """
        col_rank = "rang" if col_rank is None else col_rank
        col_median_grade = "mention_majoritaire" if col_median_grade is None else col_median_grade

        merit_profiles_dict = self.mj_data_to_dict()

        if official_lib:
            ranking, best_grades = interface_to_official_lib(merit_profiles_dict, reverse=reversed)
        else:
            # for majority-judgment-tracker has I kept percentages instead of votes, this method is preferred
            ranking, best_grades = mj(merit_profiles_dict, reverse=reversed)

        if col_rank not in self.df.columns:
            self.df[col_rank] = None
        if col_median_grade not in self.df.columns:
            self.df[col_median_grade] = None

        col_rank = self.df.columns.get_loc(col_rank)
        col_best_grade = self.df.columns.get_loc(col_median_grade)
        for c in ranking:
            idx = np.where(self.df["candidate"] == c)[0][0]
            self.df.iat[idx, col_rank] = ranking[c]

        grade_list = self._grades_colheaders
        if not reversed:
            grade_list.reverse()

        for c, val in best_grades.items():
            idx = np.where(self.df["candidate"] == c)[0][0]
            self.df.iat[idx, col_best_grade] = grade_list[val]

        return self.df

    @cached_property
    def intentions(self):
        colheaders = ["candidate"] + self._intentions_colheaders
        return self.df[colheaders]

    @cached_property
    def candidates(self):
        return self.df["candidate"].unique()

    def select_candidate(self, candidate: str) -> pd.DataFrame:
        """Returns a SurveyInterface for a specific candidate."""
        if candidate not in self.candidates:
            raise ValueError(f"Candidate '{candidate}' not found in the survey.")
        return self.df[self.df["candidate"] == candidate].copy

