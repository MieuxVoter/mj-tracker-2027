"""
This module is not finished yet. But it will be very much beneficial for the whole project.
"""

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..misc.enums import AggregationMode
from ..constants import NO_OPINION_GRADES
from ..utils.utils import check_sum_intentions
from ..utils.interface_mj import interface_to_official_lib
from ..libs.majority_judgment_2 import majority_judgment as mj


# Get the path to the current script
CURRENT_SCRIPT_PATH = Path(__file__).parent.parent
STANDARDIZATION_CSV_PATH = CURRENT_SCRIPT_PATH / "standardisation.csv"


class SurveyInterface:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        if self.nb_surveys != 1:
            raise ValueError(
                f"SurveyInterface only works with one survey" f"but {self.nb_surveys} surveys were provided."
            )
        self._sanity_check_on_intentions()
        self.no_opinion_grades = NO_OPINION_GRADES

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

        for no_opinion_grade in self.no_opinion_grades:
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

    def reorder_grades(self, new_order: list[str]) -> "SurveyInterface":
        """
        Reorder grades in the scale to change their position in ranking calculations.

        This is useful for repositioning grades like "sans opinion" to be treated
        as a middle grade rather than the worst grade.

        Parameters
        ----------
        new_order : list[str]
            List of grade names in the desired order (best to worst).
            Must contain all current grades exactly once.

        Returns
        -------
        SurveyInterface
            Self, for method chaining.

        Example
        -------
        # Move "sans opinion" to the middle for ELABE polls
        si.reorder_grades([
            "une image très positive",
            "une image plutôt positive",
            "sans opinion",
            "une image plutôt négative",
            "une image très négative"
        ])
        """
        current_grades = self.grades

        # Validate new_order
        if len(new_order) != len(current_grades):
            raise ValueError(
                f"new_order has {len(new_order)} grades but survey has {len(current_grades)} grades. "
                f"Current grades: {current_grades}"
            )

        if set(new_order) != set(current_grades):
            missing = set(current_grades) - set(new_order)
            extra = set(new_order) - set(current_grades)
            raise ValueError(
                f"new_order must contain exactly the same grades as the survey. " f"Missing: {missing}, Extra: {extra}"
            )

        # Build mapping: old_position -> new_position
        old_to_new = {}
        for new_pos, grade in enumerate(new_order):
            old_pos = current_grades.index(grade)
            old_to_new[old_pos] = new_pos

        # Create temporary storage for reordered data
        n_grades = self.nb_grades

        # Extract current intentions and grades
        intention_data = {}
        grade_data = {}
        for i in range(n_grades):
            old_col_intention = f"intention_mention_{i + 1}"
            old_col_grade = f"mention{i + 1}"
            intention_data[i] = self.df[old_col_intention].copy()
            grade_data[i] = self.df[old_col_grade].copy()

        # Apply new order
        for old_pos, new_pos in old_to_new.items():
            new_col_intention = f"intention_mention_{new_pos + 1}"
            new_col_grade = f"mention{new_pos + 1}"
            self.df[new_col_intention] = intention_data[old_pos]
            self.df[new_col_grade] = grade_data[old_pos]

        # Clear cached properties that depend on grades
        cached_props = [
            "grades",
            "nb_grades",
            "_intentions_colheaders",
            "_grades_colheaders",
            "_intentions_colheaders_idx",
            "_grades_colheaders_idx",
            "_no_opinion_colheader",
            "_grades_no_opinion_colheaders",
            "_grades_no_opinion_colheaders_idx",
            "_intentions_without_no_opinion_colheaders",
            "_intention_no_opinion_colheader",
            "has_no_opinion_grade",
            "is_no_opinion_last",
            "total_intentions",
            "total_intentions_no_opinion",
            "total_intentions_without_no_opinion",
            "intentions",
            "candidates",
        ]
        for prop in cached_props:
            if prop in self.__dict__:
                del self.__dict__[prop]

        return self

    def mj_data_to_dict(self):
        """Returns a sliced DataFrame containing only the columns relevant for Majority Judgment data."""
        colheaders = ["candidate"]
        colheaders.extend(self._intentions_colheaders)
        df_intentions = self.df[colheaders].copy()

        # Replace NaN with 0 for MJ calculation
        for col in self._intentions_colheaders:
            df_intentions[col] = df_intentions[col].fillna(0)

        result = {}
        for i in range(self.nb_candidates):
            candidate = df_intentions["candidate"].iloc[i]
            grades = [df_intentions.iloc[i, j + 1] for j in range(self.nb_grades)]
            result[candidate] = grades

        return result

    def to_no_opinion_survey(self, extra_grade_name: str = None) -> pd.DataFrame:
        """
        This function will return a new DataFrame with the no opinion grades removed.
        Adjusting the percentages of the other grades to make the sum of each row equal to 100%.

        Parameters
        ----------
        extra_grade_name : str
            The name of the extra grade to be considered as a "no opinion" grade.
            If None, only the standard "sans opinion" grade is considered.

        Returns
        -------
        pd.DataFrame
            A copy of the dataframe with no opinion grades removed and percentages adjusted.
        """
        # Create a local copy of no_opinion_grades to avoid mutating instance state
        local_no_opinion_grades = list(self.no_opinion_grades)
        if extra_grade_name is not None:
            local_no_opinion_grades.append(extra_grade_name)

        df_copy = self.df.copy()

        # Find all no opinion grade columns based on the local list
        grade_colheaders = self._grades_colheaders
        no_opinion_grade_names = []
        no_opinion_grade_col_indices = []

        for no_opinion_grade in local_no_opinion_grades:
            for i, grade_col in enumerate(grade_colheaders):
                grade_value = self.df[grade_col].iloc[0]
                if isinstance(grade_value, str) and no_opinion_grade in grade_value:
                    no_opinion_grade_names.append(grade_col)
                    no_opinion_grade_col_indices.append(i)

        # Remove duplicates while preserving order
        seen = set()
        unique_no_opinion_indices = []
        unique_no_opinion_names = []
        for idx, name in zip(no_opinion_grade_col_indices, no_opinion_grade_names):
            if idx not in seen:
                seen.add(idx)
                unique_no_opinion_indices.append(idx)
                unique_no_opinion_names.append(name)
        no_opinion_grade_col_indices = unique_no_opinion_indices
        no_opinion_grade_names = unique_no_opinion_names

        if not no_opinion_grade_col_indices:
            # No "no opinion" grades found, return unchanged copy
            return df_copy

        # Compute intention column headers for no opinion and valid grades
        no_opinion_intention_cols = [f"intention_mention_{i + 1}" for i in no_opinion_grade_col_indices]
        valid_intention_cols = [
            f"intention_mention_{i + 1}" for i in range(self.nb_grades) if i not in no_opinion_grade_col_indices
        ]

        # Calculate totals for normalization
        tot_undecided = df_copy[no_opinion_intention_cols].sum(axis=1)
        tot_decided = df_copy[valid_intention_cols].sum(axis=1)

        # Normalize the valid intention columns to sum to 100%
        # Handle division by zero: if tot_decided is 0, normalizing_coef would be inf
        normalizing_coef = (tot_decided + tot_undecided) / tot_decided.replace(0, np.nan)
        normalizing_coef = normalizing_coef.fillna(1.0)

        for col in valid_intention_cols:
            df_copy[col] = (df_copy[col] * normalizing_coef).astype(float)

        # Verify the sum is now 100%
        actual_sum = df_copy[valid_intention_cols].sum(axis=1).round(5)
        if not (actual_sum == 100).all():
            bad_rows = df_copy[actual_sum != 100]
            raise ValueError(
                f"Sum of intention mentions is not 100 after normalization. "
                f"Got values: {actual_sum[actual_sum != 100].tolist()} "
                f"for candidates: {bad_rows['candidate'].tolist() if 'candidate' in bad_rows.columns else 'unknown'}"
            )

        # Store the no opinion values in a dedicated column
        if "sans_opinion" not in df_copy.columns:
            df_copy["sans_opinion"] = 0.0
        df_copy["sans_opinion"] = df_copy["sans_opinion"].astype(float)
        df_copy["sans_opinion"] += tot_undecided

        # Zero out the no opinion intention columns
        for col in no_opinion_intention_cols:
            df_copy[col] = 0.0

        # Mark no opinion grade names as "nan"
        for grade_col in no_opinion_grade_names:
            df_copy[grade_col] = "nan"

        # Update the number of declared grades
        new_nb_grades = self.nb_grades - len(no_opinion_grade_col_indices)
        df_copy["nombre_mentions"] = new_nb_grades

        # Compact the columns: shift valid grades to fill gaps left by no opinion grades
        # This is necessary when no opinion grades are scattered (non-consecutive)
        is_no_opinion_last = max(no_opinion_grade_col_indices) == self.nb_grades - 1 and len(
            no_opinion_grade_col_indices
        ) == len([i for i in no_opinion_grade_col_indices if i >= self.nb_grades - len(no_opinion_grade_col_indices)])

        if not is_no_opinion_last:
            # Get valid column indices (those NOT in no_opinion)
            valid_grade_indices = [i for i in range(self.nb_grades) if i not in no_opinion_grade_col_indices]

            # Compact intention columns
            intention_values = [df_copy[f"intention_mention_{i + 1}"].copy() for i in valid_grade_indices]
            for new_pos, values in enumerate(intention_values):
                df_copy[f"intention_mention_{new_pos + 1}"] = values
            # Zero out trailing intention columns
            for i in range(len(valid_grade_indices), self.nb_grades):
                df_copy[f"intention_mention_{i + 1}"] = 0.0

            # Compact grade name columns
            grade_values = [df_copy[f"mention{i + 1}"].copy() for i in valid_grade_indices]
            for new_pos, values in enumerate(grade_values):
                df_copy[f"mention{new_pos + 1}"] = values
            # Set "nan" for trailing grade columns
            for i in range(len(valid_grade_indices), self.nb_grades):
                df_copy[f"mention{i + 1}"] = "nan"

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
        if "avant_mention_majortiaire" not in self.df.columns:
            self.df["avant_mention_majortiaire"] = None
        if "apres_mention_majortiaire" not in self.df.columns:
            self.df["apres_mention_majortiaire"] = None
        if "tri_majoritaire" not in self.df.columns:
            self.df["tri_majoritaire"] = None

        col_rank = self.df.columns.get_loc(col_rank)
        col_best_grade = self.df.columns.get_loc(col_median_grade)
        for c in ranking:
            idx = np.where(self.df["candidate"] == c)[0][0]
            self.df.iat[idx, col_rank] = ranking[c]

        grade_list = self.grades
        if reversed:
            grade_list.reverse()

        for candidate, best_grade in best_grades.items():
            idx = np.where(self.df["candidate"] == candidate)[0][0]
            self.df.iat[idx, col_best_grade] = grade_list[best_grade]

            if reversed:
                best_grade = self.nb_grades - 1 - best_grade

            before_best_grade = self.sum_intentions(up_to=best_grade - 1).iloc[idx]
            after_best_grade = 100 - self.sum_intentions(up_to=best_grade).iloc[idx]

            self.df.iat[idx, self.df.columns.get_loc("avant_mention_majortiaire")] = before_best_grade
            self.df.iat[idx, self.df.columns.get_loc("apres_mention_majortiaire")] = after_best_grade
            if before_best_grade > after_best_grade:
                self.df.iat[idx, self.df.columns.get_loc("tri_majoritaire")] = "approbation"
            elif before_best_grade < after_best_grade:
                self.df.iat[idx, self.df.columns.get_loc("tri_majoritaire")] = "rejet"

        return self.df

    def apply_approval(self, up_to: str, rolling_mj: bool = False):
        """
        Apply the Approval rule to the survey data.
        This method calculates the approval percentage for each candidate based on their intentions.
        """
        id_mention = self.grades.index(up_to) if up_to != "last" else self.nb_grades

        df_with_rank = self._sort_candidates_approval(
            up_to=id_mention,
            col_rank="rang_glissant" if rolling_mj else "rang",
        )
        return df_with_rank

    def _sort_candidates_approval(
        self,
        up_to: int,
        col_rank: str = None,
    ):
        """
        Reindexing candidates in the dataFrame following Approval rules

        Parameters
        ----------
        up_to : int
            The mention to sum up to. Can be "last" for all mentions or a specific mention number.
            from 0 to nb_grades-1.
        col_rank: str
            rank col to considered (ex: rang or rang_glissant)
        official_lib: bool
            if we use the official majority judgment lib from MieuxVoter
        Returns
        -------
        Return the DataFrame df sorted with the rank within Approval rules.
        """
        col_rank = "rang" if col_rank is None else col_rank
        col_approval = "approbation"

        # Calculer l'approbation pour chaque candidat
        approval = self.sum_intentions(up_to=up_to)

        # Vérifier si la colonne d'approbation existe, sinon la créer
        if col_approval not in self.df.columns:
            self.df[col_approval] = None

        # Créer un DataFrame temporaire avec les scores d'approbation et les candidats
        temp_df = pd.DataFrame({"candidate": self.df["candidate"], "approval_score": approval})

        # Trier par score d'approbation décroissant
        sorted_temp_df = temp_df.sort_values("approval_score", ascending=False)

        # Attribuer les rangs et scores d'approbation
        for rank, (_, row) in enumerate(sorted_temp_df.iterrows(), 1):
            candidate = row["candidate"]
            score = row["approval_score"]
            idx = np.where(self.df["candidate"] == candidate)[0][0]
            self.df.iat[idx, self.df.columns.get_loc(col_rank)] = rank
            self.df.iat[idx, self.df.columns.get_loc(col_approval)] = score

        self.df = self.df.sort_values(by=col_rank, na_position="last")  # Handle NAs in rank

        return self.df

    @cached_property
    def intentions(self):
        colheaders = ["candidate"] + self._intentions_colheaders
        return self.df[colheaders]

    def sum_intentions(self, up_to: str | int = "last"):
        """
        Returns the sum of intentions for each candidate up to specified mention from 1 to nb_grades.
        Args:
            up_to (str or int): The mention to sum up to. Can be "last" for all mentions or a specific mention number.
                from 0 to nb_grades-1.
        """

        if up_to == "last":
            return self.df[self._intentions_colheaders].sum(axis=1)
        elif isinstance(up_to, int):
            up_to += 1  # Convert to 1-based index for user-friendliness
            if 1 <= up_to <= self.nb_grades:
                cols_to_sum = self._intentions_colheaders[:up_to]
                return self.df[cols_to_sum].sum(axis=1)
            elif up_to == 0:
                return pd.Series(0, index=self.df.index)  # Return zero if up_to is 0
        elif isinstance(up_to, str):
            is_up_to = up_to in self._grades_colheaders
            if not is_up_to:
                raise ValueError(
                    f"Invalid value for 'up_to': {up_to}. Must be 'last' or an integer between 1 and {self.nb_grades}."
                )
            up_to_idx = self._grades_colheaders.index(up_to)
            cols_to_sum = self._intentions_colheaders[: up_to_idx + 1]
            return self.df[cols_to_sum].sum(axis=1)
        else:
            raise ValueError(
                f"Invalid value for 'up_to': {up_to}. Must be 'last' or an integer between 1 and {self.nb_grades}."
            )

    def cumulative_intentions(self):
        """
        Returns a DataFrame with cumulative intentions for each candidate.
        Each column represents the cumulative sum of intentions up to that mention.
        """
        cumulative_df = pd.DataFrame()
        cumulative_df["candidate"] = self.df["candidate"]

        cumulative_sum = np.zeros(len(self.df))
        for i, col in enumerate(self._intentions_colheaders):
            cumulative_sum += self.df[col]
            cumulative_df[f"cumulative_intention_mention_{i + 1}"] = cumulative_sum

        return cumulative_df

    @cached_property
    def candidates(self):
        return self.df["candidate"].to_list()

    @cached_property
    def ranked_candidates(self) -> list[str]:
        """Returns a list of candidates sorted by their rank."""
        if "rang" not in self.df.columns:
            raise ValueError(
                "The DataFrame does not contain a 'rang' column. "
                "Please apply majority judgment first by calling method apply_mj()."
            )
        sorted_df = self.df.sort_values(by="rang", na_position="last")  # Handle NAs in rank
        return list(sorted_df["candidate"])

    def formated_ranked_candidates(self, show_no_opinion: bool = False) -> list[str]:
        """Returns a list of candidates sorted by their rank formatted for plots"""
        has_no_opinion_col = "sans_opinion" in self.df.columns
        if has_no_opinion_col and show_no_opinion and not np.isnan(self.df["sans_opinion"].unique()[0]):
            sorted_df = self.df.sort_values(by="rang", na_position="last")  # Handle NAs in rank
            return [
                "<b>"
                + s
                + "</b>"
                + "     <br><i>(sans opinion "
                + str(sorted_df[sorted_df["candidate"] == s]["sans_opinion"].iloc[0])
                + "%)</i>     "
                for s in self.ranked_candidates
            ]
        else:
            return ["<b>" + s + "</b>" + "     " for s in self.ranked_candidates]

    def select_candidate(self, candidate: str) -> pd.DataFrame:
        """Returns a SurveyInterface for a specific candidate."""
        if candidate not in self.candidates:
            raise ValueError(f"Candidate '{candidate}' not found in the survey.")
        return self.df[self.df["candidate"] == candidate].copy
