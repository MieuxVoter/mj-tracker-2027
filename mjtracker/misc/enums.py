from enum import Enum, IntEnum
from functools import cached_property
from pathlib import Path
import pandas as pd

# Get the path to the current script
CURRENT_SCRIPT_PATH = Path(__file__).parent
STANDARDIZATION_CSV_PATH = CURRENT_SCRIPT_PATH / "../standardisation.csv"


class Candidacy(Enum):
    """
    Select candidates
    """

    ALL_CURRENT_CANDIDATES_WITH_ENOUGH_DATA = "all_current_candidates_with_enough_data"
    ALL_CURRENT_CANDIDATES = "all_current_candidates"
    ALL_CANDIDATES_FROM_BEGINNING = "all_candidates"
    SECOND_ROUND = "second_round"
    ALL = "all"


class AggregationMode(Enum):
    """
    Select how Aggregation are managed
    """

    NO_AGGREGATION = "None"
    FOUR_MENTIONS = "to_4_mentions"

    @cached_property
    def map(self):
        """
        Map the aggregation mode to the corresponding column name in the DataFrame.
        """
        df_standardisation = pd.read_csv(STANDARDIZATION_CSV_PATH, na_filter=False, index_col=False)
        # make a dict out of the csv file
        if self == self.NO_AGGREGATION:
            raise ValueError("Aggregation mode has no mapping")
        elif self == self.FOUR_MENTIONS:
            inputs = df_standardisation["mention"].tolist()
            outputs = df_standardisation["to_4_mentions"].tolist()
            return dict(zip(inputs, outputs))

    def potential_grades(self, grade: str):
        """Give a list of all the potential grades to be converted for a one of the new grades"""
        if self == self.NO_AGGREGATION:
            raise ValueError("Aggregation mode has no mapping")
        elif self == self.FOUR_MENTIONS:
            df_standardisation = pd.read_csv(STANDARDIZATION_CSV_PATH, na_filter=False, index_col=False)
            # get the values of the dict
            values = df_standardisation[df_standardisation["to_4_mentions"] == grade]["mention"].tolist()
            # unique only
            return list(dict.fromkeys(values))

    @cached_property
    def grades(self):
        """
        Get the grade names of the aggregation mode.
        """
        if self == self.NO_AGGREGATION:
            raise ValueError("Aggregation mode has no mapping")
        elif self == self.FOUR_MENTIONS:
            map = self.map
            # get the values of the dict
            values = list(map.values())
            # unique only
            return list(dict.fromkeys(values))

    @property
    def nb(self):
        """
        Get the number of mentions for the aggregation mode.
        """
        if self == self.NO_AGGREGATION:
            raise ValueError("Aggregation mode has no mapping")
        elif self == self.FOUR_MENTIONS:
            return 4

    @property
    def string_label(self):
        """
        Get the string label of the aggregation mode.
        """
        if self == self.NO_AGGREGATION:
            return ""
        else:
            return f"_{self.name.lower()}"


class PollingOrganizations(Enum):
    """
    Select how Institutes
    """

    ALL = "Opinion Way, ELABE, IFOP, IPSOS"
    # MIEUX_VOTER = "Opinion Way"
    ELABE = "ELABE"
    # IFOP = "IFOP"
    IPSOS = "IPSOS"


class UntilRound(Enum):
    """
    Select which round
    """

    FIRST = "2022-04-10"
    SECOND = "2022-04-24"
