import numpy as np

# from .libs.majority_judgment import majority_judgment as mj


def interface_to_official_lib(merit_profiles_dict: dict, reverse: bool):
    """source: https://github.com/MieuxVoter/majority-judgment-library-python/tree/main"""
    from majority_judgment import majority_judgment as mj
    from majority_judgment import compute_majority_values
    from majority_judgment import median_grade

    official_merit_profiles_dict = dict()
    for k, v in merit_profiles_dict.items():
        official_merit_profiles_dict[k] = [[i] * x for i, x in enumerate(v)]
        official_merit_profiles_dict[k] = [item for sublist in official_merit_profiles_dict[k] for item in sublist]

    set_num_votes = {len(votes) for votes in official_merit_profiles_dict.values()}
    if not len(set_num_votes) == 1:
        raise NotImplementedError(f"Unbalanced grades have not been implemented yet. Got {set_num_votes}")

    best_grades = {}
    for candidate, votes in merit_profiles_dict.items():
        best_grades[candidate] = median_grade(np.cumsum(votes) / np.sum(votes))

    return mj(official_merit_profiles_dict, reverse=reverse), best_grades
