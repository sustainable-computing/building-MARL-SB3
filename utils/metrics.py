import pandas as pd
import pickle
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List


def calculate_spearman_correlation(ground_truth: pd.DataFrame,
                                   all_estimated_scores: Dict[str, dict],
                                   sort_ascending: Dict[str, bool] = None,
                                   zone_exclusions: List[str] = [],
                                   policy_exclusions: List[str] = []):

    assert set(all_estimated_scores.keys()) == set(sort_ascending.keys()), \
        "All methods must be given a sort order"

    all_zones = ground_truth["zone"].unique()
    all_methods = all_estimated_scores.keys()
    spearman_correlations = {method: {} for method in all_methods}

    for zone in all_zones:
        if zone in zone_exclusions:
            continue
        zone_ground_truth = ground_truth[ground_truth["zone"] == zone]
        zone_ground_truth = zone_ground_truth.sort_values(by="energy")
        estimated_scores = {method: [] for method in all_methods}
        for _, row in zone_ground_truth.iterrows():
            policy = row["policy"]
            if policy in policy_exclusions:
                continue
            for method in all_methods:
                est_score = all_estimated_scores[method][policy][zone]
                estimated_scores[method].append(est_score)
        for method in all_methods:
            if not sort_ascending[method]:
                estimated_scores[method] = estimated_scores[method][::-1]
            zone_spearmanr = spearmanr(zone_ground_truth["energy"], estimated_scores[method])
            spearman_correlations[method][zone] = zone_spearmanr

    return spearman_correlations


def convert_score_dict_to_df(score_dict: Dict[str, dict],
                             policy_exclusions: List[str] = []):
    df = {method: pd.DataFrame(columns=["policy", "zone", "score"]) for method in score_dict.keys()}
    for method in score_dict.keys():
        policy_vals = []
        zone_vals = []
        score_vals = []
        for policy in score_dict[method].keys():
            if policy in policy_exclusions:
                continue
            for zone in score_dict[method][policy].keys():
                policy_vals.append(policy)
                zone_vals.append(zone)
                score_vals.append(score_dict[method][policy][zone])
        df[method]["policy"] = policy_vals
        df[method]["zone"] = zone_vals
        df[method]["score"] = score_vals
    return df


def calculate_regret_at_k(ground_truth: pd.DataFrame,
                          all_estimated_scores: Dict[str, dict],
                          k: int,
                          sort_ascending: Dict[str, bool],
                          zone_exclusions: List[str] = [],
                          policy_exclusions: List[str] = []):
    all_estimated_scores_df = convert_score_dict_to_df(all_estimated_scores, policy_exclusions)
    all_methods = all_estimated_scores.keys()

    regret_at_k = {method: {} for method in all_methods}

    all_zones = set(ground_truth["zone"].unique()) - set(zone_exclusions)
    for method in all_estimated_scores_df.keys():
        estimated_score_zones = set(all_estimated_scores_df[method]["zone"].unique()) - set(zone_exclusions)
        assert set(estimated_score_zones) == set(all_zones), \
            f"All zones must be present in estimated scores. {estimated_score_zones} != {set(all_zones)}"
        for zone in all_zones:
            if zone in zone_exclusions:
                continue
            zone_ground_truth = ground_truth[ground_truth["zone"] == zone]
            zone_estimated_scores = \
                all_estimated_scores_df[method][all_estimated_scores_df[method]["zone"] == zone]
            regret = _regret_at_k(zone_ground_truth,
                                  zone_estimated_scores,
                                  k, sort_ascending[method])
            regret_at_k[method][zone] = regret
    return regret_at_k


def _regret_at_k(ground_truth: pd.DataFrame,
                 estimated_scores: pd.DataFrame,
                 k: int,
                 prediction_sort_ascending: bool,
                 return_top_k_policies: bool = False):
    ground_truth = ground_truth.sort_values(by="energy")
    max_delta = max(ground_truth["energy"]) - min(ground_truth["energy"])
    prediction = estimated_scores.sort_values(by=["score"], ascending=prediction_sort_ascending)

    top_k_predictions = prediction.head(k)
    top_k_policies = []
    top_k_values = []
    for _, row in top_k_predictions.iterrows():
        policy = row["policy"]
        top_k_policies.append(policy)

        policy_ground_truth_value = \
            ground_truth[ground_truth["policy"] == policy]["energy"].values[0]
        top_k_values.append(policy_ground_truth_value)

    best_ground_truth_value = min(ground_truth["energy"])
    best_predicted_value = min(top_k_values)
    regret = (best_predicted_value - best_ground_truth_value) / max_delta

    if return_top_k_policies:
        return regret, top_k_policies, top_k_policies[top_k_values.index(best_predicted_value)]
    return regret


if __name__ == "__main__":
    estimated_scores_locs = {
        "SNIP*_JAN": "../../data/policy_evaluation/ope_additional_data_collection/jan/combined/snip/policy_scores.pkl",
        "SNIP*_JUNE": "../../data/policy_evaluation/ope_additional_data_collection/june/combined/snip/policy_scores.pkl"
    }
    estimated_scores_sort_order = {
        key: False for key in estimated_scores_locs.keys()
    }

    additional_data_loc = {
        "SNIP*_JAN": "data/policy_evaluation/ope_additional_data_collection/jan/combined/snip/additional_data.pkl",
        "SNIP*_JUNE": "../../data/policy_evaluation/ope_additional_data_collection/june/combined/snip/additional_data.pkl"
    }

    one_month_eval_loc = "../../data/policy_evaluation/brute_force/reports/evaluation_report_20220820.csv"
    one_year_eval_loc = "../../data/policy_evaluation/brute_force/reports/evaluation_report_2023-02-27_12-43-04.csv"

    estimated_scores = {}
    for method in estimated_scores_locs.keys():
        with open(estimated_scores_locs[method], "rb") as f:
            estimated_scores[method] = pickle.load(f)

    one_month_df = pd.read_csv(one_month_eval_loc)
    one_year_df = pd.read_csv(one_year_eval_loc)

    one_month_df["policy"] = one_month_df["policy"].apply(lambda name: "data/policy_libraries/"+name)
    zone_exclusions = list(set(one_month_df["zone"].unique()) - set(one_year_df["zone"]))

    calculate_regret_at_k(one_month_df, estimated_scores, 1, estimated_scores_sort_order, zone_exclusions=zone_exclusions)
