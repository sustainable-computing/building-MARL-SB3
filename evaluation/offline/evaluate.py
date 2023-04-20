from evaluation.offline import OPEMethodStrings
from evaluation.offline.iw import InverseProbabilityWeighting
from evaluation.offline.iw import SelfNormalizedInverseProbabilityWeighting
from evaluation.offline.zcp import SNIP
from evaluation.offline.zcp import GaussianKernel
from policies.utils import load_policy_library

from datetime import datetime
from typing import List
import numpy as np
import os
import pandas as pd
import pickle
import tqdm
import yaml


def evaluate(methods: List[str] = ["ipw"],
             log_data_path: str = "data/rule_based_log_data/denver/0_cleaned_log.csv",
             policy_library_path: str = "data/policy_libraries/policy_library_20220820",
             policy_type: str = "single_agent_ac",
             init_policy_log_std: float = np.log(0.1),
             init_policy_log_std_path: str = "",
             zone: str = None,
             num_days: int = 15,
             start_day: int = 0,
             start_month: int = 1,
             start_year: int = 2019,
             use_behavior_p_score: bool = True,
             behavior_policy_path: str = "data/rule_based_log_data/denver/action_probs_all_data.pkl",
             save_path: str = "data/policy_evaluation/ope/",
             parallelize: bool = False,
             max_cpu_cores: int = 1,
             gaussian_kernel_bandwidth: float = 0.3):

    kwargs = locals()
    for method in methods:
        assert method in OPEMethodStrings._value2member_map_, \
            f"Invalid method. Please choose from {OPEMethodStrings._value2member_map_.keys()}"
    if use_behavior_p_score:
        assert os.path.exists(behavior_policy_path), \
            f"Behavior policy path {behavior_policy_path} does not exist."
        with open(behavior_policy_path, "rb") as f:
            behavior_policy = pickle.load(f)

    save_path = _create_save_path(save_path, methods)
    _save_run_config(save_path, kwargs)

    log_data_df = pd.read_csv(log_data_path)
    log_data_df["time"] = pd.to_datetime(log_data_df["time"])
    # start_date = log_data_df["time"].values[0]
    start_date = datetime(year=start_year, month=start_month, day=start_day)
    end_date = start_date + pd.Timedelta(days=num_days)
    log_data_df = log_data_df[(log_data_df["time"] >= start_date) &
                              (log_data_df["time"] < end_date)]

    if zone is None:
        zones = log_data_df["zone"].unique()
    else:
        zones = [zone]

    policies, policy_paths = load_policy_library(policy_library_path, policy_type,
                                                 init_policy_log_std, init_policy_log_std_path,
                                                 eval_mode=True)
    policy_scores = {}
    if parallelize:
        # TODO: Implement parallelization
        raise NotImplementedError("Parallelization not implemented yet.")

    for method in methods:
        for zone in zones:
            zone_log_data_df = log_data_df[log_data_df["zone"] == zone]
            method_obj = _get_method(method, zone_log_data_df, kwargs)
            for policy, policy_path in tqdm.tqdm(zip(policies, policy_paths), total=len(policies)):
                score = method_obj.evaluate_policy(evaluation_policy=policy,
                                                   evaluation_policy_distribution_fuc=policy.get_distribution,
                                                   behavior_policy=behavior_policy,
                                                   score="mean")
                if method not in policy_scores:
                    policy_scores[method] = {}
                if policy_path not in policy_scores[method]:
                    policy_scores[method][policy_path] = {}
                policy_scores[method][policy_path][zone] = score

    for method in methods:
        with open(os.path.join(save_path, method, "policy_scores.pkl"), "wb") as f:
            pickle.dump(policy_scores[method], f)


def _get_method(method, log_data, kwargs):
    if method == OPEMethodStrings.ipw.value:
        method_obj = InverseProbabilityWeighting(log_data)
    elif method == OPEMethodStrings.snip.value:
        method_obj = SNIP(log_data)
    elif method == OPEMethodStrings.snipw.value:
        method_obj = SelfNormalizedInverseProbabilityWeighting(log_data)
    elif method == OPEMethodStrings.gk.value:
        method_obj = GaussianKernel(log_data=log_data,
                                    bandwidth=kwargs["gaussian_kernel_bandwidth"])

    return method_obj


def _create_save_path(save_dir, methods):
    save_path = os.path.join(save_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    for method in methods:
        method_path = os.path.join(save_path, method)
        os.makedirs(method_path)
    return save_path


def _save_run_config(save_path, args):
    with open(os.path.join(save_path, "run_config.yaml"), "w") as f:
        yaml.dump(args, f)


def get_policy_score(method: str,
                     zone_log_data_df: pd.DataFrame,
                     policy: object,
                     behavior_policy: object) -> float:

    if method == "ipw":
        ope_method = InverseProbabilityWeighting(zone_log_data_df)
        score, _, _, _, _ = ope_method.evaluate_policy(policy.get_distribution, behavior_policy, score="mean")
    elif method == "snipw":
        ope_method = SelfNormalizedInverseProbabilityWeighting(zone_log_data_df)
        score, _, _, _, _ = ope_method.evaluate_policy(policy.get_distribution, behavior_policy, score="mean")
    elif method == "snip":
        ope_method = SNIP(zone_log_data_df)
        score = ope_method.evaluate_policy(policy, behavior_policy)
    elif method == "gk":
        ope_method = GaussianKernel(zone_log_data_df)
        score = ope_method.evaluate_policy(policy)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    return score


if __name__ == "__main__":
    # Run tests if necessary
    evaluate()
