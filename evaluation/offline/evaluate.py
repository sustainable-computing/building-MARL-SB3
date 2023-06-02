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
             zones: List[str] = [],
             use_full_log_data: bool = False,
             num_days: List[int] = [15],
             start_day: List[int] = [1],
             start_month: List[int] = [1],
             start_year: List[int] = [2019],
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

    if not use_full_log_data:
        temp_df = pd.DataFrame(columns=log_data_df.columns)
        for year, month, day, num_day in zip(start_year, start_month, start_day, num_days):
            start_row = log_data_df[(log_data_df["time"].dt.year == year) &
                                    (log_data_df["time"].dt.month == month) &
                                    (log_data_df["time"].dt.day == day)].index[0]
            start_date = datetime(year, month, day)
            end_date = start_date + pd.Timedelta(days=num_day)
            subset_data = log_data_df[(log_data_df["time"] >= start_date) &
                                      (log_data_df["time"] < end_date)]
            temp_df = temp_df.append(subset_data)
        log_data_df = temp_df

    if zones == []:
        zones = log_data_df["zone"].unique()
    else:
        zones = zones
        for zone in zones:
            assert zone in log_data_df["zone"].unique(), f"Zone {zone} not in log data."

    policies, policy_paths = load_policy_library(policy_library_path, policy_type,
                                                 init_policy_log_std, init_policy_log_std_path,
                                                 eval_mode=True)
    policy_scores = {}
    all_additional_data = {}
    if parallelize:
        # TODO: Implement parallelization
        raise NotImplementedError("Parallelization not implemented yet.")

    for method in methods:
        for zone in zones:
            zone_log_data_df = log_data_df[log_data_df["zone"] == zone]
            method_obj = _get_method(method, zone_log_data_df, kwargs)
            for policy, policy_path in tqdm.tqdm(zip(policies, policy_paths), total=len(policies)):

                score, additional_data = get_policy_score(method, method_obj,
                                                          policy, behavior_policy,
                                                          return_additional=True)
                if method not in policy_scores:
                    policy_scores[method] = {}
                if method not in all_additional_data:
                    all_additional_data[method] = {}
                if policy_path not in all_additional_data[method]:
                    all_additional_data[method][policy_path] = {}

                if policy_path not in policy_scores[method]:
                    policy_scores[method][policy_path] = {}
                policy_scores[method][policy_path][zone] = score
                all_additional_data[method][policy_path][zone] = additional_data

    for method in methods:
        with open(os.path.join(save_path, method, "policy_scores.pkl"), "wb") as f:
            pickle.dump(policy_scores[method], f)
        with open(os.path.join(save_path, method, "additional_data.pkl"), "wb") as f:
            pickle.dump(all_additional_data[method], f)


def _get_method(method, log_data, kwargs):
    if "rule_based_behavior_policy" not in kwargs:
        kwargs["rule_based_behavior_policy"] = True
    if method == OPEMethodStrings.ipw.value:
        method_obj = \
            InverseProbabilityWeighting(log_data, no_grad=True)
    elif method == OPEMethodStrings.snip.value:
        if "log_data" in kwargs:
            del kwargs["log_data"]
        method_obj = SNIP(log_data, **kwargs)
    elif method == OPEMethodStrings.snipw.value:
        method_obj = SelfNormalizedInverseProbabilityWeighting(log_data)
    elif method == OPEMethodStrings.gk.value:
        method_obj = GaussianKernel(log_data=log_data,
                                    bandwidth=kwargs["gaussian_kernel_bandwidth"])
    else:
        raise NotImplementedError(f"Method {method} not implemented yet.")

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
                     method_obj: object,
                     policy: object,
                     behavior_policy: object,
                     return_additional: bool = True):

    additional_data = {}

    if method == "ipw":
        scaled_rewards, states, rewards, action_probs, behavior_probs = \
            method_obj.evaluate_policy(evaluation_policy_distribution=policy.get_distribution,
                                       behavior_policy=behavior_policy,
                                       return_additional=True)
        score = scaled_rewards.mean()
        additional_data["scaled_rewards"] = scaled_rewards.detach().numpy()
        additional_data["action_probs"] = action_probs.detach().numpy()
        additional_data["behavior_probs"] = behavior_probs.detach().numpy()

    elif method == "snipw":
        scaled_rewards, states, rewards, action_probs, behavior_probs = \
            method_obj.evaluate_policy(evaluation_policy_distribution=policy.get_distribution,
                                       behavior_policy=behavior_policy,
                                       return_additional=True)
        score = scaled_rewards.mean()
        additional_data["scaled_rewards"] = scaled_rewards.detach().numpy()
        additional_data["action_probs"] = action_probs.detach().numpy()
        additional_data["behavior_probs"] = behavior_probs.detach().numpy()

    elif method == "snip":
        score, losses, scaled_rewards, states, rewards, action_probs, behavior_probs = \
            method_obj.evaluate_policy(evaluation_policy=policy,
                                       behavior_policy=behavior_policy,
                                       return_additional=True)
        additional_data["losses"] = losses.detach().numpy()
        additional_data["scaled_rewards"] = scaled_rewards.detach().numpy()
        additional_data["action_probs"] = action_probs.detach().numpy()
        additional_data["behavior_probs"] = behavior_probs.detach().numpy()

    elif method == "gk":
        score = \
            method_obj.evaluate_policy(policy)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    if return_additional:
        return score, additional_data
    else:
        return score
