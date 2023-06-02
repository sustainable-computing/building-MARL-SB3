from buildingenvs import BuildingEnvStrings
from buildingenvs import TypeABuilding
from buildingenvs import DOOEBuilding
from buildingenvs import FiveZoneBuilding
from evaluation.offline.evaluate import _get_method, get_policy_score
from policies.utils import load_policy_library, load_policy
from utils.configs import load_config
from utils.metrics import convert_score_dict_to_df

from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
from stable_baselines3.common.utils import set_random_seed
import tqdm
import torch as th
import yaml


def evaluate_policies(
    building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
    building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
    zone: str = None,
    policy_library_path: str = "data/policy_libraries/policy_library_20220820",
    policy_type: str = "single_agent_ac",
    init_policy_log_std: float = np.log(0.1),
    init_policy_log_std_path: str = "",
    num_days: int = 365,
    start_day: int = 1,
    start_month: int = 1,
    start_year: int = 1995,
    save_path: str = "data/policy_evaluation/brute_force/",
    energy_plus_loc: str = "/Applications/EnergyPlus-9-3-0/",
    parallelize: bool = False,
    max_cpu_cores: int = 1,
    seed: int = 1337,
    device: str = "cpu"
):
    kwargs = locals()
    set_random_seed(seed)
    save_path = _create_save_path(save_path)
    _save_run_config(save_path, kwargs)

    building_config = load_building_config(building_config_loc)
    if not zone:
        zones = building_config["control_zones"]
    else:
        zones = [zone]

    policies, policy_paths = load_policy_library(policy_library_path, policy_type,
                                                 init_policy_log_std, init_policy_log_std_path,
                                                 eval_mode=False, device=device)
    total_energy_consumptions = {}
    for zone in zones:
        zone_env, config = _get_zone_env(building_env, building_config_loc, zone,
                                         save_path, energy_plus_loc)
        zone_env.set_runperiod(num_days, start_year, start_month, start_day)
        zone_env.set_timestep(config["timesteps_per_hour"])

        for policy, policy_path in tqdm.tqdm(zip(policies, policy_paths), total=len(policies)):
            state = zone_env.reset()
            state = state[zone]
            while not zone_env.is_terminate():
                with th.no_grad():
                    policy_action, _, _ = policy(state)
                state, _, _, _ = zone_env.step(policy_action.cpu().numpy())
                state = state[zone]
            if zone not in total_energy_consumptions:
                total_energy_consumptions[zone] = {}
            total_energy_consumptions[zone][policy_path] = zone_env.total_energy_consumption
    _save_energy_consumptions(save_path, total_energy_consumptions)


def evaluate_rule_based(
    building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
    building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
    zone: str = None,
    num_days: int = 365,
    start_day: int = 1,
    start_month: int = 1,
    start_year: int = 1995,
    save_path: str = "data/policy_evaluation/brute_force/",
    energy_plus_loc: str = "/Applications/EnergyPlus-9-3-0/",
    parallelize: bool = False,
    max_cpu_cores: int = 1,
    seed: int = 1337,
    policy_default_action: float = 0.3
):
    kwargs = locals()
    set_random_seed(seed)
    save_path = _create_save_path(save_path)
    _save_run_config(save_path, kwargs)

    building_config = load_building_config(building_config_loc)
    if not zone:
        zones = building_config["control_zones"]
    else:
        zones = [zone]

    total_energy_consumptions = {}
    log_data = []
    env, config = _get_zone_env(building_env, building_config_loc, zones,
                                save_path, energy_plus_loc)
    env.set_runperiod(num_days, start_year, start_month, start_day)
    env.set_timestep(config["timesteps_per_hour"])

    state = env.reset()
    while not env.is_terminate():
        policy_action = th.Tensor([policy_default_action]*len(zones))
        state, rewards, _, info = env.step(policy_action)
        cobs_state = info["cobs_state"]
        for zone in zones:
            log_data.append({
                "time": cobs_state["time"],
                "timestep": cobs_state["timestep"],
                "zone": zone,
                "outdoor_temp": cobs_state["outdoor temperature"],
                "solar_irradiation": cobs_state["site solar radiation"],
                "time_hour": cobs_state["time"].hour,
                "zone_humidity": cobs_state[f"{zone} humidity"],
                "zone_temp": cobs_state["temperature"][zone],
                "zone_occupancy": cobs_state["occupancy"][zone],
                "action":  cobs_state[f"{zone} position"],
                "reward": rewards[zones.index(zone)],
                "reward_total_hvac": cobs_state["total hvac"],
                "cooling_energy": cobs_state[f"{zone} Air System Sensible Cooling Energy"],
                "heating_energy": cobs_state[f"{zone} Air System Sensible Heating Energy"],
            })
    total_energy_consumptions[zone] = env.total_energy_consumption
    _save_energy_consumptions(save_path, total_energy_consumptions)
    _save_log_data(save_path, log_data)


def run_full_simulation(
    building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
    building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
    policy_map_config_loc: str = "configs/policymapconfigs/denver/gt_best_one_year_denver.yaml",
    save_path: str = "data/policy_evaluation/brute_force/",
    energy_plus_loc: str = "/Applications/EnergyPlus-9-3-0/",
    seed: int = 1337,
    device: str = "cpu"
):
    building_config = load_building_config(building_config_loc)
    zones = building_config["control_zones"]

    policy_map_config = load_config(policy_map_config_loc)

    policy_map = {month: {} for month in range(1, 13)}
    if -1 in policy_map_config["zone_policy_map"]:
        for month in policy_map:
            policy_map[month] = policy_map_config["zone_policy_map"][-1]
    else:
        for month in policy_map:
            if month in policy_map_config["zone_policy_map"]:
                month_map_config = policy_map_config["zone_policy_map"][month]
                assert set(month_map_config.keys()) == set(zones), \
                    f"Policy map config for month {month} does not contain all zones"
                policy_map[month] = month_map_config
            else:
                raise ValueError(f"Policy map config does not contain month {month}")

    kwargs = locals()
    set_random_seed(seed)
    save_path = _create_save_path(save_path)
    _save_run_config(save_path, kwargs)

    policy_map = load_policy_map(policy_map, device)

    env, config = _get_zone_env(building_env, building_config_loc, zones,
                                save_path, energy_plus_loc)

    num_days, start_year, start_month, start_day = \
        policy_map_config["num_days"], policy_map_config["start_year"], \
        policy_map_config["start_month"], policy_map_config["start_day"]
    env.set_runperiod(num_days, start_year, start_month, start_day)
    env.set_timestep(config["timesteps_per_hour"])

    total_energy_consumptions = {}
    log_data = []
    state, info = env.reset(return_info=True)
    while not env.is_terminate():
        actions = []
        month = info["cobs_state"]["time"].month
        for zone in zones:
            policy = policy_map[month][zone]["policy_obj"]
            with th.no_grad():
                policy_action, _, _ = policy(th.tensor(state[zone], device=device))
            actions.append(policy_action)
        actions = th.tensor(actions, device=device)
        state, rewards, _, info = env.step(actions)
        cobs_state = info["cobs_state"]
        for zone in zones:
            log_data.append({
                "time": cobs_state["time"],
                "timestep": cobs_state["timestep"],
                "zone": zone,
                "outdoor_temp": cobs_state["outdoor temperature"],
                "solar_irradiation": cobs_state["site solar radiation"],
                "time_hour": cobs_state["time"].hour,
                "zone_humidity": cobs_state[f"{zone} humidity"],
                "zone_temp": cobs_state["temperature"][zone],
                "zone_occupancy": cobs_state["occupancy"][zone],
                "action":  cobs_state[f"{zone} position"],
                "reward": rewards[zones.index(zone)],
                "reward_total_hvac": cobs_state["total hvac"],
            })
    total_energy_consumptions[zone] = env.total_energy_consumption
    _save_energy_consumptions(save_path, total_energy_consumptions)
    _save_log_data(save_path, log_data)


def run_full_automated_swapping_simulation(
    building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
    building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
    policy_library_path: str = "data/policy_libraries/policy_library_20220820",
    policy_type: str = "single_agent_ac",
    init_policy_log_std: float = np.log(0.1),
    init_policy_log_std_path: str = "",
    policy_map_config_loc: str = "configs/policymapconfigs/denver/gt_best_one_year_denver.yaml",
    save_path: str = "data/policy_evaluation/brute_force/",
    energy_plus_loc: str = "/Applications/EnergyPlus-9-3-0/",
    seed: int = 1337,
    device: str = "cpu"
):
    building_config = load_building_config(building_config_loc)
    zones = building_config["control_zones"]

    policy_map_config = load_config(policy_map_config_loc)
    assert len(policy_map_config["zone_policy_map"]) == 1, \
        "Automated swapping simulation only takes in one month assignment" \
        "cannot assign policies for multiple months"

    starting_month = list(policy_map_config["zone_policy_map"].keys())[0]
    assert starting_month in range(1, 13), \
        "Starting month must be between 1 and 12"
    policy_map = {starting_month: {}}
    month_map_config = policy_map_config["zone_policy_map"][starting_month]
    assert set(month_map_config.keys()) == set(zones), \
        f"Policy map config for month {starting_month} does not contain all zones"
    policy_map[starting_month] = month_map_config

    kwargs = locals()
    set_random_seed(seed)
    save_path = _create_save_path(save_path)
    _save_run_config(save_path, kwargs)

    policy_map = load_policy_map(policy_map, device)

    env, config = _get_zone_env(building_env, building_config_loc, zones,
                                save_path, energy_plus_loc)

    num_days, start_year, start_month, start_day = \
        policy_map_config["num_days"], policy_map_config["start_year"], \
        policy_map_config["start_month"], policy_map_config["start_day"]
    assert start_month == starting_month, \
        "Starting month in policy map config does not match starting month in run config"
    env.set_runperiod(num_days, start_year, start_month, start_day)
    env.set_timestep(config["timesteps_per_hour"])

    total_energy_consumptions = {}
    log_data = []
    state, info = env.reset(return_info=True)
    while not env.is_terminate():
        actions = []
        month = info["cobs_state"]["time"].month
        if month not in policy_map:
            policy_map[month] = _estimate_policy_map(month-1, policy_map[month-1],
                                                     log_data, policy_map_config,
                                                     policy_library_path, policy_type,
                                                     init_policy_log_std,
                                                     init_policy_log_std_path, device,
                                                     save_path,
                                                     kwargs)
        for zone in zones:
            policy = policy_map[month][zone]["policy_obj"]
            with th.no_grad():
                policy_action, _, _ = policy(th.tensor(state[zone], device=device))
            actions.append(policy_action)
        actions = th.tensor(actions, device=device)
        state, rewards, _, info = env.step(actions)
        cobs_state = info["cobs_state"]
        for i, zone in enumerate(zones):
            log_data.append({
                "time": cobs_state["time"],
                "timestep": cobs_state["timestep"],
                "zone": zone,
                "outdoor_temp": cobs_state["outdoor temperature"],
                "solar_irradiation": cobs_state["site solar radiation"],
                "time_hour": cobs_state["time"].hour,
                "zone_humidity": cobs_state[f"{zone} humidity"],
                "zone_temp": cobs_state["temperature"][zone],
                "zone_occupancy": cobs_state["occupancy"][zone],
                "action":  actions[i].item(),
                "reward": rewards[zones.index(zone)],
                "reward_total_hvac": cobs_state["total hvac"],
            })
    total_energy_consumptions[zone] = env.total_energy_consumption
    _save_energy_consumptions(save_path, total_energy_consumptions)
    _save_log_data(save_path, log_data)


def load_policy_map(policy_map: dict, device: str = "cpu"):
    for month in policy_map:
        for zone in policy_map[month]:
            policy_path = policy_map[month][zone]["policy"]
            policy_init_log_std = policy_map[month][zone]["init_log_std"]
            policy_obj = load_policy(policy_path, "single_agent_ac",
                                     policy_init_log_std, "", True,
                                     device)
            policy_map[month][zone]["policy_obj"] = policy_obj
    return policy_map


def load_building_config(building_config_loc):
    with open(building_config_loc, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _estimate_policy_map(prev_month_num, prev_policy_map,
                         log_data, policy_map_config,
                         policy_library_path,
                         policy_type, init_policy_log_std,
                         init_policy_log_std_path,
                         device, save_path,
                         kwargs):

    best_estimated_policy_map = {}

    log_data_df = pd.DataFrame.from_records(log_data)
    log_data_df["time"] = pd.to_datetime(log_data_df["time"])
    log_data_df = log_data_df[log_data_df["time"].dt.month == prev_month_num]

    policies, policy_paths = load_policy_library(policy_library_path, policy_type,
                                                 init_policy_log_std,
                                                 init_policy_log_std_path,
                                                 eval_mode=True, device=device)

    ope_method = policy_map_config["ope_method"]
    policy_scores = {ope_method: {}}
    additional_data_dict = {ope_method: {}}
    kwargs["rule_based_behavior_policy"] = False
    for zone in prev_policy_map:
        zone_log_data_df = log_data_df[log_data_df["zone"] == zone]
        method_obj = _get_method(ope_method, zone_log_data_df, kwargs)
        behavior_policy = prev_policy_map[zone]["policy_obj"]

        if zone not in policy_scores:
            additional_data_dict[ope_method][zone] = {}
            best_estimated_policy_map[zone] = {}

        for policy, policy_path in tqdm.tqdm(zip(policies, policy_paths), total=len(policies)):
            score, additional_data = get_policy_score(ope_method, method_obj,
                                                      policy, behavior_policy,
                                                      return_additional=True)
            # print(zone, policy_path, score)
            if policy_path not in policy_scores:
                policy_scores[ope_method][policy_path] = {}
            policy_scores[ope_method][policy_path][zone] = score
            additional_data_dict[ope_method][zone][policy_path] = additional_data
        zone_score_df = convert_score_dict_to_df(policy_scores)[ope_method]
        # print(zone_score_df)
        sorted_scores_df = zone_score_df.sort_values(by="score")
        best_zone_policy = sorted_scores_df["policy"].values[0]
        best_zone_policy_idx = policy_paths.index(best_zone_policy)
        best_zone_policy_obj = policies[best_zone_policy_idx]

        best_estimated_policy_map[zone]["policy"] = best_zone_policy
        best_estimated_policy_map[zone]["policy_obj"] = best_zone_policy_obj

    log_data_save_path = os.path.join(save_path, "monthly_policy_ranking/")
    if not os.path.exists(log_data_save_path):
        os.makedirs(log_data_save_path)
    with open(os.path.join(log_data_save_path, f"policy_scores_{prev_month_num}.pkl"), "wb+") as f:
        pickle.dump(policy_scores, f)
    with open(os.path.join(log_data_save_path, f"additional_data_{prev_month_num}.pkl"), "wb+") as f:
        pickle.dump(additional_data_dict, f)

    return best_estimated_policy_map


def _save_energy_consumptions(save_path, total_energy_consumptions):
    with open(os.path.join(save_path, "raw_energy_consumptions.pkl"), "wb+") as f:
        pickle.dump(total_energy_consumptions, f)

    with open(os.path.join(save_path, "evaluation_report.csv"), "w+") as f:
        for zone in total_energy_consumptions:
            if isinstance(total_energy_consumptions[zone], dict):
                for policy in total_energy_consumptions[zone]:
                    f.write(f"{zone},{policy},{total_energy_consumptions[zone][policy]}\n")
            else:
                f.write(f"{zone},{total_energy_consumptions[zone]}\n")


def _save_log_data(save_path, log_data):
    df = pd.DataFrame(log_data)
    df.to_csv(os.path.join(save_path, "log_data.csv"), index=False)


def _get_zone_env(building_env, building_config_loc, zone,
                  log_dir, energy_plus_loc):
    config = load_building_config(building_config_loc)

    if zone is not None:
        if isinstance(zone, str):
            config["control_zones"] = [zone]
        elif isinstance(zone, list):
            config["control_zones"] = zone

    if hasattr(building_env, "value"):
        building_env = building_env.value

    if building_env in ["denver", "sf"]:
        env = TypeABuilding(config, log_dir, energy_plus_loc, logger=None)
    elif building_env == "dooe":
        env = DOOEBuilding(config, log_dir, energy_plus_loc, logger=None)
    elif building_env == "five_zone":
        env = FiveZoneBuilding(config, log_dir, energy_plus_loc, logger=None)
    else:
        raise ValueError("Building environment not supported")

    return env, config


def _create_save_path(save_dir):
    save_path = os.path.join(save_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(save_path)
    return save_path


def _save_run_config(save_path, args):
    with open(os.path.join(save_path, "run_config.yaml"), "w") as f:
        yaml.dump(args, f)

