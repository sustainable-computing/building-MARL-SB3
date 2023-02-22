from buildingenvs import BuildingEnvStrings
from buildingenvs import TypeABuilding
from buildingenvs import DOOEBuilding
from buildingenvs import FiveZoneBuilding
from policies.utils import load_policy_library

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
                                                 eval_mode=True)
    total_energy_consumptions = {}
    for zone in zones:
        zone_env, config = _get_zone_env(building_env, building_config_loc, zone,
                                         save_path, energy_plus_loc)
        zone_env.set_runperiod(num_days, start_year, start_month, start_day)
        zone_env.set_timestep(config["timesteps_per_hour"])

        for policy, policy_path in tqdm.tqdm(zip(policies, policy_paths), total=len(policies)):
            for ep in range(1):
                state = zone_env.reset()
                state = state[zone]
                while not zone_env.is_terminate():
                    # print(state["Perimeter_top_ZN_1 vav energy"])
                    with th.no_grad():
                        policy_action, _, _ = policy(state)
                    state, _, _, _ = zone_env.step(policy_action)
                    state = state[zone]
                if zone not in total_energy_consumptions:
                    total_energy_consumptions[zone] = {}
                total_energy_consumptions[zone][policy_path] = zone_env.total_energy_consumption
    _save_energy_consumptions(save_path, total_energy_consumptions)   


def load_building_config(building_config_loc):
    with open(building_config_loc, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _save_energy_consumptions(save_path, total_energy_consumptions):
    with open(os.path.join(save_path, "raw_energy_consumptions.pkl"), "wb+") as f:
        pickle.dump(total_energy_consumptions, f)

    with open(os.path.join(save_path, "evaluation_report.csv", "w+")) as f:
        for zone in total_energy_consumptions:
            for policy in total_energy_consumptions[zone]:
                f.write(f"{zone},{policy},{total_energy_consumptions[zone][policy]}\n")


def _get_zone_env(building_env, building_config_loc, zone,
                  log_dir, energy_plus_loc):
    config = load_building_config(building_config_loc)
    config["control_zones"] = [zone]

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
