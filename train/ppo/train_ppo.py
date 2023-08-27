from algorithms.ppo import MultiAgentPPO
from buildingenvs import BuildingEnvStrings
from buildingenvs import TypeABuilding
from buildingenvs import DOOEBuilding
from buildingenvs import FiveZoneBuilding
from callbacks.statevisitcallback import StateVisitCallback
from policies.multiagentpolicy import MultiAgentACPolicy
from policies.utils.policy_extractor import save_zone_policies
from utils.logs import create_log_dir
from utils.configs import load_config, save_config

from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
import torch as th
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml


def get_log_dirs(log_dir, run_name):
    log_dir = create_log_dir(log_dir, run_name)
    model_save_dir = create_log_dir(log_dir, "model")

    return log_dir, model_save_dir


def get_logger(log_dir):
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    return logger


def load_building_config(building_config_loc):
    return load_config(building_config_loc)


def get_env(building_env, building_config_loc,
            log_dir, energy_plus_loc, args,
            logger):
    config = load_building_config(building_config_loc)

    if hasattr(building_env, "value"):
        building_env = building_env.value

    if building_env in ["denver", "sf"]:
        env = TypeABuilding(config, log_dir, energy_plus_loc, logger, args["reward_signal"])
    elif building_env == "dooe":
        env = DOOEBuilding(config, log_dir, energy_plus_loc, logger, args["reward_signal"])
    elif building_env == "five_zone":
        env = FiveZoneBuilding(config, log_dir, energy_plus_loc, logger, args["reward_signal"])
    else:
        raise ValueError("Building environment not supported")

    env.set_runperiod(args["run_period"], args["train_year"], args["train_month"], args["train_day"])
    env.set_timestep(config["timesteps_per_hour"])
    return env, config


def save_configs(log_dir, args, building_config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    save_config(args, os.path.join(log_dir, "run_config.yaml"))
    save_config(building_config, os.path.join(log_dir, "building_config.yaml"))


def create_checkpoint_callback(save_freq, log_dir, prefix="b_denver"):
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=log_dir,
        name_prefix=prefix,
        save_replay_buffer=True,
    )
    return checkpoint_callback


def get_callbacks(log_dir, args):
    checkpoint_callback = create_checkpoint_callback(args["model_save_freq"],
                                                     log_dir, prefix=args["run_name"])
    # total_energy_callback = TotalEnergyCallback()
    state_visit_callback = StateVisitCallback(log_dir)
    # callbacks = [checkpoint_callback, total_energy_callback]
    callbacks = [checkpoint_callback, state_visit_callback]
    return callbacks


def get_model(env, kwargs, config):
    n_steps = config["timesteps_per_hour"] * 24 * kwargs["num_train_days"]
    if "diverse_training" not in kwargs.keys():
        kwargs["diverse_training"] = False
    if kwargs["diverse_training"]:
        model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                              diverse_policy_library_loc=kwargs["diverse_policy_library_loc"],
                              diversity_weight=kwargs["diversity_weight"],
                              diverse_policy_library_log_std_loc=kwargs["diverse_policy_library_log_std_loc"],
                              n_steps=n_steps, batch_size=kwargs["batch_size"],
                              device=kwargs["device"],
                              policy_kwargs={"control_zones": env.control_zones,
                                             "device": kwargs["device"],
                                             "retrain": kwargs["retrain"],
                                             "policy_map_config_loc": kwargs["policy_map_config_loc"]},
                              normalize_advantage=kwargs["normalize_advantage"],
                              ent_coef=kwargs["ent_coef"],
                              learning_rate=kwargs["learning_rate"])
    else:
        model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                              n_steps=n_steps, batch_size=kwargs["batch_size"],
                              device=kwargs["device"],
                              policy_kwargs={"control_zones": env.control_zones,
                                             "device": kwargs["device"],
                                             "retrain": kwargs["retrain"],
                                             "policy_map_config_loc": kwargs["policy_map_config_loc"]},
                              normalize_advantage=kwargs["normalize_advantage"],
                              ent_coef=kwargs["ent_coef"],
                              learning_rate=kwargs["learning_rate"])
    return model


def init_wandb(project_name, config):
    if config["wandb_run_name"] == "":
        name = None
    else:
        name = config["wandb_run_name"]
    run = wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        name=name
    )
    print("WANDB URL: ", run.get_url())
    return run


def train(building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
          building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
          run_name: str = "",
          log_dir: str = "data/trainlogs/",
          train_year: int = 1991,
          train_month: int = 1,
          train_day: int = 1,
          reward_signal: str = "standard",
          num_train_days: int = 30,
          model_save_freq: int = 2880,
          normalize_advantage: bool = False,
          ent_coef: float = 0.01,
          learning_rate: float = 0.0003,
          num_episodes: int = 500,
          batch_size: int = 32,
          seed: int = 1337,
          device: str = "cpu",
          energy_plus_loc: str = "/Applications/EnergyPlus-9-3-0/",
          diverse_training: bool = False,
          diverse_policy_library_loc: str = "data/diverse_policies/",
          diverse_policy_library_log_std_loc: str = "",
          diversity_weight: float = 0.1,
          use_wandb: bool = False,
          wandb_project_name: str = "ppo-train",
          wandb_run_name: str = "",
          torch_compile: bool = False,
          **kwargs):

    retrain=False
    policy_map_config_loc = None
    run_period = num_train_days
    kwargs = locals()
    set_random_seed(kwargs["seed"], using_cuda="cuda" in kwargs["device"])

    log_dir, model_dir = get_log_dirs(kwargs["log_dir"], kwargs["run_name"])

    logger = get_logger(log_dir)

    env, config = get_env(kwargs["building_env"], kwargs["building_config_loc"],
                          log_dir, kwargs["energy_plus_loc"], kwargs, logger)

    save_configs(log_dir, kwargs, config)

    callbacks = get_callbacks(model_dir, kwargs)

    if use_wandb:
        kwargs["building_config"] = config
        run = init_wandb(kwargs["wandb_project_name"], kwargs)
        callbacks.append(WandbCallback(verbose=2, model_save_freq=kwargs["model_save_freq"],
                                       model_save_path=model_dir,
                                       gradient_save_freq=kwargs["model_save_freq"]))

    model = get_model(env, kwargs, config)
    model.set_logger(logger)

    if torch_compile:
        model = th.compile(model)

    total_timesteps = kwargs["num_episodes"] * config["timesteps_per_hour"] * 24 * kwargs["num_train_days"]
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks,
                tb_log_name=kwargs["run_name"])
    model.save(os.path.join(model_dir, "final_model"))
    save_zone_policies(model.policy, save_dir=os.path.join(model_dir, "finalSplitPolicies"))
    if use_wandb:
        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_dir(os.path.join(model_dir, "finalSplitPolicies"), name="finalSplitPolicies")
        artifact.add_file(os.path.join(model_dir, "final_model.zip"), name="final_model")
        run.log_artifact(artifact)
        run.finish()
        wandb.finish()

    return log_dir, model


def retrain_policies(
        building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
        building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
        run_name: str = "",
        log_dir: str = "data/trainlogs/",
        train_year: int = 1991,
        train_month: int = 1,
        train_day: int = 1,
        num_train_days: int = 30,
        run_period: int = 356,
        model_save_freq: int = 2880,
        policy_map_config_loc: str = "data/policy_map.json",
        normalize_advantage: bool = False,
        ent_coef: float = 0.01,
        learning_rate: float = 0.0003,
        num_episodes: int = 500,
        batch_size: int = 32,
        seed: int = 1337,
        device: str = "cpu",
        energy_plus_loc: str = "/Applications/EnergyPlus-9-3-0/",
        use_wandb: bool = False,
        wandb_project_name: str = "ppo-train",
        wandb_run_name: str = "",
        torch_compile: bool = False,
        retrain=True,
        **kwargs
):
    kwargs = locals()

    set_random_seed(kwargs["seed"], using_cuda="cuda" in kwargs["device"])

    log_dir, model_dir = get_log_dirs(kwargs["log_dir"], kwargs["run_name"])

    logger = get_logger(log_dir)

    env, config = get_env(kwargs["building_env"], kwargs["building_config_loc"],
                          log_dir, kwargs["energy_plus_loc"], kwargs, logger)

    save_configs(log_dir, kwargs, config)

    callbacks = get_callbacks(model_dir, kwargs)

    if use_wandb:
        kwargs["building_config"] = config
        run = init_wandb(kwargs["wandb_project_name"], kwargs)
        callbacks.append(WandbCallback(verbose=2, model_save_freq=kwargs["model_save_freq"],
                                       model_save_path=model_dir,
                                       gradient_save_freq=kwargs["model_save_freq"]))

    model = get_model(env, kwargs, config)
    model.set_logger(logger)

    if torch_compile:
        model = th.compile(model)

    total_timesteps = kwargs["num_episodes"] * config["timesteps_per_hour"] * 24 * kwargs["run_period"]
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks,
                tb_log_name=kwargs["run_name"])
    model.save(os.path.join(model_dir, "final_model"))
    save_zone_policies(model.policy, save_dir=os.path.join(model_dir, "finalSplitPolicies"))
    if use_wandb:
        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_dir(os.path.join(model_dir, "finalSplitPolicies"), name="finalSplitPolicies")
        artifact.add_file(os.path.join(model_dir, "final_model.zip"), name="final_model")
        run.log_artifact(artifact)
        run.finish()
        wandb.finish()

    return log_dir, model


if __name__ == "__main__":
    train()
