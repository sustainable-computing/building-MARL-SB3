from algorithms.ppo import MultiAgentPPO
from buildingenvs import BuildingEnvStrings
from buildingenvs import TypeABuilding
from buildingenvs import DOOEBuilding
from buildingenvs import FiveZoneBuilding
from callbacks.statevisitcallback import StateVisitCallback
from policies.multiagentpolicy import MultiAgentACPolicy
from policies.utils.policy_extractor import save_zone_policies

from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml


def get_log_dirs(log_dir, run_name):
    current_dt = datetime.now()
    dt_str = current_dt.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, run_name, dt_str)
    model_save_dir = os.path.join(log_dir, "model")
    return log_dir, model_save_dir


def get_logger(log_dir):
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    return logger


def load_building_config(building_config_loc):
    with open(building_config_loc, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_env(building_env, building_config_loc,
            log_dir, energy_plus_loc, args,
            logger):
    config = load_building_config(building_config_loc)

    if hasattr(building_env, "value"):
        building_env = building_env.value

    if building_env in ["denver", "sf"]:
        env = TypeABuilding(config, log_dir, energy_plus_loc, logger)
    elif building_env == "dooe":
        env = DOOEBuilding(config, log_dir, energy_plus_loc, logger)
    elif building_env == "five_zone":
        env = FiveZoneBuilding(config, log_dir, energy_plus_loc, logger)
    else:
        raise ValueError("Building environment not supported")

    env.set_runperiod(args["num_train_days"], args["train_year"], args["train_month"], args["train_day"])
    env.set_timestep(config["timesteps_per_hour"])
    return env, config


def save_configs(log_dir, args, building_config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "run_config.yaml"), "w") as f:
        yaml.dump(args, f)
    with open(os.path.join(log_dir, "building_config.yaml"), "w") as f:
        yaml.dump(building_config, f)


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


def get_model(env, args, config):
    n_steps = config["timesteps_per_hour"] * 24 * args["num_train_days"]
    if args["diverse_training"]:
        model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                              diverse_policy_library_loc=args["diverse_policy_library_loc"],
                              diversity_weight=args["diversity_weight"],
                              diverse_policy_library_log_std_loc=args["diverse_policy_library_log_std_loc"],
                              n_steps=n_steps, batch_size=args["batch_size"],
                              device=args["device"],
                              policy_kwargs={"control_zones": env.control_zones,
                                             "device": args["device"]},
                              normalize_advantage=args["normalize_advantage"],
                              ent_coef=args["ent_coef"],
                              learning_rate=args["learning_rate"])
    else:
        model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                              n_steps=n_steps, batch_size=args["batch_size"],
                              device=args["device"],
                              policy_kwargs={"control_zones": env.control_zones,
                                             "device": args["device"]},
                              normalize_advantage=args["normalize_advantage"],
                              ent_coef=args["ent_coef"],
                              learning_rate=args["learning_rate"])
    return model


def init_wandb(project_name, config):
    run = wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )
    print("WANDB URL: ", run.get_url())
    return run


def train(building_env: BuildingEnvStrings = BuildingEnvStrings.denver,
          building_config_loc: str = "configs/buildingconfig/building_denver.yaml",
          run_name: str = "denvertest",
          log_dir: str = "data/trainlogs/",
          train_year: int = 1991,
          train_month: int = 1,
          train_day: int = 1,
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
          wandb_project_name: str = "ppo-train"):

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

    total_timesteps = kwargs["num_episodes"] * config["timesteps_per_hour"] * 24 * kwargs["num_train_days"]
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks,
                tb_log_name=kwargs["run_name"])
    model.save(os.path.join(model_dir, "final_model"))
    save_zone_policies(model.policy, save_dir=os.path.join(model_dir, "finalSplitPolicies"))
    if use_wandb:
        run.finish()


if __name__ == "__main__":
    train()
