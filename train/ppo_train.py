from algorithms.ppo import MultiAgentPPO
from buildingenvs import BuildingEnvStrings
from buildingenvs import TypeABuilding
from buildingenvs import DOOEBuilding
from buildingenvs import FiveZoneBuilding
from policies.multiagentpolicy import MultiAgentACPolicy
from policies.utils.policy_extractor import save_zone_policies

from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
import typer
import os
import yaml


app = typer.Typer()


@app.command("optimal")
def train_optimal(building_env: BuildingEnvStrings =
                            typer.Option(BuildingEnvStrings.denver.value,
                                         help="The building environment to train agents on"),
                  building_config_loc: str =
                            typer.Option("configs/buildingconfig/building_denver.yaml",
                                         help="The location of the building config file"),
                  run_name: str = typer.Option("denvertest",
                                               help="The name of the run"),
                  log_dir: str = typer.Option("data/", help="The location to save the logs"),
                  train_year: int = typer.Option(1991, help="The year to train on"),
                  train_month: int = typer.Option(1, help="The month to train on"),
                  train_day: int = typer.Option(1, help="The day to train on"),
                  num_train_days: int = typer.Option(30, help="The number of days to train on"),
                  model_save_freq: int = typer.Option(2880, help="The frequency to save the model"),
                  normalize_advantage: bool =
                            typer.Option(False, help="Whether to normalize the calculated advantage"),
                  ent_coef: float = typer.Option(0.01, help="The entropy coefficient"),
                  num_episodes: int =
                            typer.Option(500, help="The number of episodes to train for"),
                  batch_size: int = typer.Option(32, help="The batch size"),
                  seed: int = typer.Option(1337, help="The seed for the environment"),
                  device: str = typer.Option("cpu", help="The device to train on"),
                  energy_plus_loc: str = typer.Option("/Applications/EnergyPlus-9-3-0/",
                                         help="The location of the energyplus executable")
                  ):
    train(building_env=building_env,
          building_config_loc=building_config_loc,
          run_name=run_name,
          log_dir=log_dir,
          train_year=train_year,
          train_month=train_month,
          train_day=train_day,
          num_train_days=num_train_days,
          model_save_freq=model_save_freq,
          normalize_advantage=normalize_advantage,
          ent_coef=ent_coef,
          num_episodes=num_episodes,
          batch_size=batch_size,
          seed=seed,
          device=device,
          energy_plus_loc=energy_plus_loc,
          diverse_training=False)


@app.command("diverse")
def train_diverse(diverse_policy_library_loc: str =
                            typer.Option("data/policies/",
                                         help="The location of the policy library"),
                  diversity_weight: float = typer.Option(0.01, help="The weight of diversity"),
                  building_env: BuildingEnvStrings =
                            typer.Option(BuildingEnvStrings.denver.value,
                                         help="The building environment to train agents on"),
                  building_config_loc: str =
                            typer.Option("configs/buildingconfig/building_denver.yaml",
                                         help="The location of the building config file"),
                  run_name: str = typer.Option("denvertest",
                                               help="The name of the run"),
                  log_dir: str = typer.Option("data/", help="The location to save the logs"),
                  train_year: int = typer.Option(1991, help="The year to train on"),
                  train_month: int = typer.Option(1, help="The month to train on"),
                  train_day: int = typer.Option(1, help="The day to train on"),
                  num_train_days: int = typer.Option(30, help="The number of days to train on"),
                  model_save_freq: int = typer.Option(2880, help="The frequency to save the model"),
                  normalize_advantage: bool =
                            typer.Option(False, help="Whether to normalize the calculated advantage"),
                  ent_coef: float = typer.Option(0.01, help="The entropy coefficient"),
                  num_episodes: int =
                            typer.Option(500, help="The number of episodes to train for"),
                  batch_size: int = typer.Option(32, help="The batch size"),
                  seed: int = typer.Option(1337, help="The seed for the environment"),
                  device: str = typer.Option("cpu", help="The device to train on"),
                  energy_plus_loc: str = typer.Option("/Applications/EnergyPlus-9-3-0/",
                                         help="The location of the energyplus executable")):
    train(building_env=building_env,
          building_config_loc=building_config_loc,
          run_name=run_name,
          log_dir=log_dir,
          train_year=train_year,
          train_month=train_month,
          train_day=train_day,
          num_train_days=num_train_days,
          model_save_freq=model_save_freq,
          normalize_advantage=normalize_advantage,
          ent_coef=ent_coef,
          num_episodes=num_episodes,
          batch_size=batch_size,
          seed=seed,
          device=device,
          energy_plus_loc=energy_plus_loc,
          diverse_training=True,
          diverse_policy_library_loc=diverse_policy_library_loc,
          diversity_weight=diversity_weight)


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
    if building_env.value in ["denver", "sf"]:
        env = TypeABuilding(config, log_dir, energy_plus_loc, logger)
    elif building_env.value == "dooe":
        env = DOOEBuilding(config, log_dir, energy_plus_loc, logger)
    elif building_env.value == "five_zone":
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
    # callbacks = [checkpoint_callback, total_energy_callback]
    callbacks = [checkpoint_callback]
    return callbacks


def list_diverse_policies(diverse_policy_library_loc):
    return sorted(os.listdir(diverse_policy_library_loc))


def get_model(env, args, config):
    n_steps = config["timesteps_per_hour"] * 24 * args["num_train_days"]
    if args["diverse_training"]:
        diverse_policies = list_diverse_policies(args["diverse_policy_library_loc"])
        model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                              diverse_policies=diverse_policies,
                              diversity_weight=args["diversity_weight"],
                              n_steps=n_steps, batch_size=args["batch_size"],
                              device=args["device"], policy_kwargs={"control_zones": env.control_zones,
                                                                "device": args["device"]},
                              normalize_advantage=args["normalize_advantage"],
                              ent_coef=args["ent_coef"])
    else:
        model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                              n_steps=n_steps, batch_size=args["batch_size"],
                              device=args["device"], policy_kwargs={"control_zones": env.control_zones,
                                                                    "device": args["device"]},
                              normalize_advantage=args["normalize_advantage"],
                              ent_coef=args["ent_coef"])
    return model


def train(**kwargs):
    set_random_seed(kwargs["seed"], using_cuda="cuda" in kwargs["device"])

    log_dir, model_dir = get_log_dirs(kwargs["log_dir"], kwargs["run_name"])

    logger = get_logger(log_dir)

    env, config = get_env(kwargs["building_env"], kwargs["building_config_loc"],
                          log_dir, kwargs["energy_plus_loc"], kwargs, logger)

    save_configs(log_dir, kwargs, config)

    callbacks = get_callbacks(model_dir, kwargs)

    model = get_model(env, kwargs, config)
    model.set_logger(logger)

    total_timesteps = kwargs["num_episodes"] * config["timesteps_per_hour"] * 24 * kwargs["num_train_days"]
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks,
                tb_log_name=kwargs["run_name"])
    model.save(os.path.join(model_dir, "final_model"))
    save_zone_policies(model.policy, save_dir=os.path.join(model_dir, "finalSplitPolicies"))


if __name__ == "__main__":
    train(
        building_env="dooe",
        building_config_loc="data/building_config/dooe.yaml",
        run_name="dooe",
        log_dir="data/logs",
        train_year=1991,
        train_month=1,
        train_day=1,
        num_train_days=30,
        model_save_freq=2880,
        normalize_advantage=True,
        ent_coef=0.01,
        num_episodes=500,
        batch_size=32,
        seed=1337,
        device="cpu",
        energy_plus_loc="/Applications/EnergyPlus-9-3-0/",
    )
