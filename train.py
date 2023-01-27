import argparse
from datetime import datetime
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import yaml


from algorithms.ppo import MultiAgentPPO
from buildingenvs import TypeABuilding
from buildingenvs import DOOEBuilding
from config import energy_plus_loc
from config import device
from policies.multiagentpolicy import MultiAgentACPolicy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--building_env",
                        type=str,
                        help="The building environment to train agents on",
                        choices=["denver", "sf", "doee"],
                        default="doee")

    parser.add_argument("--building_config_loc",
                        type=str,
                        help="The location of the building config file",
                        default="configs/buildingconfig/building_dooe.yaml")

    parser.add_argument("--run_name",
                        type=str,
                        help="The name of the run",
                        default="doeetest")

    parser.add_argument("--log_dir",
                        type=str,
                        help="The location to save the logs",
                        default="data/")

    parser.add_argument("--train_year",
                        type=int,
                        help="The year to train on",
                        default=1991)

    parser.add_argument("--train_month",
                        type=int,
                        help="The month to train on",
                        choices=range(1, 13),
                        default=1)

    parser.add_argument("--train_day",
                        type=int,
                        help="The day to train on",
                        choices=range(1, 32),
                        default=1)

    parser.add_argument("--num_train_days",
                        type=int,
                        help="The number of days to train on",
                        default=30)

    parser.add_argument("--model_save_freq",
                        type=int,
                        help="The numSteps/frequency to save the model",
                        default=2880)

    parser.add_argument("--normalize_advantage",
                        type=bool,
                        help="Whether to normalize the calculated advantage",
                        default=False)

    parser.add_argument("--num_episodes",
                        type=int,
                        help="The number of episodes to train on",
                        default=500)

    parser.add_argument("--batch_size",
                        type=int,
                        help="The size of each batch",
                        default=32)

    args = parser.parse_args()
    return args


def create_checkpoint_callback(save_freq, log_dir, prefix="b_denver"):
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=log_dir,
        name_prefix=prefix,
        save_replay_buffer=True,
    )
    return checkpoint_callback


def get_log_dirs(log_dir, run_name):
    current_dt = datetime.now()
    dt_str = current_dt.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, run_name, dt_str)
    model_save_dir = os.path.join(log_dir, "model")
    return log_dir, model_save_dir


def load_building_config(building_config_loc):
    with open(building_config_loc, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_env(building_env, building_config_loc,
            log_dir, energy_plus_loc, args,
            logger):
    config = load_building_config(building_config_loc)
    if building_env in ["denver", "sf"]:
        env = TypeABuilding(config, log_dir, energy_plus_loc, logger)
    elif building_env == "doee":
        env = DOOEBuilding(config, log_dir, energy_plus_loc, logger)

    env.set_runperiod(args.num_train_days, args.train_year, args.train_month, args.train_day)
    env.set_timestep(config["timesteps_per_hour"])
    return env, config


def get_logger(log_dir):
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    return logger


def get_callbacks(model_save_freq, log_dir):
    checkpoint_callback = create_checkpoint_callback(model_save_freq, log_dir)
    # total_energy_callback = TotalEnergyCallback()
    # callbacks = [checkpoint_callback, total_energy_callback]
    callbacks = [checkpoint_callback]
    return callbacks


def get_model(env, args, config):
    n_steps = config["timesteps_per_hour"] * 24 * args.num_train_days
    model = MultiAgentPPO(MultiAgentACPolicy, env, verbose=1,
                          n_steps=n_steps, batch_size=args.batch_size,
                          device=device, policy_kwargs={"control_zones": env.control_zones,
                                                        "device": device},
                          normalize_advantage=args.normalize_advantage)
    return model


def save_configs(log_dir, args, building_config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "run_config.yaml"), "w") as f:
        yaml.dump(args, f)
    with open(os.path.join(log_dir, "building_config.yaml"), "w") as f:
        yaml.dump(building_config, f)


def main():
    args = parse_args()

    log_dir, model_dir = get_log_dirs(args.log_dir, args.run_name)

    logger = get_logger(log_dir)
    env, config = get_env(args.building_env, args.building_config_loc,
                          log_dir, energy_plus_loc, args, logger)

    save_configs(log_dir, args, config)

    callbacks = get_callbacks(args.model_save_freq, model_dir)

    model = get_model(env, args, config)
    model.set_logger(logger)

    total_timesteps = args.num_episodes * config["timesteps_per_hour"] * 24 * args.num_train_days

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks,
                tb_log_name=args.run_name)
    model.save(os.path.join(model_dir, "final_model"))


if __name__ == "__main__":
    main()
