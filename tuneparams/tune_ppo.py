from buildingenvs import BuildingEnvStrings
from train.ppo import train_ppo
from utils.configs import load_config, save_config
from utils.logs import create_log_dir

import pandas as pd
import os
import typer
import wandb


app = typer.Typer()


@app.command("tune-ppo-optimal")
def tune_ppo_optimal_command(
    building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
    building_config_loc: str = typer.Option(
        "configs/buildingconfig/building_denver.yaml",
        help="The location of the building config file"),
    sweep_config_loc: str = typer.Option(
        "configs/sweepconfigs/ppo_sweep_0.yaml",
        help="The location of the sweep config file"),
    sweep_name: str = typer.Option("ppo_sweep_0"),
    energy_plus_loc: str = typer.Option(...),
    num_days: int = typer.Option(...),
    start_day: int = typer.Option(1),
    start_month: int = typer.Option(1),
    start_year: int = typer.Option(1991),
    log_dir: str = typer.Option("data/hyperparam_sweeps/"),
    wandb_project_name: str = typer.Option(""),
    seed: int = typer.Option(1337)
):

    sweep_config = load_config(sweep_config_loc)
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    log_dir = create_log_dir(log_dir, sweep_name)
    save_config(sweep_config, os.path.join(log_dir, "sweep_config.yaml"))
    default_args = locals()

    wandb.agent(sweep_id, construct_train_function(default_args), count=1, project=wandb_project_name)


def construct_train_function(default_args):
    def train_function():
        wandb.init()
        config = {**default_args, **wandb.config}
        config["use_wandb"] = False
        log_dir, _ = train_ppo.train(**config)
        progress_csv = os.path.join(log_dir, "progress.csv")
        progress_df = pd.read_csv(progress_csv)
        final_energy_consumption = progress_df["total_energy_consumption"].iloc[-1]
        wandb.log({"final_energy_consumption": final_energy_consumption})
    return train_function
