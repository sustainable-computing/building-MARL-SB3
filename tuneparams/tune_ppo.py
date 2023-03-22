from buildingenvs import BuildingEnvStrings
from train.ppo import train_ppo
from utils.configs import load_config, save_config
from utils.logs import create_log_dir

import pandas as pd
import os
import typer
import wandb


app = typer.Typer()


@app.command("ppo-optimal")
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
    run_name: str = typer.Option("ppo_run_0"),
    sweep_id: str = typer.Option(""),
    energy_plus_loc: str = typer.Option(...),
    num_days: int = typer.Option(...),
    start_day: int = typer.Option(1),
    start_month: int = typer.Option(1),
    start_year: int = typer.Option(1991),
    log_dir: str = typer.Option("data/hyperparam_sweeps/"),
    wandb_project_name: str = typer.Option(""),
    seed: int = typer.Option(1337),
    device: str = typer.Option("cpu"),
):

    sweep_config = load_config(sweep_config_loc)
    save_config(sweep_config, os.path.join(log_dir, "sweep_config.yaml"))
    log_dir = create_log_dir(log_dir, run_name, use_dt_str=False)
    default_args = locals()

    wandb.agent(sweep_id, construct_train_function(default_args),
                count=1, project=wandb_project_name)


def construct_train_function(default_args):
    def train_function():
        wandb.init()
        config = {**default_args, **wandb.config}
        wandb.alert(title="Sweep Updates",
                    text=f"Agent training started for sweep {config['sweep_id']}")
        config["use_wandb"] = False
        log_dir, _ = train_ppo.train(**config)
        progress_csv = os.path.join(log_dir, "progress.csv")
        progress_df = pd.read_csv(progress_csv)
        final_energy_consumption = progress_df["total_energy_consumption"].iloc[-1]
        wandb.log({"final_energy_consumption": final_energy_consumption})
    return train_function
