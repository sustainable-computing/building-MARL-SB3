import typer
import numpy as np

from buildingenvs import BuildingEnvStrings
from evaluation.online.evaluate import evaluate_policies, evaluate_rule_based
from evaluation.online.evaluate import run_full_simulation, run_full_automated_swapping_simulation


app = typer.Typer()


@app.command("evaluate-policies")
def evaluate(
    building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
    building_config_loc: str = typer.Option(
        "configs/buildingconfig/building_denver.yaml",
        help="The location of the building config file"),
    zone: str = typer.Option(None),
    energy_plus_loc: str = typer.Option(...),
    policy_library_path: str = typer.Option(...),
    policy_type: str = typer.Option(...),
    init_policy_log_std: float = typer.Option(np.log(0.1)),
    init_policy_log_std_path: str = typer.Option(""),
    num_days: int = typer.Option(...),
    start_day: int = typer.Option(1),
    start_month: int = typer.Option(1),
    start_year: int = typer.Option(1991),
    save_path: str = typer.Option("data/policy_evaluation/brute_force/"),
    parallelize: bool = typer.Option(False),
    max_cpu_cores: int = typer.Option(1),
    seed: int = typer.Option(1337),
    device: str = typer.Option("cpu")
):
    if hasattr(policy_type, "value"):
        policy_type = policy_type.value
    evaluate_policies(**locals())


@app.command("evaluate-rule-based")
def evaluate_rbc(
    building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
    building_config_loc: str = typer.Option(
        "configs/buildingconfig/building_denver.yaml",
        help="The location of the building config file"),
    zone: str = typer.Option(None),
    policy_default_action: float = typer.Option(0.3),
    energy_plus_loc: str = typer.Option(...),
    num_days: int = typer.Option(...),
    start_day: int = typer.Option(1),
    start_month: int = typer.Option(1),
    start_year: int = typer.Option(1991),
    save_path: str = typer.Option("data/policy_evaluation/rule_based/"),
    parallelize: bool = typer.Option(False),
    max_cpu_cores: int = typer.Option(1),
    seed: int = typer.Option(1337)
):
    evaluate_rule_based(**locals())


@app.command("run-full-simulation")
def run_full_sim(
    building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
    building_config_loc: str = typer.Option(
        "configs/buildingconfig/building_denver.yaml",
        help="The location of the building config file"),
    policy_map_config_loc: str = "configs/policymapconfigs/denver/gt_best_one_year_denver.yaml",
    energy_plus_loc: str = typer.Option(...),
    save_path: str = typer.Option("data/policy_map_full_sim/"),
    seed: int = typer.Option(1337),
    device: str = typer.Option("cpu")
):
    run_full_simulation(**locals())


@app.command("run-full-automated-swapping-simulation")
def run_full_sim(
    building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
    building_config_loc: str = typer.Option(
        "configs/buildingconfig/building_denver.yaml",
        help="The location of the building config file"),
    policy_library_path: str = typer.Option(...),
    policy_type: str = typer.Option(...),
    init_policy_log_std: float = typer.Option(np.log(0.1)),
    init_policy_log_std_path: str = typer.Option(""),
    policy_map_config_loc: str = "configs/policymapconfigs/denver/gt_best_one_year_denver.yaml",
    energy_plus_loc: str = typer.Option(...),
    save_path: str = typer.Option("data/policy_map_full_sim/"),
    seed: int = typer.Option(1337),
    device: str = typer.Option("cpu")
):
    run_full_automated_swapping_simulation(**locals())
