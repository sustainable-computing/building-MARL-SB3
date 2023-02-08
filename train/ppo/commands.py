import typer
from buildingenvs import BuildingEnvStrings


app = typer.Typer()
state = {}


@app.callback()
def _main(building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
          building_config_loc: str = typer.Option(
                "configs/buildingconfig/building_denver.yaml",
                help="The location of the building config file"),
          run_name: str = typer.Option("denvertest", help="The name of the run"),
          log_dir: str = typer.Option("data/trainlogs/", help="The location to save the logs"),
          train_year: int = typer.Option(1991, help="The year to train on"),
          train_month: int = typer.Option(1, help="The month to train on"),
          train_day: int = typer.Option(1, help="The day to train on"),
          num_train_days: int = typer.Option(
              30, help="The number of days to train on"),
          model_save_freq: int = typer.Option(
              2880, help="The frequency to save the model"),
          normalize_advantage: bool =
          typer.Option(False, help="Whether to normalize the calculated advantage"),
          ent_coef: float = typer.Option(0.01, help="The entropy coefficient"),
          num_episodes: int = typer.Option(500, help="The number of episodes to train for"),
          batch_size: int = typer.Option(32, help="The batch size"),
          seed: int = typer.Option(1337, help="The seed for the environment"),
          device: str = typer.Option("cpu", help="The device to train on"),
          energy_plus_loc: str = typer.Option("/Applications/EnergyPlus-9-3-0/",
                                              help="The location of the energyplus executable")):
    state["building_env"] = building_env
    state["building_config_loc"] = building_config_loc
    state["run_name"] = run_name
    state["log_dir"] = log_dir
    state["train_year"] = train_year
    state["train_month"] = train_month
    state["train_day"] = train_day
    state["num_train_days"] = num_train_days
    state["model_save_freq"] = model_save_freq
    state["normalize_advantage"] = normalize_advantage
    state["ent_coef"] = ent_coef
    state["num_episodes"] = num_episodes
    state["batch_size"] = batch_size
    state["seed"] = seed
    state["device"] = device
    state["energy_plus_loc"] = energy_plus_loc


@app.command("optimal")
def train_optimal():
    from train.ppo.train_ppo import train
    state["diverse_training"] = False
    train(**state)


@app.command("diverse")
def train_diverse(diverse_policy_library_loc: str = typer.Option(
                    "data/policies/", help="The location of the policy library"),
                  diversity_weight: float = typer.Option(0.01, help="The weight of diversity")):
    from train.ppo.train_ppo import train
    state["diverse_policy_library_loc"] = diverse_policy_library_loc
    state["diversity_weight"] = diversity_weight
    state["diverse_training"] = True
    train(**state)
