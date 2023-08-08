import typer
from buildingenvs import BuildingEnvStrings


app = typer.Typer()


@app.command("optimal")
def train_optimal(
    building_env: BuildingEnvStrings = typer.Option(
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
    num_train_days: int = typer.Option(30, help="The number of days to train on"),
    model_save_freq: int = typer.Option(2880, help="The frequency to save the model"),
    normalize_advantage: bool = typer.Option(False,
                                             help="Whether to normalize the calculated advantage"),
    ent_coef: float = typer.Option(0.01, help="The entropy coefficient"),
    learning_rate: float = typer.Option(0.0003, help="The learning rate"),
    num_episodes: int = typer.Option(500, help="The number of episodes to train for"),
    batch_size: int = typer.Option(32, help="The batch size"),
    seed: int = typer.Option(1337, help="The seed for the environment"),
    device: str = typer.Option("cpu", help="The device to train on"),
    energy_plus_loc: str = typer.Option("/Applications/EnergyPlus-9-3-0/",
                                        help="The location of the energyplus executable"),
    use_wandb: bool = typer.Option(False, help="Whether to use wandb for logging metics"),
    wandb_project_name: str = typer.Option("ppo-train", help="The name of the wandb project"),
    wandb_run_name: str = typer.Option("", help="The name of the wandb run"),
    torch_compile: bool = typer.Option(False, help="Whether to use  torch.compile")
):

    from train.ppo.train_ppo import train
    kwargs = locals()
    kwargs["diverse_training"] = False
    kwargs["retrain"] = False
    train(**kwargs)


@app.command("diverse")
def train_diverse(
    building_env: BuildingEnvStrings = typer.Option(
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
    normalize_advantage: bool = typer.Option(False,
                                             help="Whether to normalize the calculated advantage"),
    ent_coef: float = typer.Option(0.01, help="The entropy coefficient"),
    learning_rate: float = typer.Option(0.0003, help="The learning rate"),
    num_episodes: int = typer.Option(500, help="The number of episodes to train for"),
    batch_size: int = typer.Option(32, help="The batch size"),
    seed: int = typer.Option(1337, help="The seed for the environment"),
    device: str = typer.Option("cpu", help="The device to train on"),
    energy_plus_loc: str = typer.Option("/Applications/EnergyPlus-9-3-0/",
                                        help="The location of the energyplus executable"),
    use_wandb: bool = typer.Option(False, help="Whether to use wandb for logging metics"),
    wandb_project_name: str = typer.Option("ppo-train",
                                           help="The name of the wandb project"),
    wandb_run_name: str = typer.Option("",
                                       help="The name of the wandb run"),
    torch_compile: bool = typer.Option(False, help="Whether to use  torch.compile"),
    diverse_policy_library_loc: str = typer.Option(
        "data/policies/", help="The location of the policy library"),
    diversity_weight: float = typer.Option(0.01, help="The weight of diversity"),
    diverse_policy_library_log_std_loc: str =
        typer.Option("", help="The location of the policy library log stds")
):

    from train.ppo.train_ppo import train
    kwargs = locals()
    kwargs["diverse_policy_library_loc"] = diverse_policy_library_loc
    kwargs["diversity_weight"] = diversity_weight
    kwargs["diverse_training"] = True
    kwargs["diverse_policy_library_log_std_loc"] = diverse_policy_library_log_std_loc
    train(**kwargs)


@app.command("retrain")
def retrain(
    building_env: BuildingEnvStrings = typer.Option(
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
    policy_map_config_loc: str = typer.Option("data/policy_map.json",
                                              help="The location of the policy map"),
    model_save_freq: int = typer.Option(2880, help="The frequency to save the model"),
    normalize_advantage: bool = typer.Option(False,
                                             help="Whether to normalize the calculated advantage"),
    ent_coef: float = typer.Option(0.01, help="The entropy coefficient"),
    learning_rate: float = typer.Option(0.0003, help="The learning rate"),
    num_episodes: int = typer.Option(500, help="The number of episodes to train for"),
    batch_size: int = typer.Option(32, help="The batch size"),
    seed: int = typer.Option(1337, help="The seed for the environment"),
    device: str = typer.Option("cpu", help="The device to train on"),
    energy_plus_loc: str = typer.Option("/Applications/EnergyPlus-9-3-0/",
                                        help="The location of the energyplus executable"),
    use_wandb: bool = typer.Option(False, help="Whether to use wandb for logging metics"),
    wandb_project_name: str = typer.Option("ppo-train",
                                           help="The name of the wandb project"),
    wandb_run_name: str = typer.Option("",
                                       help="The name of the wandb run"),
    torch_compile: bool = typer.Option(False, help="Whether to use  torch.compile"),
):

    from train.ppo.train_ppo import retrain_policies
    kwargs = locals()
    kwargs["diverse_training"] = False
    kwargs["retrain"] = True
    kwargs["policy_map_config_loc"] = policy_map_config_loc
    retrain_policies(**kwargs)
