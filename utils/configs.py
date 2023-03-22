import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.dump(config, f)
