import os
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config(config, config_path):
    base_path = os.path.dirname(config_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
