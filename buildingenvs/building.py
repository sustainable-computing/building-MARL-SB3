from abc import abstractmethod
import gym
import torch as th


class Building(gym.Env):
    def __init__(self, config: dict, log_dir: str, **kwargs):
        super(Building, self).__init__(**kwargs)

        self.name = config["building_name"]
        self.num_zones = config["num_zones"]

        self.observations = config["observations"]

        self.weather_loc = config["weather_file"]
        self.idf_file_name = config["building_idf_file"]

        self.available_zones = config["available_zones"]

        if "control_zones" in config:
            self.control_zones = config["control_zones"]
        else:
            self.control_zones = self.available_zones

        if "airloops" in config:
            self.airloops = config["airloops"]

        self.log_dir = log_dir

    def set_runperiod(self, days: int, start_year: int, start_month: int,
                      start_day: int, specify_year: bool = False):
        self.model.set_runperiod(days=days, start_year=start_year, start_month=start_month,
                                 start_day=start_day, specify_year=specify_year)

    def set_timestep(self, timestep_per_hour: int):
        self.model.set_timestep(timestep_per_hour=timestep_per_hour)

    @abstractmethod
    def init_model(self):
        """To be implemented by subclasses"""
        pass

    @abstractmethod
    def get_additional_states(self) -> dict:
        """To be implemented by subclasses"""
        pass

    @abstractmethod
    def reset(self):
        """To be implemented by subclasses"""
        pass

    def is_terminate(self) -> bool:
        return self.model.is_terminate()

    @abstractmethod
    def step(self, actions: dict) -> dict:
        """To be implemented by subclasses"""
        pass
