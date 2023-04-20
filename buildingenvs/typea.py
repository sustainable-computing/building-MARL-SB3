from gym import spaces
import numpy as np
import wandb

from buildingenvs import Building
from cobs import Model


class TypeABuilding(Building):
    def __init__(self,
                 config: dict,
                 log_dir: str,
                 energy_plus_dir: str,
                 logger: object = None):
        super().__init__(config, log_dir)

        Model.set_energyplus_folder(energy_plus_dir)

        self.logger = logger

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_zones,))

        self.observation_space = spaces.Dict({
                        zone: spaces.Box(low=np.array([-np.inf]*len(self.observations)),
                                         high=np.array([np.inf]*len(self.observations)))
                        for zone in self.control_zones
                       })

        self.init_model()

        self.total_energy_consumption = 0

    def init_model(self):
        additional_states = self.get_additional_states()
        self.model = Model(idf_file_name=self.idf_file_name,
                           weather_file=self.weather_loc,
                           eplus_naming_dict=additional_states,
                           tmp_idf_path=self.log_dir)

        for key in additional_states:
            self.model.add_configuration("Output:Variable",
                                         {
                                            "Key Value": key[1],
                                            "Variable Name": key[0],
                                            "Reporting Frequency": "Timestep"
                                         })

        for zone in self.control_zones:
            self.model.add_configuration("Schedule:Constant",
                                         {
                                            "Name": f"{zone} VAV Customized Schedule",
                                            "Schedule Type Limits Name": "Fraction",
                                            "Hourly Value": 0
                                         })
            self.model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                          identifier={
                                            "Name": f"{zone} VAV Box Component"
                                            },
                                          update_values={
                                            "Zone Minimum Air Flow Input Method":
                                            "Scheduled",
                                            "Minimum Air Flow Fraction Schedule Name":
                                            f"{zone} VAV Customized Schedule"
                                          })

    def get_additional_states(self) -> dict:
        additional_states = {
            ("Zone Air Relative Humidity", zone):
                f"{zone} humidity" for zone in self.available_zones
            }
        additional_states.update({
            ("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"):
                f"{zone} vav energy" for zone in self.available_zones
            })
        additional_states.update({
            ("Air System Electric Energy", airloop): f"{airloop} energy"
            for airloop in set(self.airloops.values())
            })
        additional_states.update({
            ("Zone Air Terminal Minimum Air Flow Fraction", f"{zone} VAV Box Component"):
                f"{zone} position" for zone in self.available_zones
        })
        additional_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = \
            "outdoor temperature"

        solar_radiation_key = \
            ('Site Direct Solar Radiation Rate per Area', 'Environment')
        additional_states[solar_radiation_key] = "site solar radiation"

        total_hvac_key = ('Facility Total HVAC Electric Demand Power', 'Whole Building')
        additional_states[total_hvac_key] = "total hvac"

        return additional_states

    def get_state_dict(self, state):
        zonewise_state = {}
        for zone in self.control_zones:
            occupancy = 1 if state["occupancy"][zone] > 0 else 0
            zone_state = [
                state["outdoor temperature"],
                state["site solar radiation"],
                state["time"].hour,
                state[f"{zone} humidity"],
                state["temperature"][zone],
                occupancy
            ]
            zonewise_state[zone] = zone_state
        return zonewise_state

    def step(self, actions):
        action_list = []
        for zone, action in zip(self.control_zones, actions):
            action_list.append({
                "priority": 0,
                "component_type": "Schedule:Constant",
                "control_type": "Schedule Value",
                "actuator_key": f"{zone} VAV Customized Schedule",
                "value": action.item(),
                "start_time": self.current_obs_timestep + 1
                })

        state = self.model.step(action_list)
        zonewise_state = self.get_state_dict(state)
        self.total_energy_consumption += state["total hvac"]
        self.current_obs_timestep = state["timestep"]
        rewards = np.array([-state[f"{zone} vav energy"] for zone in self.control_zones])
        done = self.model.is_terminate()
        info = {
            "cobs_state": state
        }
        return zonewise_state, rewards, done, info

    def reset(self):
        if self.logger:
            self.logger.record("total_energy_consumption", self.total_energy_consumption)
            if wandb.run is not None:
                wandb.log({"total_energy_consumption": self.total_energy_consumption})
        state = self.model.reset()
        zonewise_state = self.get_state_dict(state)
        self.total_energy_consumption = state["total hvac"]
        self.current_obs_timestep = state["timestep"]
        return zonewise_state
