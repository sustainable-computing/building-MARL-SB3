from gym import spaces
import numpy as np

from buildingenvs import Building
from cobs import Model


class DOOEBuilding(Building):
    def __init__(self,
                 config: dict,
                 log_dir: str,
                 energy_plus_dir: str,
                 logger: object = None):
        super().__init__(config=config, log_dir=log_dir)

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
                                          identifier={"Name": f"VAV HW Rht {zone}"},
                                          update_values={
                                                "Zone Minimum Air Flow Input Method": "Scheduled",
                                                "Constant Minimum Air Flow Fraction": "",
                                                "Minimum Air Flow Fraction Schedule Name":
                                                f"{zone} VAV Customized Schedule"
                                            })

    def get_additional_states(self) -> dict:
        additional_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity"
                             for zone in self.available_zones}
        additional_states.update({
            ("Zone Air System Sensible Heating Rate", f"{zone}"):
            f"{zone} vav heating energy" for zone in self.available_zones
            })
        additional_states.update({
            ("Zone Air System Sensible Cooling Rate", f"{zone}"):
            f"{zone} vav cooling energy" for zone in self.available_zones
            })
        additional_states.update({
            ("Zone Air Terminal VAV Damper Position", f"VAV HW Rht {zone}"):
            f"{zone} position" for zone in self.available_zones
            })
        additional_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = \
            "outdoor temperature"
        additional_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = \
            "site solar radiation"
        additional_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = \
            "total hvac"
        additional_states[('Schedule Value', 'HVACOperationSchd')] = "operations availability"

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
                "value": action,
                "start_time": self.current_obs_timestep + 1
                })

        state = self.model.step(action_list)
        zonewise_state = self.get_state_dict(state)
        self.total_energy_consumption += state["total hvac"]
        self.current_obs_timestep = state["timestep"]
        vav_energies = {zone: state[f"{zone} vav heating energy"] +
                                  state[f"{zone} vav cooling energy"]
                        for zone in self.control_zones}
        rewards = np.array([-vav_energies[zone] for zone in self.control_zones])
        done = self.model.is_terminate()
        info = {}
        return zonewise_state, rewards, done, info

    def reset(self):
        if self.logger:
            self.logger.record("total_energy_consumption", self.total_energy_consumption)
        state = self.model.reset()
        zonewise_state = self.get_state_dict(state)
        self.total_energy_consumption = state["total hvac"]
        self.current_obs_timestep = state["timestep"]
        return zonewise_state
