import numpy as np
import pickle as pkl
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class StateVisitCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(StateVisitCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.state_visits = {}

    def _on_step(self) -> bool:
        observation = self.locals["obs_tensor"]
        for zone in observation:
            zone_obs = observation[zone]
            zone_obs = zone_obs.squeeze().cpu().numpy()
            if zone not in self.state_visits:
                self.state_visits[zone] = np.array([zone_obs])
            else:
                self.state_visits[zone] = np.vstack((self.state_visits[zone], zone_obs))
        return True

    def _on_training_end(self) -> None:
        with open(f"{self.log_dir}/state_visits.pkl", "wb") as f:
            pkl.dump(self.state_visits, f)

        if wandb.run is not None:
            wandb.save(f"{self.log_dir}/state_visits.pkl")
