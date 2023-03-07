from stable_baselines3.common.callbacks import BaseCallback
import wandb


class TotalEnergyCallback(BaseCallback):

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self):
        energy_consumption = self.model.env.envs[0].total_energy_consumption
        print("LOGGING ENERGY")
        self.logger.record("total_energy_consumption", energy_consumption)
        if wandb.run is not None:
            print("wandb logging energy")
            wandb.log({"total_energy_consumption": energy_consumption})
