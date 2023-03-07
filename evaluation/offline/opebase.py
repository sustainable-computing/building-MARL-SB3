from abc import ABC, abstractmethod

from stable_baselines3.common.policies import BasePolicy


class OPEBase(ABC):
    @abstractmethod
    def evaluate_policy(self,
                        evaluation_policy: BasePolicy = None,
                        evaluation_policy_distribution_fuc: callable = None,
                        behavior_policy: dict = None,
                        score: str = "mean",
                        **kwargs
                        ):
        """To override in child class"""
        pass
