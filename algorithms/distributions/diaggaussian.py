from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.distributions import Normal
import torch as th


class MultiAgentDiagGaussianDistribution(DiagGaussianDistribution):
    """Re-implementation of DiagGaussianDistribution to support multi-dim action space

    """
    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return log_prob

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()
