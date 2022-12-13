from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch as th


class MultiAgentDiagGaussianDistribution(DiagGaussianDistribution):
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
