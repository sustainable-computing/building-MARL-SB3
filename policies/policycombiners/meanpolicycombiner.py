from policies.policycombiners.policycombinersbase import PolicyCombinersBase
from policies.singleagentpolicy import SingleAgentACPolicy

from torch.distributions import Normal


import torch as th
from typing import List, Tuple


class MeanPolicyCombiner(PolicyCombinersBase):
    def __init__(self,
                 policies: List[SingleAgentACPolicy],
                 device=th.device("cpu")):
        super().__init__(policies, device)

    def __call__(self, features: th.Tensor) -> Tuple:
        return self.forward(features)

    def forward(self, features: th.Tensor) -> Tuple:
        if len(features.shape) == 1:
            batch_size = 1
        else:
            batch_size = features.shape[0]
        actions = th.zeros((batch_size, len(self.policies)), device=self.device)
        values = th.zeros((batch_size, len(self.policies)), device=self.device)
        log_probs = th.zeros((batch_size, len(self.policies)), device=self.device)

        for i, policy in enumerate(self.policies):
            actions[:, i], values[:, i], log_probs[:, i] = policy.forward(features)

        action = actions.mean()
        value = values.mean()
        log_prob = log_probs.mean()

        return action, value, log_prob

    def get_distribution(self, features: th.Tensor) -> th.distributions.Distribution:
        if len(features.shape) == 1:
            batch_size = 1
        else:
            batch_size = features.shape[0]
        means = th.zeros((batch_size, len(self.policies)), device=self.device)
        stds = th.zeros((batch_size, len(self.policies)), device=self.device)

        for i, policy in enumerate(self.policies):
            distribution = policy.get_distribution(features).distribution
            means[:, i] = distribution.loc.reshape(1, -1)
            stds[:, i] = distribution.scale.reshape(1, -1)

        mean = means.mean(dim=1)
        std = stds.mean(dim=1)

        # Taking a shortcut, initializing a dummy class to not raise error in IPW
        distribution = Normal(mean.reshape(-1, 1), std.reshape(-1, 1))

        class Dummy:
            def __init__(self, distribution):
                self.distribution = distribution

        return Dummy(distribution)
