from policies.singleagentpolicy import SingleAgentACPolicy

import torch as th
from typing import List


class PolicyCombinersBase:
    def __init__(self,
                 policies: List[SingleAgentACPolicy],
                 device=th.device("cpu")):
        self.policies = policies
        self.device = device

    def __call__(self, features: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def forward(self, features: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def get_distribution(self, features: th.Tensor) -> th.distributions.Distribution:
        raise NotImplementedError
