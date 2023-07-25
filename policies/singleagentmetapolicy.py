from policies.singleagentpolicy import SingleAgentACPolicy
from policies.policycombiners.meanpolicycombiner import MeanPolicyCombiner
from policies.policycombiners.statevaluecombiner import MaxStateValueCombiner
from policies.policycombiners.ucbpolicyselector import UCBPolicyCombiner

import torch as th
from typing import List, Tuple


class SingleAgentMetaPolicy:
    def __init__(self,
                 policies: List[SingleAgentACPolicy],
                 combining_method: str = "mean",
                 device: th.device = th.device("cpu"),
                 **kwargs):
        self.policies = policies
        self.combining_method = combining_method
        self.device = device

        if self.combining_method == "mean":
            self.policy_combiner = MeanPolicyCombiner(self.policies, self.device)
        elif self.combining_method == "maxstatevalue":
            self.policy_combiner = MaxStateValueCombiner(self.policies, self.device)
        elif self.combining_method == "ucb":
            self.policy_combiner = UCBPolicyCombiner(self.policies, self.device, **kwargs)
        else:
            raise NotImplementedError(f"Combining method {self.combining_method} not implemented")

    def __call__(self, features: th.Tensor) -> Tuple:
        return self.forward(features)

    def forward(self, features: th.Tensor) -> Tuple:
        return self.policy_combiner.forward(features)

    def get_distribution(self, features: th.Tensor) -> th.distributions.Distribution:
        return self.policy_combiner.get_distribution(features)
