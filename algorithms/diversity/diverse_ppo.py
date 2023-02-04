from algorithms.diversity.diverse_base import BaseDiversity
from policies.singleagentpolicy import SingleAgentACPolicy

import torch as th
from typing import List
import os
import yaml
import gym


class PPODiversityHandler(BaseDiversity):
    def __init__(self,
                 diversity_weight: float = 0.0,
                 diverse_policies: List[str] = [],
                 diverse_policies_init_log_std_loc: str = None,
                 obs_space: gym.spaces.Box = None,
                 action_space: gym.spaces.Box = None,
                 *args,
                 **kwargs):

        self.diversity_weight = diversity_weight
        self.diverse_policies = diverse_policies
        self.diverse_policy_paths = diverse_policies

        self.obs_space = obs_space
        self.action_space = action_space

        self.load_diverse_policies()

    def load_diverse_policies(self):
        assert os.path.isfile(self.diverse_policies_init_log_std_loc), \
            f"File {self.diverse_policies_init_log_std_loc} does not exist"

        with open(self.diverse_policies_init_log_std_loc, "r") as f:
            policy_loc_log_std = yaml.load(f, Loader=yaml.FullLoader)

        diverse_policies = []
        for policy in self.diverse_policy_paths:
            policy_name = os.path.basename(policy)
            policy_obj = SingleAgentPolicy(observation_space=self.obs_space,
                                           action_space=self.action_space,
                                           log_std_init=policy_loc_log_std[policy_name])
            policy_obj.load_state_dict(th.load(policy))
            diverse_policies.append(policy_obj)

        self.diverse_policies = diverse_policies

    def calculate_diversity_loss(self,
                                 obs: th.Tensor,
                                 action: th.Tensor,
                                 returns: th.Tensor,
                                 log_prob: th.Tensor):
        for other_policy in self.diverse_policies:
            other_logprobs, other_state_values = other_policy.evaluate_actions(obs, action)
            other_state_values = th.squeeze(other_state_values)
            ratios = th.exp(th.abs(other_logprobs - log_prob))
            ratios = th.max(ratios, 1.0 / ratios)
            ratios = th.max(ratios, 100)
            other_advantages = returns - other_state_values.detach()

            diversity_loss = ratios / th.abs(other_advantages)

        if len(self.diverse_policies) > 0:
            diversity_loss /= len(self.diverse_policies)

        return diversity_loss
