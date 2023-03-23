from algorithms.diversity.diverse_base import BaseDiversity
from policies import PolicyTypeStrings
from policies.utils import load_policy_library

import torch as th
from typing import List


class PPODiversityHandler(BaseDiversity):
    def __init__(self,
                 diversity_weight: float = 0.0,
                 diverse_policies: List[str] = [],
                 diverse_policy_library_log_std_loc: str = "",
                 *args,
                 **kwargs):

        self.diversity_weight = diversity_weight
        self.diverse_policies = diverse_policies
        self.diverse_policy_paths = diverse_policies
        self.diverse_policy_library_log_std_loc = diverse_policy_library_log_std_loc

        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = "cpu"

        self.load_diverse_policies(self.device)

    def load_diverse_policies(self, device="cpu"):
        policies, policy_paths = load_policy_library(self.diverse_policies,
                                                     PolicyTypeStrings.single_agent_ac,
                                                     init_log_std_path=self.diverse_policy_library_log_std_loc,
                                                     device=device)
        self.diverse_policies = policies
        self.diverse_policy_paths = policy_paths

    def calculate_diversity_loss(self,
                                 obs: th.Tensor,
                                 action: th.Tensor,
                                 returns: th.Tensor,
                                 log_prob: th.Tensor):
        diversity_losses = th.zeros(len(obs), device=self.device)
        for i, zone in enumerate(list(obs.keys())):
            zn_obs = obs[zone]
            zn_actions = action[:, i]
            zn_log_prob = log_prob[:, i]
            zn_returns = returns[:, i]
            for other_policy in self.diverse_policies:
                with th.no_grad():
                    other_state_values, other_logprobs, _ = other_policy.evaluate_actions(zn_obs, zn_actions.reshape(-1, 1))
                other_state_values = th.squeeze(other_state_values)
                log_ratios = th.abs(other_logprobs.squeeze() - zn_log_prob)
                log_ratios = th.clamp(log_ratios, max=5)
                ratios = th.exp(log_ratios)
                # ratios = th.exp(th.abs(other_logprobs.squeeze() - zn_log_prob))
                ratios = th.max(ratios, 1 / ratios)
                # ratios = th.max(ratios, th.Tensor([100]))
                ratios = th.clamp(ratios, max=100)
                other_advantages = zn_returns - other_state_values.detach()
                other_advantages = th.clamp(other_advantages, min=0.1)
                diversity_loss = ratios / th.abs(other_advantages)
                diversity_losses[i] += diversity_loss.mean()
        if len(self.diverse_policies) > 1:
            diversity_losses /= len(self.diverse_policies)

        return diversity_losses
