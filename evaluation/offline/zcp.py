from evaluation.offline.iw import InverseProbabilityWeighting
from evaluation.offline.opebase import OPEBase

import pandas as pd
import numpy as np
from obp.ope import (
    ContinuousOffPolicyEvaluation,
    KernelizedInverseProbabilityWeighting
)
import torch
import torch.nn.functional as F
import types
import torch.nn as nn


class SNIP(OPEBase):
    def __init__(self, log_data: pd.DataFrame,
                 eps_clip: float = 0.2,
                 gamma: float = 1.,
                 rule_based_behavior_policy: bool = True,
                 **kwargs):

        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = "cpu"
        self.log_data = log_data
        self.rule_based_behavior_policy = rule_based_behavior_policy
        self.ipw_obj = \
            InverseProbabilityWeighting(log_data,
                                        rule_based_behavior_policy=rule_based_behavior_policy,
                                        no_grad=False)
        self.eps_clip = eps_clip
        self.gamma = gamma

    def snip_forward_linear(self, layer, x):
        return F.linear(x, layer.weight * layer.weight_mask, layer.bias)

    def snip(self, layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    def calculate_loss(self, use_behavior_policy=True, return_additional=True, **kwargs):
        if self.ipw_obj.rule_based_behavior_policy:
            scaled_rewards, states, rewards, policy_action_prob, behavior_action_prob = \
                self.ipw_obj.optimized_evaluate_policy(self.policy.get_distribution, self.behavior_policy, score="",
                                                       device=self.device,
                                                       reward_signal=kwargs["reward_signal"])
        else:
            scaled_rewards, states, rewards, policy_action_prob, behavior_action_prob = \
                self.ipw_obj.optimized_evaluate_policy(self.policy.get_distribution, self.behavior_policy, score="",
                                                       device=self.device,
                                                       reward_signal=kwargs["reward_signal"])

        # scaled_rewards = -scaled_rewards
        discounted_rewards = []
        disc_reward = 0
        for reward in scaled_rewards.reshape(-1).tolist()[::-1]:
            disc_reward = reward + (self.gamma * disc_reward)
            discounted_rewards.insert(0, disc_reward)
        if isinstance(states, list):
            states = torch.stack(states).to(self.device)
        state_values = self.policy.mlp_extractor.critic_network(states).squeeze()
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        ratios = policy_action_prob / behavior_action_prob

        advantages = discounted_rewards - state_values
        surr_1 = ratios * advantages.reshape(-1, 1)
        surr_2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages.reshape(-1, 1)

        loss = -torch.min(surr_1, surr_2)
        if not return_additional:
            return loss
        else:
            return loss, scaled_rewards, states, rewards, policy_action_prob, behavior_action_prob

    def evaluate_policy(self, evaluation_policy: callable = None,
                        behavior_policy: dict = None,
                        return_additional=False, **kwargs):
        self.policy = evaluation_policy
        self.behavior_policy = behavior_policy
        for layer in self.policy.mlp_extractor.actor_network.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False
            # Override the forward methods:
            # if isinstance(layer, nn.Conv2d):
            #     layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(self.snip_forward_linear, layer)

        self.policy.mlp_extractor.zero_grad()
        if not return_additional:
            loss = self.calculate_loss(self.log_data, return_additional=return_additional,
                                       reward_signal=kwargs["reward_signal"])
        else:
            loss, scaled_rewards, states, rewards,\
                policy_action_prob, behavior_action_prob = \
                    self.calculate_loss(self.log_data, return_additional=return_additional,
                                        reward_signal=kwargs["reward_signal"])
        loss.mean().backward()

        metric_array = []
        for layer in self.policy.mlp_extractor.actor_network.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(self.snip(layer))

        sum = 0.
        for i in range(len(metric_array)):
            sum += torch.sum(metric_array[i])
        if not return_additional:
            return sum.item()
        else:
            return sum.item(), loss.cpu(), \
                scaled_rewards.cpu(), \
                states.cpu(), rewards.cpu(), \
                policy_action_prob.cpu(), \
                behavior_action_prob.cpu()


class GaussianKernel(OPEBase):
    def __init__(self, log_data: pd.DataFrame = None,
                 bandwidth: float = 0.3):
        self.log_data = log_data
        self.kernel = "gaussian"
        self.bandwidth = bandwidth

        self.process_log_data()
        self.initialize_gk()

    def initialize_gk(self):
        bandit_feedback = {
            "action": self.actions,
            "reward": self.rewards,
            "pscore": np.ones((len(self.log_data)))
        }
        ope_estimator = KernelizedInverseProbabilityWeighting(kernel=self.kernel,
                                                              bandwidth=self.bandwidth)
        ope = ContinuousOffPolicyEvaluation(bandit_feedback=bandit_feedback,
                                            ope_estimators=[ope_estimator])
        self.ope_estimator = ope

    def process_log_data(self):
        states = []
        actions = []
        rewards = []
        state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                      "zone_humidity", "zone_temp", "zone_occupancy"]
        for i, row in self.log_data.iterrows():
            state = [row[var] for var in state_vars]
            actions.append(row['action'])
            rewards.append(row['reward'])
            states.append(state)
        self.states = np.array(states, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.float32)
        self.rewards = np.array(rewards, dtype=np.float32)

    def evaluate_policy(self, evaluation_policy: callable = None, **kwargs):
        states = torch.Tensor(self.states)
        with torch.no_grad():
            actions, _, _ = evaluation_policy(states)
        estimated_value = self.ope_estimator.estimate_policy_values(action_by_evaluation_policy=actions.squeeze().numpy())
        return estimated_value["kernelized_ipw"]
