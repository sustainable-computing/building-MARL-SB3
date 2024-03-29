from algorithms.distributions.diaggaussian import MultiAgentDiagGaussianDistribution

import gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule
from typing import Optional, List, Union, Dict, Type, Tuple
import torch as th
from torch import nn


class MultiAgentNetworkV1(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        control_zones: List[str],
        device: th.device = th.device("cpu"),
    ):
        super().__init__()
        self.actor_networks = nn.ModuleDict()
        self.critic_networks = nn.ModuleDict()
        self.control_zones = control_zones
        self.train_device = device

        for zone in control_zones:
            self.actor_networks[zone] = self.base_network(feature_dim)
            self.critic_networks[zone] = self.base_network(feature_dim)

        self.latent_dim_pi = len(control_zones)
        self.latent_dim_vf = len(control_zones)

    @classmethod
    def base_network(cls, feature_dim) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, features: th.DictType) -> th.Tensor:
        actions = th.zeros(size=(len(features[self.control_zones[0]]), 1, len(self.control_zones),),
                           device=self.train_device)
        values = th.zeros(size=(len(features[self.control_zones[0]]), 1, len(self.control_zones),),
                          device=self.train_device)
        for i, zone in enumerate(self.control_zones):
            actions[:, :, i] = self.actor_networks[zone](features[zone])
            values[:, :, i] = self.critic_networks[zone](features[zone])

        actions = actions.squeeze(1)
        values = values.squeeze(1)
        return actions, values

    def forward_actor(self, features: th.DictType) -> th.Tensor:
        actions = {}
        for zone in self.control_zones:
            actions[zone] = self.actor_networks[zone](features[zone])
        actions = actions.squeeze(1)
        return actions

    def forward_critics(self, features: th.DictType) -> th.Tensor:
        values = {}
        for zone in self.control_zones:
            values[zone] = self.critic_networks[zone](features[zone])
        values = values.squeeze(1)
        return values


class MultiAgentACPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 action_space: gym.spaces.Dict,
                 lr_schedule: Schedule,
                 control_zones: List[str],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 device: th.device = th.device("cpu"),
                 *args,
                 **kwargs):
        self.control_zones = control_zones
        self.train_device = device

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            control_zones,
            *args,
            **kwargs
        )
        self.features_dim = observation_space[self.control_zones[0]].shape[0]

        self._rebuild(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.features_dim = self.observation_space[self.control_zones[0]].shape[0]
        self.mlp_extractor = MultiAgentNetworkV1(
            feature_dim=self.features_dim,
            control_zones=self.control_zones,
            device=self.train_device
        )
        self.mlp_extractor.to(self.train_device)

    def _rebuild(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = \
            MultiAgentDiagGaussianDistribution(action_dim=int(np.prod(self.action_space.shape)))
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.mlp_extractor.latent_dim_pi, log_std_init=self.log_std_init
            )

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = latent_pi

        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, observations, deterministic: bool = False):
        latent_pi, vf = self.mlp_extractor(observations)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = 1 / (1 + th.exp(-actions))
        return actions, vf, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, vf = self.mlp_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        inv_sigm_actions = th.log(actions / (1 - actions))
        log_prob = distribution.log_prob(inv_sigm_actions)
        entropy = distribution.entropy()
        return vf, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        latent_pi, _ = self.mlp_extractor(obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        _, values = self.mlp_extractor(obs)
        return values

    def _extract_mlp_zone_policies(self) -> Dict[str, nn.Module]:
        zone_policies = {}
        zone_policy_log_std = {}
        for i, zone in enumerate(self.control_zones):
            network = nn.ModuleDict()
            network["actor_network"] = self.mlp_extractor.actor_networks[zone]
            network["critic_network"] = self.mlp_extractor.critic_networks[zone]
            zone_policies[zone] = network
            zone_policy_log_std[zone] = self.log_std[i].item()
        return zone_policies, zone_policy_log_std
