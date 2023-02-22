from algorithms.distributions.diaggaussian import MultiAgentDiagGaussianDistribution

import gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule
from typing import Optional, List, Union, Dict, Type, Tuple
import torch as th
from torch import nn


class SingleAgentNetworkV1(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        device: th.device = th.device("cpu"),
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.actor_network = self.base_network(feature_dim, action_dim)
        self.critic_network = self.base_network(feature_dim, action_dim)

        self.train_device = device

        self.latent_dim_pi = self.action_dim
        self.latent_dim_vf = self.action_dim

    @classmethod
    def base_network(cls, feature_dim, action_dim) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        actions = self.actor_network(features)
        values = self.critic_network(features)

        return actions, values

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        actions = self.actor_network(features)
        return actions

    def forward_critics(self, features: th.Tensor) -> th.Tensor:
        values = self.critic_network(features)
        return values


class SingleAgentACPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 lr_schedule: Schedule = lambda _: 0.1,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 device: th.device = th.device("cpu"),
                 *args,
                 **kwargs):
        self.train_device = device
        self.features_dim = observation_space.shape[0]
        self.actions_dim = action_space.shape[0]

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )

        self._rebuild(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.features_dim = self.observation_space.shape[0]
        self.mlp_extractor = SingleAgentNetworkV1(
            feature_dim=self.features_dim,
            action_dim=self.actions_dim,
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
        if isinstance(observations, list):
            observations = th.Tensor(observations)
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

    def get_probability(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        latent_pi, _ = self.mlp_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        inv_sigm_actions = th.log(actions / (1 - actions))
        log_prob = distribution.log_prob(inv_sigm_actions)
        return log_prob.exp()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        latent_pi, _ = self.mlp_extractor(obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        _, values = self.mlp_extractor(obs)
        return values
