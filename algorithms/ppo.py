from algorithms.buffers.dictrolloutbuffer import CustomDictRolloutBuffer
from buildingenvs.wrappers import MultiAgentDummyVecEnv
from algorithms.diversity import PPODiversityHandler

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.policies import BasePolicy
import torch as th
from torch.nn import functional as F
import wandb


class MultiAgentPPO(PPO):
    def __init__(self,
                 policy: BasePolicy = None,
                 env: gym.Env = None,
                 diversity_weight: float = 0.0,
                 diverse_policy_library_loc: str = "",
                 diverse_policy_library_log_std_loc: str = "",
                 *args, **kwargs):
        env = MultiAgentDummyVecEnv([lambda: env])
        super().__init__(policy, env, *args, **kwargs)

        if PPODiversityHandler.is_diverse_training(diversity_weight, diverse_policy_library_loc):
            self.diverse_training = True
            self.diversity_handler = PPODiversityHandler(diversity_weight,
                                                         diverse_policy_library_loc,
                                                         diverse_policy_library_log_std_loc,
                                                         device=self.device)
        else:
            self.diverse_training = False

    def _setup_model(self) -> None:

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = CustomDictRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        all_entropy_losses = [[] for _ in range(self.policy.action_space.shape[0])]
        all_pg_losses = [[] for _ in range(self.policy.action_space.shape[0])]
        all_value_losses = [[] for _ in range(self.policy.action_space.shape[0])]
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = [[] for _ in range(self.policy.action_space.shape[0])]
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325

                if self.normalize_advantage and len(advantages) > 1:
                    for idx in range(self.policy.action_space.shape[0]):
                        advantages[:, idx] = (advantages[:, idx] - advantages[:, idx].mean()) / (advantages[:, idx].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean(axis=0)

                # Logging
                for i in range(len(policy_loss)):
                    all_pg_losses[i].append(policy_loss[i].item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred, reduction="none").mean(axis=0)
                for i in range(len(value_loss)):
                    all_value_losses[i].append(value_loss[i].item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob, axis=0)
                else:
                    entropy_loss = -th.mean(entropy, axis=0)
                for i in range(len(entropy_loss)):
                    all_entropy_losses[i].append(entropy_loss[i].item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if self.diverse_training:
                    diversity_loss = \
                        self.diversity_handler.calculate_diversity_loss(rollout_data.observations,
                                                                        rollout_data.actions,
                                                                        rollout_data.returns,
                                                                        log_prob)
                    diversity_loss = self.diversity_handler.diversity_weight * diversity_loss
                    loss += diversity_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio, axis=0).cpu().numpy()
                    for i in range(len(approx_kl_div)):
                        approx_kl_divs[i].append(approx_kl_div[i])

                if self.target_kl is not None and approx_kl_div.mean() > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                for i in range(len(loss)):
                    loss[i].backward(retain_graph=True)
                # loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        for i in range(len(all_entropy_losses)):
            # self.logger.record(f"train/entropy_loss_{i}", np.mean(all_entropy_losses[i]))
            self.log_data(f"train/entropy_loss_{i}", np.mean(all_entropy_losses[i]))
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        for i in range(len(all_pg_losses)):
            # self.logger.record(f"train/policy_gradient_loss_{i}", np.mean(all_pg_losses[i]))
            self.log_data(f"train/policy_gradient_loss_{i}", np.mean(all_pg_losses[i]))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        for i in range(len(all_value_losses)):
            # self.logger.record(f"train/value_loss_{i}", np.mean(all_value_losses[i]))
            self.log_data(f"train/value_loss_{i}", np.mean(all_value_losses[i]))
        # self.logger.record("train/value_loss", np.mean(value_losses))
        for i in range(len(approx_kl_divs)):
            # self.logger.record(f"train/approx_kl_{i}", np.mean(approx_kl_divs[i]))
            self.log_data(f"train/approx_kl_{i}", np.mean(approx_kl_divs[i]))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.log_data("train/clip_fraction", np.mean(clip_fractions))
        for i in range(len(loss)):
            # self.logger.record(f"train/loss_{i}", loss[i].item())
            self.log_data(f"train/loss_{i}", loss[i].item())
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        self.log_data("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            # self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            self.log_data("train/std", th.exp(self.policy.log_std).mean().item())

        if self.diverse_training:
            # self.logger.record("train/diversity_weight", self.diversity_handler.diversity_weight)
            self.log_data("train/diversity_weight", self.diversity_handler.diversity_weight)
            for i in range(len(diversity_loss)):
                # self.logger.record(f"train/diversity_loss_{i}", diversity_loss[i].item())
                self.log_data(f"train/diversity_loss_{i}", diversity_loss[i].item())

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.log_data("train/n_updates", self._n_updates)
        # self.logger.record("train/clip_range", clip_range)
        self.log_data("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            # self.logger.record("train/clip_range_vf", clip_range_vf)
            self.log_data("train/clip_range_vf", clip_range_vf)

    def log_data(self, name, value):
        self.logger.record(f"{name}", value)
        if wandb.run is not None:
            wandb.log({f"{name}": value})
