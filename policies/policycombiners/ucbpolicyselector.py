from policies.policycombiners.policycombinersbase import PolicyCombinersBase
from policies.singleagentpolicy import SingleAgentACPolicy

import numpy as np
import os
import pickle as pkl
import torch as th
from typing import List, Tuple


class UCBPolicyCombiner(PolicyCombinersBase):
    def __init__(self,
                 policies: List[SingleAgentACPolicy],
                 device: th.device = th.device("cpu"),
                 update_frequency: int = 96,
                 rho: float = 2.0,
                 selection_cutoff: int = 1440,
                 save_path: str = "",
                 prev_month_num: int = 1,
                 **kwargs):
        super().__init__(policies, device)

        self.update_frequency = update_frequency
        self.selection_count = 0
        self.selection_cutoff = selection_cutoff
        self.policy_paths = kwargs["policy_paths"]

        self.rho = rho
        self.arm_count = np.zeros(len(self.policies), dtype=np.int32)
        self.arm_scores = np.ones(len(self.policies)) * np.inf
        self.arm_scores_all = [[] for _ in range(len(self.policies))]

        self.all_arm_reward_buffer = [[] for _ in range(len(self.policies))]

        if "ucb_reward_limits" not in kwargs["policy_map_config"]:
            raise ValueError("ucb_reward_limits not provided")
        self.ucb_reward_limits = kwargs["policy_map_config"]["ucb_reward_limits"]
        self.zone = kwargs["zone"]
        self.min_reward = self.ucb_reward_limits["min"][kwargs["zone"]]
        self.max_reward = self.ucb_reward_limits["max"][kwargs["zone"]]

        self.save_path = os.path.join(save_path, "ucb_logs")
        if prev_month_num == 12:
            self.current_month_num = 1
        else:
            self.current_month_num = prev_month_num + 1
        self.initialize_save_path()

    def __call__(self, features: th.Tensor) -> Tuple:
        return self.forward(features)

    def forward(self, features: th.Tensor) -> Tuple:
        if 0 < self.selection_count < self.selection_cutoff:
            if self.selection_count % self.update_frequency == 0:
                self.update_ucb()
                self.chosen_arm_idx = self.select_policy()
                self.all_arm_reward_buffer = [[] for _ in range(len(self.policies))]

            action = self.policies[self.chosen_arm_idx].forward(features)
            self.selection_count += 1
            return action
        else:
            self.chosen_arm_idx = self.select_greedy_policy()
            self.selection_count += 1
            return self.policies[self.chosen_arm_idx].forward(features)

    def get_distribution(self, features: th.Tensor) -> th.distributions.Distribution:
        greedy_policy = self.select_greedy_policy()
        return self.policies[greedy_policy].get_distribution(features)

    def update_ucb(self) -> None:
        for arm_idx, arm_reward_buffer in enumerate(self.all_arm_reward_buffer):
            if len(arm_reward_buffer) > 0:
                # if self.arm_count[arm_idx] % len(arm_reward_buffer) != 0:
                #     raise ValueError("Arm selection count and arm reward buffer length mismatch")
                sum_arm_reward = np.sum(arm_reward_buffer)
                normalized_arm_reward = (-sum_arm_reward - self.min_reward) / (self.max_reward - self.min_reward)
                if normalized_arm_reward < 0 or normalized_arm_reward > 1:
                    print(f"sum_arm_reward, normalized_arm_reward, self.min_reward: {sum_arm_reward, normalized_arm_reward, self.min_reward}")
                    if normalized_arm_reward < 0:
                        normalized_arm_reward = 0
                    elif normalized_arm_reward > 1:
                        normalized_arm_reward = 1
                    # raise ValueError("Normalized arm reward out of bounds")
                self.arm_scores_all[arm_idx].append(-normalized_arm_reward)
                self.arm_scores[arm_idx] = np.mean(self.arm_scores_all[arm_idx])
        self.save_ucb_data()

    def set_arm_reward(self, reward: float) -> None:
        self.all_arm_reward_buffer[self.chosen_arm_idx].append(reward)

    def select_policy(self) -> int:
        ucb_values = np.array([self.calc_ucb_value(arm_count, np.sum(self.arm_count), rho=self.rho) for arm_count in self.arm_count])
        chosen_arm_idx = np.argmax(self.arm_scores + ucb_values)

        self.arm_count[chosen_arm_idx] += 1
        return chosen_arm_idx

    def select_greedy_policy(self) -> int:
        # ucb_values = np.array(self.calc_ucb_value(arm_count, np.sum(self.arm_counts), rho=self.rho) for arm_count in self.arm_counts)
        chosen_arm_idx = np.argmax(self.arm_scores)

        return chosen_arm_idx

    def calc_ucb_value(self, arm_count: int, total_count: int, rho: float) -> float:
        if total_count == 0 or arm_count == 0:
            return np.inf
        return np.sqrt(rho * np.log(total_count) / arm_count)

    def save_ucb_data(self):
        arm_scores_all_save_path = os.path.join(self.save_path, f"zone_{self.zone}", f"{self.current_month_num}.pkl")
        log_data_save_path = os.path.join(self.save_path, f"zone_{self.zone}", f"{self.current_month_num}_log_data.csv")

        with open(arm_scores_all_save_path, "wb+") as f:
            pkl.dump(self.arm_scores_all, f)

        log_data = ""
        for i in range(len(self.policies)):
            log_data += f"{self.policy_paths[i]},"
            log_data += f"{self.arm_scores[i]},"
            log_data += f"{self.arm_count[i]},"
        log_data += "\n"

        with open(log_data_save_path, "a+") as f:
            f.write(log_data)

    def initialize_save_path(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(os.path.join(self.save_path, f"zone_{self.zone}")):
            os.makedirs(os.path.join(self.save_path, f"zone_{self.zone}"))

        log_data_header = ""
        for i in range(len(self.policies)):
            log_data_header += f"arm_{i+1}_path,"
            log_data_header += f"arm_{i+1}_score,"
            log_data_header += f"arm_{i+1}_count,"

        with open(os.path.join(self.save_path, f"zone_{self.zone}",
                               f"{self.current_month_num}_log_data.csv"), "w+") as f:
            f.write(log_data_header + "\n")
