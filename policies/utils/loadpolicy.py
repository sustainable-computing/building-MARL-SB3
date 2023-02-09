import torch as th
import os
import glob
from gym import spaces
import numpy as np
import yaml


def load_policy_library(policy_library_path: str, policy_type: str,
                        init_log_std: float = np.log(0.1), init_log_std_path: str = "",
                        eval_mode: bool = True):
    policy_paths = glob.glob(os.path.join(policy_library_path, "**.pt**"))
    if len(policy_paths) == 0:
        raise ValueError(f"No policies found in {policy_library_path}")
    policies = []
    for policy_path in policy_paths:
        policy = th.load(policy_path)
        if "actor_network" in policy.keys():
            assert init_log_std_path != "",\
                "init_log_std_path must be specified if using this policy type."
            obs_size = policy["actor_network"][0].in_features
            for layer in policy["actor_network"]:
                if hasattr(layer, "out_features"):
                    action_size = layer.out_features
            with open(init_log_std_path, "r") as f:
                init_log_std = yaml.safe_load(f)
            init_log_std = init_log_std[policy_path]
        else:
            # Older type of policies
            keys = sorted(list(policy.keys()))

            # Not sure if this is 100% bulletproof
            last_layer_idx = np.max([int(key.split(".")[1]) for key in keys])

            obs_size = policy["actor.0.weight"].shape[1]
            action_size = policy[f"actor.{last_layer_idx}.weight"].shape[0]
            for key in keys:
                if "actor" in key:
                    policy[key.replace("actor", "actor_network")] = policy.pop(key)
                elif "critic" in key:
                    policy[key.replace("critic", "critic_network")] = policy.pop(key)

        policy = load_policy(policy, policy_type, obs_size, action_size, init_log_std)
        if eval_mode:
            policy.eval()
        policies.append(policy)
    return policies, policy_paths


def load_policy(policy, policy_type, obs_size, action_size, init_log_std):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,))
    action_space = spaces.Box(low=0, high=1, shape=(action_size,))
    if hasattr(policy_type, "value"):
        policy_type = policy_type.value
    if policy_type == "single_agent_ac":
        from policies.singleagentpolicy import SingleAgentACPolicy
        policy_obj = SingleAgentACPolicy(observation_space=obs_space,
                                         action_space=action_space,
                                         log_std_init=init_log_std)
    elif policy_type == "multi_agent_ac":
        # from policies.multiagentpolicy import MultiAgentACPolicy
        # policy_obj = MultiAgentACPolicy(observation_space=obs_space,
        #                                 action_space=action_space,
        #                                 init_log_std=init_log_std)
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid policy type {policy_type}")

    policy_obj.mlp_extractor.load_state_dict(policy)
    return policy_obj


if __name__ == "__main__":
    policies = load_policy_library("data/policy_libraries/policy_library_20220820", "single_agent_ac")
    print(policies)
