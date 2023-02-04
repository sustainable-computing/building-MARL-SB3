from policies.multiagentpolicy import MultiAgentACPolicy
import torch as th
import os
import yaml


def save_zone_policies(policy, save_dir, prefix=None, suffix=None, extension="pt"):
    zone_policies, zone_policies_log_std = extract_zone_policies(policy)
    policy_loc_log_std = {}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for zone, policy in zone_policies.items():
        save_name = construct_save_name(zone, prefix, suffix, extension)
        save_loc = os.path.join(save_dir, save_name)
        policy_loc_log_std[save_name] = zone_policies_log_std[zone]
        th.save(policy, save_loc)

    with open(os.path.join(save_dir, "policy_loc_log_std.yaml"), "w") as f:
        yaml.dump(policy_loc_log_std, f)


def construct_save_name(zone, prefix, suffix, extension):
    if prefix is not None and suffix is not None:
        save_name = f"{prefix}.{zone}.{suffix}.{extension}"
    elif prefix is not None and suffix is None:
        save_name = f"{prefix}.{zone}.{extension}"
    elif prefix is None and suffix is not None:
        save_name = f"{zone}.{suffix}.{extension}"
    else:
        save_name = f"{zone}.{extension}"
    return save_name


def extract_zone_policies(policy):
    if isinstance(policy, MultiAgentACPolicy):
        return extract_multiagent_ac_policies(policy)


def extract_multiagent_ac_policies(policy):
    zone_policies = policy._extract_mlp_zone_policies()
    return zone_policies
