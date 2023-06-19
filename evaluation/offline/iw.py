from evaluation.offline.opebase import OPEBase

import pandas as pd
import numpy as np
from scipy.integrate import quad
import torch


class InverseProbabilityWeighting(OPEBase):
    """Inverse probability weighting

    Source code adapted from https://github.com/st-tech/zr-obp
    """

    def __init__(self, log_data: pd.DataFrame = None,
                 univariate_action: bool = True,
                 no_grad: bool = False,
                 rule_based_behavior_policy: bool = True,
                 **kwargs):
        """Class constructor for the InverseProbabilityWeighting class

        Args:
            log_data (pandas.DataFrame): The dataset
        """
        self.log_data = log_data
        self.univariate_action = univariate_action
        self.no_grad = no_grad
        self.rule_based_behavior_policy = rule_based_behavior_policy

    def evaluate_policy(self, evaluation_policy_distribution_fuc: callable = None,
                        behavior_policy: dict = None, score: str = "mean",
                        return_additional: bool = True, **kwargs):
        """Method to conduct offline policy evaluation

        Args:
            eval_action_model (torch.model): The trained actor model
            behavior_action_model (dict): The saved action model generated from log data
            score (str): String indicating which scoring metric to use. Default is mean
        """
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = "cpu"

        data = self.log_data.to_dict("records")
        action_prob = torch.zeros((len(data)), device=self.device)
        rewards = torch.zeros((len(data)), device=self.device)

        behavior_prob = torch.zeros((len(data)), device=self.device)
        states = []

        if not self.rule_based_behavior_policy:
            behavior_policy = behavior_policy.get_distribution

        for i, row in enumerate(data):
            state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                          "zone_humidity", "zone_temp", "zone_occupancy"]
            state = torch.tensor([row[var] for var in state_vars], device=self.device)
            if self.rule_based_behavior_policy:
                state_bins = [np.digitize(row[var],
                              behavior_policy[f"{var}_bins"])-1 for var in state_vars]
                action_bin = np.digitize(row["action"], behavior_policy["action_bins"]) - 1
                state_bins_str = "{},{},{},{},{},{}".format(*state_bins)
                if self.no_grad:
                    with torch.no_grad():
                        action_dist = evaluation_policy_distribution_fuc(state).distribution
                else:
                    action_dist = evaluation_policy_distribution_fuc(state).distribution
                action_prob[i] = self.calculate_action_probability(action_dist, action_bin,
                                                                behavior_policy["action_bins"])
                try:
                    behavior_prob[i] = behavior_policy[state_bins_str][action_bin] / \
                        behavior_policy["total_count"]
                except KeyError:
                    behavior_prob[i] = 1
                rewards[i] = row["reward"]
                states.append(state)
            else:
                action = torch.tensor([row["action"]], device=self.device)
                if self.no_grad:
                    with torch.no_grad():
                        action_dist = evaluation_policy_distribution_fuc(state).distribution
                        behavior_dist = behavior_policy(state).distribution
                else:
                    action_dist = evaluation_policy_distribution_fuc(state).distribution
                    behavior_dist = behavior_policy(state).distribution
                action_prob[i] = action_dist.log_prob(self.inv_sigmoid(action)).exp()
                behavior_prob[i] = behavior_dist.log_prob(self.inv_sigmoid(action)).exp()
                rewards[i] = row["reward"]
                states.append(state)

        behavior_prob += 1e-10
        iw = action_prob / behavior_prob
        iw = torch.clamp(iw, 0, 1e4)
        ret_data = iw * rewards
        if not return_additional:
            if score == "mean":
                return ret_data.mean()
            else:
                return ret_data
        else:
            if score == "mean":
                return ret_data.mean(), states, rewards, action_prob, behavior_prob
            else:
                return ret_data, states, rewards, action_prob, behavior_prob

    def optimized_evaluate_policy(self, evaluation_policy_distribution_fuc: callable = None,
                                  behavior_policy: dict = None,
                                  score: str = "mean",
                                  return_additional: bool = True,
                                  **kwargs):
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = "cpu"

        state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                      "zone_humidity", "zone_temp", "zone_occupancy"]
        rewards = torch.tensor(self.log_data["reward"].values.reshape(-1, 1),
                               device=self.device,
                               dtype=torch.float32)
        # rewards_1 = torch.tensor(self.log_data["cooling_energy"].values.reshape(-1, 1),
        #                          device=self.device,
        #                          dtype=torch.float32)
        # rewards_2 = torch.tensor(self.log_data["heating_energy"].values.reshape(-1, 1),
        #                          device=self.device,
        #                          dtype=torch.float32)
        # rewards = rewards_1 + rewards_2

        if not self.rule_based_behavior_policy:
            states = torch.tensor(self.log_data[state_vars].values.astype(np.float),
                                  device=self.device,
                                  dtype=torch.float32)
            actions = torch.tensor(self.log_data["action"].values.reshape(-1, 1),
                                   device=self.device,
                                   dtype=torch.float32)
            if self.no_grad:
                with torch.no_grad():
                    action_dist = evaluation_policy_distribution_fuc(states).distribution
                    behavior_dist = behavior_policy(states).distribution
            else:
                action_dist = evaluation_policy_distribution_fuc(states).distribution
                behavior_dist = behavior_policy.get_distribution(states).distribution

            action_probs = action_dist.log_prob(self.inv_sigmoid(actions)).exp()
            behavior_probs = behavior_dist.log_prob(self.inv_sigmoid(actions)).exp()
        else:
            # raise NotImplementedError("Optimized rule based policy evaluation not implemented yet")
            states = torch.tensor(self.log_data[state_vars].values.astype(np.float),
                                  device=self.device,
                                  dtype=torch.float32)
            states_bins = torch.zeros_like(states, device=self.device, dtype=torch.float32)
            action_bins = torch.tensor(np.digitize(self.log_data["action"].values.round(3),
                                                    behavior_policy["action_bins"])-1,
                                        device=self.device,
                                        dtype=torch.float32).reshape(-1, 1)
            state_bin_strs = []
            for i, state_var in enumerate(state_vars):
                states_bins[:, i] = torch.tensor(np.digitize(states[:, i],
                                                             behavior_policy[f"{state_var}_bins"])-1,
                                                 device=self.device,
                                                 dtype=torch.float32)
            for i in range(states_bins.shape[0]):
                state_bin_strs.append("{},{},{},{},{},{}".format(*states_bins[i].numpy().astype(np.int).tolist()))
            behavior_probs = torch.zeros((states_bins.shape[0], 1), device=self.device,
                                            dtype=torch.float32)
            for i, state_bin_str in enumerate(state_bin_strs):
                if state_bin_str not in behavior_policy:
                    print(state_bin_str, "not in behavior policy")
                    behavior_probs[i] = 1
                elif action_bins[i].item() not in behavior_policy[state_bin_str]:
                    print(action_bins[i].item(), "not in behavior policy with state", state_bin_str)
                    behavior_probs[i] = 1
                else:
                    behavior_probs[i] = behavior_policy[state_bin_str][action_bins[i].item()] / \
                        behavior_policy["total_count"]
            if self.no_grad:
                with torch.no_grad():
                    action_dist = evaluation_policy_distribution_fuc(states).distribution
            else:
                action_dist = evaluation_policy_distribution_fuc(states).distribution
            action_probs = self.optimized_calculate_action_probability(action_dist, action_bins,
                                                                       behavior_policy["action_bins"])

        behavior_probs = torch.clamp(behavior_probs, min=1e-10)
        iw = action_probs / behavior_probs
        iw = torch.clamp(iw, 0, 1e4)
        ret_data = iw * rewards
        if not return_additional:
            if score == "mean":
                return ret_data.mean()
            else:
                return ret_data
        else:
            if score == "mean":
                return ret_data.mean(), states, rewards, action_probs, behavior_probs
            else:
                return ret_data, states, rewards, action_probs, behavior_probs

    def inv_sigmoid(self, value):
        if self.device == "cpu":
            return np.log(value / (1 - value))
        else:
            return torch.log(value / (1 - value))

    def optimized_calculate_action_probability(self, dist, bin, action_bins):
        left_bins = torch.zeros_like(bin)
        right_bins = torch.zeros_like(bin)
        bin = bin.numpy().astype(np.int)
        for i in range(bin.shape[0]):
            left_sigm_action_bin = action_bins[bin[i].item()]
            if left_sigm_action_bin == 0:
                left_bins[i] = -np.inf
            else:
                left_bins[i] = self.inv_sigmoid(left_sigm_action_bin)
            if bin[i] == len(action_bins) - 1:
                right_bins[i] = np.inf
            else:
                right_bins[i] = self.inv_sigmoid(action_bins[bin[i].item()+1])
        integral = dist.cdf(right_bins) - dist.cdf(left_bins)
        return integral

    def calculate_action_probability(self, dist, bin, action_bins):
        bin_l = self.inv_sigmoid(action_bins[bin])
        if bin == len(action_bins) - 1:
            bin_r = np.inf
        else:
            bin_r = self.inv_sigmoid(action_bins[bin+1])
        if self.univariate_action:
            integral = dist.cdf(torch.Tensor([bin_r])) - dist.cdf(torch.Tensor([bin_l]))
        else:
            func = lambda inp: dist.log_prob(inp).exp()
            if bin_l == -np.inf:
                bin_l = -10
            if bin_r == np.inf:
                bin_r = 10
            # if self.retain_grad_fn:
            #     int_method = torchquad.Trapezoid()
            #     integral = int_method.integrate(func, dim=1, N=1000,
            #                                     integration_domain=[[bin_l, bin_r]],
            #                                     backend="torch")
            # else:
            #     integral, err = quad(func, bin_l, bin_r)
        return integral


class SelfNormalizedInverseProbabilityWeighting(OPEBase):
    """Self-Normalized Inverse probability weighting

    Source code adapted from https://github.com/st-tech/zr-obp
    """

    def __init__(self, log_data: pd.DataFrame = None,
                 univariate_action: bool = True):
        """Class constructor for the InverseProbabilityWeighting class

        Args:
            log_data (pandas.DataFrame): The dataset
        """
        self.log_data = log_data
        self.univariate_action = univariate_action

    def evaluate_policy(self, evaluation_policy_distribution_fuc: callable = None,
                        behavior_policy: dict = None, score: str = "mean",
                        return_additional=True, **kwargs):
        """Method to conduct offline policy evaluation

        Args:
            eval_action_model (torch.model): The trained actor model
            behavior_action_model (Dict): The saved action model generated from log data
            score (str): String indicating which scoring metric to use. Default is mean
        """
        data = self.log_data.to_dict("records")
        action_prob = torch.zeros((len(data)))
        rewards = torch.zeros((len(data)))

        behavior_prob = torch.zeros((len(data)))
        states = []

        for i, row in enumerate(data):
            state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                          "zone_humidity", "zone_temp", "zone_occupancy"]
            state = torch.Tensor([row[var] for var in state_vars])

            state_bins = [np.digitize(row[var],
                          behavior_policy[f"{var}_bins"])-1 for var in state_vars]
            action_bin = np.digitize(row["action"], behavior_policy["action_bins"]) - 1
            state_bins_str = "{},{},{},{},{},{}".format(*state_bins)
            with torch.no_grad():
                action_dist = evaluation_policy_distribution_fuc(state).distribution
            action_prob[i] = self.calculate_action_probability(action_dist, action_bin,
                                                               behavior_policy["action_bins"])
            behavior_prob[i] = behavior_policy[state_bins_str][action_bin] / \
                behavior_policy["total_count"]
            rewards[i] = row["reward"]
            states.append(state)

        iw = action_prob / behavior_prob
        ret_data = iw * rewards / iw.mean()
        if not return_additional:
            if score == "mean":
                return ret_data.mean()
            else:
                return ret_data
        else:
            if score == "mean":
                return ret_data.mean(), states, rewards, action_prob, behavior_prob
            else:
                return ret_data, states, rewards, action_prob, behavior_prob

    def inv_sigmoid(self, value):
        return np.log(value / (1 - value))

    def calculate_action_probability(self, dist, bin, action_bins):
        bin_l = self.inv_sigmoid(action_bins[bin])
        if bin == len(action_bins) - 1:
            bin_r = np.inf
        else:
            bin_r = self.inv_sigmoid(action_bins[bin+1])
        if self.univariate_action:
            integral = dist.cdf(torch.Tensor([bin_r])) - dist.cdf(torch.Tensor([bin_l]))
        else:
            func = lambda inp: dist.log_prob(inp).exp()
            if bin_l == -np.inf:
                bin_l = -10
            if bin_r == np.inf:
                bin_r = 10
            # if self.retain_grad_fn:
            #     int_method = torchquad.Trapezoid()
            #     integral = int_method.integrate(func, dim=1, N=1000,
            #                                     integration_domain=[[bin_l, bin_r]],
            #                                     backend="torch")
            else:
                integral, err = quad(func, bin_l, bin_r)
        return integral
