from ope.opebase import OPEBase

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
                 no_grad: bool = False, **kwargs):
        """Class constructor for the InverseProbabilityWeighting class

        Args:
            log_data (pandas.DataFrame): The dataset
        """
        self.log_data = log_data
        self.univariate_action = univariate_action
        self.no_grad = no_grad

    def evaluate_policy(self, evaluation_policy_distribution_fuc: callable = None,
                        behavior_policy: dict = None, score: str = "mean", **kwargs):
        """Method to conduct offline policy evaluation

        Args:
            eval_action_model (torch.model): The trained actor model
            behavior_action_model (dict): The saved action model generated from log data
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
            if self.no_grad:
                with torch.no_grad():
                    action_dist = evaluation_policy_distribution_fuc(state).distribution
            else:
                action_dist = evaluation_policy_distribution_fuc(state).distribution
            action_prob[i] = self.calculate_action_probability(action_dist, action_bin,
                                                               behavior_policy["action_bins"])
            behavior_prob[i] = behavior_policy[state_bins_str][action_bin] / \
                behavior_policy["total_count"]
            rewards[i] = row["reward"]
            states.append(state)

        iw = action_prob / behavior_prob
        ret_data = iw * rewards
        if score == "mean":
            return ret_data.mean(), states, rewards, action_prob, behavior_prob
        else:
            return ret_data, states, rewards, action_prob, behavior_prob

    def inv_sigmoid(self, value):
        return np.log(value / (1 - value))

    def calculate_action_probability(self, dist, bin, action_bins):
        bin_l = self.inv_sigmoid(action_bins[bin])
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
                        behavior_policy: dict = None, score: str = "mean", **kwargs):
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
        if score == "mean":
            return ret_data.mean()
        else:
            return ret_data

    def inv_sigmoid(self, value):
        return np.log(value / (1 - value))

    def calculate_action_probability(self, dist, bin, action_bins):
        bin_l = self.inv_sigmoid(action_bins[bin])
        if bin == len(action_bins-1):
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
