import numpy as np
import torch
import torch.nn as nn


class AbstractProposal(nn.Module):
    """
    Abstract proposal for distribution estimation.
    When coding a new proposal, one should inherit from this class and implement the following methods:
        - log_prob_simple(x): compute the log probability of the proposal.
        - sample_simple(nb_sample): sample from the proposal.

    Attributes:
    -----------
        input_size: input size of the proposal.

    Methods:
    --------
        get_data(dataset, nb_sample_for_init): get a subset of data from the dataset to initialize the proposal.
        log_prob(x): compute the log probability of the proposal.
        sample(nb_sample): sample from the proposal.
    """

    def __init__(
        self,
        input_size,
    ) -> None:
        super().__init__()
        self.input_size = input_size

    def get_data(self, dataset, nb_sample_for_init):
        """
        Consider a subset of data for initialization
        """
        index = np.random.choice(len(dataset), min(nb_sample_for_init, len(dataset)))
        data = torch.cat([dataset.__getitem__(i)["data"] for i in index]).reshape(
            -1, *self.input_size
        )
        return data

    def log_prob_simple(self, x):
        raise NotImplementedError

    def sample_simple(
        self,
        nb_sample,
    ):
        raise NotImplementedError

    def log_prob(self, x):
        """
        Compute the log probability of the proposal.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        return self.log_prob_simple(x).reshape(x.shape[0], 1)

    def sample(self, nb_sample, return_log_prob=False):
        """
        Sample from the proposal.
        """
        samples = self.sample_simple(nb_sample=nb_sample)
        if return_log_prob:
            log_prob = self.log_prob_simple(samples)
            return samples, log_prob
        else:
            return samples
