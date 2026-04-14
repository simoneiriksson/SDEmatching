import torch
import torch.nn as nn
import normflows as nf
import matplotlib.pyplot as plt

class Prior(nn.Module):
    """
    Base class for prior distributions.

    Args:
        device (str): Device to run the computations on.

    Methods:
        sample: Samples from the prior (output shape: [n_samples, dim]).
        log_prob: Computes log-probability of samples (input shape: [n_samples, dim]).
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def sample(self, n_samples=1):
        raise NotImplementedError("Emission class must implement the sample method.")

    def log_prob(self, sample):
        raise NotImplementedError("Emission class must implement the log_prob method.")

class GaussianPrior(Prior):
    """
    Gaussian prior distribution.

    Args:
        mean (torch.Tensor): Mean vector (shape: [dim]).
        std (torch.Tensor): Standard deviation (scalar or shape: [dim]).
        device (str): Device to run the computations on.

    Methods:
        sample: Samples from the Gaussian prior (output shape: [n_samples, dim]).
        log_prob: Computes log-probability of samples (input shape: [n_samples, dim]).
    """
    def __init__(self, mean=torch.tensor(0), log_std=torch.tensor(1), device='cpu', trainable=False):
        super().__init__(device=device)
        self.trainable = trainable
        self.dim = mean.shape[0]
        if self.trainable:
            self.log_std = torch.nn.Parameter(log_std.to(self.device))
            self.mean = torch.nn.Parameter(mean.to(self.device))

        else:
            self.log_std = log_std.to(self.device)
            self.mean = mean.to(self.device)

    def sample(self, n_samples=1):
        eps = torch.randn(n_samples, self.dim, device=self.device)
        emission = eps * self.log_std.exp() + self.mean.unsqueeze(0)
        return emission

    def log_prob(self, sample):
        sample = sample.to(self.device)
        logprob = torch.distributions.MultivariateNormal(self.mean, torch.eye(self.dim, device=self.device) * (self.log_std * 2).exp()).log_prob(sample)
        return logprob
    
class NFPrior(Prior):
    """
    Normalizing Flow-based prior distribution.

    Args:
        prior_nf (nf.NormalizingFlow): Normalizing flow model.
        emission_dim (int): Dimensionality of the emission space.
        device (str): Device to run the computations on.

    Methods:
        sample: Samples from the prior (output shape: [n_samples, emission_dim]).
        log_prob: Computes log-probability of samples (input shape: [n_samples, emission_dim]).
    """
    def __init__(self, prior_nf, emission_dim, device='cpu'):
        super().__init__(device=device)
        self.emission_dim = emission_dim
        self.prior_nf = prior_nf
        if not isinstance(self.prior_nf, nf.NormalizingFlow):
            raise ValueError("prior_nf must be an instance of normflow.NormalizingFlow")

    def sample(self, n_samples=1, eps=None):
        if eps is None:
            return self.prior_nf.sample(num_samples=n_samples)[0].to(self.device)
        else:
            return self.prior_nf.forward(eps.to(self.device))

    def sample_and_log_prob(self, n_samples=1, eps=None):
        if eps is None:
            samples, log_prob = self.prior_nf.sample(num_samples=n_samples)
            return samples.to(self.device), log_prob.to(self.device)
        else:
            return self.prior_nf.forward_and_log_det(eps.to(self.device))

    def log_prob(self, sample):
        return self.prior_nf.log_prob(sample.to(self.device))

    def __repr__(self):
        return f"GeneralNFEmission(emission_dim={self.emission_dim}, internal_dim={self.internal_dim}, prior_nf={self.prior_nf.__class__.__name__})"



