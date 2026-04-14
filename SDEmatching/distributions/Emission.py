import torch
import torch.nn as nn
import normflows as nf
from SDEMatching.utils import to_tensor
class Emission(nn.Module):
    """
    Base class for emission functions.
            
    Methods:
        sample: Samples emissions from the distribution.
            Args:
                state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
                n_samples (int): Number of samples to generate.
            Returns:
                torch.Tensor: Emission samples (shape: [n_samples, batch_size, observation_dim]).
        log_prob: Computes the log-probability of emissions.
            Args:
                sample (torch.Tensor): Emission samples (shape: [n_samples, batch_size, observation_dim]).
                state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            Returns:
                torch.Tensor: Log-probabilities (shape: [n_samples, batch_size]).
    """
    def __init__(self):
        super(Emission, self).__init__()

    def sample(self,  state, n_samples=1):
        # takes samples from distribution. output shape is #n_samples * #state_dim * #observation_dim
        raise NotImplementedError("Emission class must implement the sample method.")
    def log_prob(self, sample, state):
        raise NotImplementedError("Emission class must implement the log_prob method.")
    
class DistEmission(Emission):
    """
    Emission class for arbitrary distributions.

    Args:
        dist (callable): A callable that returns a distribution object when given a state.
        device (str): Device to run the computations on.
        **kwargs: Additional arguments for the distribution.

    Methods:
        sample: Samples emissions from the distribution.
        log_prob: Computes the log-probability of emissions.
    """
    def __init__(self, dist, device='cpu', **kwargs):
        super().__init__()
        self.dist = dist
        self.device = device
        self.kwargs = kwargs

    def sample(self, state, n_samples=1):
        """
        Samples emissions from the distribution.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Emission samples (shape: [n_samples, batch_size, observation_dim]).
        """
        state = state.to(self.device)
        return self.dist(state).sample((n_samples,))

    def log_prob(self, sample, state):
        """
        Computes the log-probability of emissions.

        Args:
            sample (torch.Tensor): Emission samples (shape: [n_samples, batch_size, observation_dim]).
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).

        Returns:
            torch.Tensor: Log-probabilities (shape: [n_samples, batch_size]).
        """
        sample, state = sample.to(self.device), state.to(self.device)
        logprob = self.dist(state).log_prob(sample)
        return logprob

class GaussianEmission(Emission):
    """
    Gaussian emission model.

    Args:
        log_std (torch.Tensor): Log standard deviation (scalar or shape: [dim]).
        dim (int): Dimensionality of the emission space.
        device (str): Device to run the computations on.

    Methods:
        sample: Samples emissions from the Gaussian distribution.
        log_prob: Computes the log-probability of emissions.
    """
    def __init__(self, log_std=0.0, dim=None, device='cpu', trainable=False):
        super(GaussianEmission, self).__init__()
        assert dim is not None
        self.dim = dim
        self.device = device
        self.trainable = trainable
        if trainable:
            self.log_std = torch.nn.Parameter(to_tensor(log_std).to(device))
        else:
            self.log_std = to_tensor(log_std).to(device)
            
    def sample(self, state, n_samples=1):
        """
        Samples emissions from the Gaussian distribution.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Emission samples (shape: [n_samples, batch_size, dim]).
        """
        state = state.to(self.device)
        #eps = torch.randn(n_samples, *state.shape, device=self.device)
        #return eps * self.log_std.exp() + state.unsqueeze(0)
        
        return torch.distributions.MultivariateNormal(
            state, torch.eye(state.shape[1], device=self.device) * (self.log_std * 2).exp()).sample((1,))


    def log_prob(self, sample, state):
        """
        Computes the log-probability of emissions.

        Args:
            sample (torch.Tensor): Emission samples (shape: [n_samples, batch_size, dim]).
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).

        Returns:
            torch.Tensor: Log-probabilities (shape: [n_samples, batch_size]).
        """
        sample, state = sample.to(self.device), state.to(self.device)
        logprob = torch.distributions.MultivariateNormal(state, torch.eye(state.shape[1], device=self.device) * (self.log_std * 2).exp()).log_prob(sample)
        return logprob
    
class NFEmission(Emission):
    """
    Normalizing Flow-based emission model.

    Args:
        emission_dim (int): Dimensionality of the emission space.
        emission_nf (nf.ConditionalNormalizingFlow): Normalizing flow model.
        device (str): Device to run the computations on.

    Methods:
        sample: Samples emissions using the normalizing flow.
        sample_and_log_det: Samples emissions and computes the log-determinant.
        log_prob: Computes the log-probability of emissions.
    """
    def __init__(self, emission_dim, emission_nf, device='cpu'):
        super(NFEmission, self).__init__()
        self.emission_dim = emission_dim
        self.emission_nf = emission_nf
        self.device = device
        if not isinstance(self.emission_nf, nf.ConditionalNormalizingFlow):
            raise ValueError("emission_nf must be an instance of normflow.ConditionalNormalizingFlow")

    def sample(self, state, n_samples=1, eps=None):
        """
        Samples emissions using the normalizing flow.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            n_samples (int): Number of samples to generate.
            eps (torch.Tensor, optional): Latent variable (shape: [n_samples, batch_size, latent_dim]).

        Returns:
            torch.Tensor: Emission samples (shape: [n_samples, batch_size, emission_dim]).
        """
        state = state.to(self.device)
        if eps is None:
            return self.emission_nf.sample(num_samples=n_samples, context=state)[0]
        else:
            return self.emission_nf.forward(eps.to(self.device), context=state)

    def sample_and_log_det(self, state, n_samples=1, eps=None):
        """
        Samples emissions and computes the log-determinant.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            n_samples (int): Number of samples to generate.
            eps (torch.Tensor, optional): Latent variable (shape: [n_samples, batch_size, latent_dim]).

        Returns:
            tuple: Emission samples (shape: [n_samples, batch_size, emission_dim]) and log-determinant (shape: [n_samples, batch_size]).
        """
        state = state.to(self.device)
        if eps is None:
            return self.emission_nf.sample(num_samples=n_samples, context=state)
        else:
            return self.emission_nf.forward_and_log_det(eps.to(self.device), context=state)

    def log_prob(self, sample, state):
        """
        Computes the log-probability of emissions.

        Args:
            sample (torch.Tensor): Emission samples (shape: [n_samples, batch_size, emission_dim]).
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).

        Returns:
            torch.Tensor: Log-probabilities (shape: [n_samples, batch_size]).
        """
        sample, state = sample.to(self.device), state.to(self.device)
        return self.emission_nf.log_prob(sample, context=state)

    def __repr__(self):
        return f"NFEmission(emission_dim={self.emission_dim}, emission_nf={self.emission_nf.__class__.__name__})"

class DeltaEmission(nn.Module):
    """
    Delta emission model (deterministic).

    Args:
        device (str): Device to run the computations on.

    Methods:
        sample: Returns deterministic emissions equal to the state.
        log_prob: Returns zero log-probability for all emissions.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def sample(self, state, n_samples=1):
        """
        Returns deterministic emissions equal to the state.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Deterministic emissions (shape: [n_samples, batch_size, state_dim]).
        """
        state = state.to(self.device)
        return state.unsqueeze(0).repeat([n_samples] + [1] * state.dim())

    def log_prob(self, sample, state):
        """
        Returns zero log-probability for all emissions.

        Args:
            sample (torch.Tensor): Emission samples (shape: [n_samples, batch_size, state_dim]).
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).

        Returns:
            torch.Tensor: Zero log-probabilities (shape: [n_samples, batch_size]).
        """
        return torch.zeros(sample.shape[0], device=self.device)
