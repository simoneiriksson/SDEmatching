import torch
import numpy as np
from torch import nn
from torch.nn import functional as F, init

from normflows.flows import Flow
import normflows as nf

from SDEmatching.utils.utils import to_tensor


class AffineFlow(torch.nn.Module):
    """Normal distribution flow"""
    def __init__(self, dim, device='cpu'):
        """Constructor
        Args:
          num_channels: Number of channels of the data
            - methods required: 
                - __call__(epsilon, context = context) = forward taking an epsilon and returning sample
                - inverse(state, context=conditions)
                - sample(num_samples=conditions.shape[0], context=conditions): sampling epsilon from base distribution and returning result
                - forward_and_log_det(z=eps, context=conditions): forward taking an epsilon and returning sample and logprob
                - log_prob(state, context=conditions) return logprob
                
            properties required: 
                - base_dist (being a torch distribution)

        """
        super().__init__()
        self.dim = dim
        self.device = device
        #self.q0 = StandardMultivariateNormal(dim, device=device)  # Use helper class
        self.q0 = torch.distributions.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device)) # Use torch distribution
    def _eps_logprob(self, eps):
        """
        Compute the log probability of epsilon under the base distribution.

        Args:
            eps (torch.Tensor): Tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Log probability of shape (batch_size,).
        """
        eps = eps.to(self.device)
        return -self.dim * 0.5 * torch.log(torch.pi * torch.tensor(2, device=self.device)) - 0.5 * (eps**2).sum(dim=1)

    def forward_and_log_det(self, z, context=None):
        """
        Perform the forward transformation and compute the log determinant.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            tuple: Transformed tensor of shape (batch_size, dim) and log determinant of shape (batch_size,).
        """
        mu, log_std = context.chunk(2, dim=1)
        z = z.to(self.device)        
        log_det = log_std.sum(dim=1) 
        z_ = z * log_std.exp() + mu
        return z_, log_det

    def inverse(self, z, context=None):
        """
        Perform the inverse transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Inverse transformed tensor of shape (batch_size, dim).
        """
        mu, log_std = context.chunk(2, dim=1)
        z = z.to(self.device)        
        z_ = (z - mu) / log_std.exp()
        return z_

    def sample(self, num_samples=1, context=None):
        """
        Sample from the flow.

        Args:
            num_samples (int): Number of samples to generate.
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            tuple: Sampled tensor of shape (num_samples, dim) and log probability of shape (num_samples,).
        """
        eps = self.q0.sample((num_samples,))  # Sample from standard multivariate normal
        z_, log_det = self.forward_and_log_det(eps, context=context)
        return z_, self.q0.log_prob(eps) - log_det

    def forward(self, z, context=None):
        """
        Perform the forward transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, dim).
        """
        return self.forward_and_log_det(z, context=context)[0]

    def log_prob(self, z, context=None):
        """
        Compute the log probability of the input tensor.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Log probability of shape (batch_size,).
        """        
        mu, log_std = context.chunk(2, dim=1)
        z = z.to(self.device)        
        eps = (z - mu) / log_std.exp()
        return self.q0.log_prob(eps) - log_std.sum(dim=1) 



class NormalFlow(torch.nn.Module):
    """Normal distribution flow"""
    def __init__(self, dim, mean_func, log_std_func, device='cpu'):
        """Constructor
        Args:
          num_channels: Number of channels of the data
            - methods required: 
                - __call__(epsilon, context = context) = forward taking an epsilon and returning sample
                - inverse(state, context=conditions)
                - sample(num_samples=conditions.shape[0], context=conditions): sampling epsilon from base distribution and returning result
                - forward_and_log_det(z=eps, context=conditions): forward taking an epsilon and returning sample and logprob
                - log_prob(state, context=conditions) return logprob
                
            properties required: 
                - base_dist (being a torch distribution)

        """
        super().__init__()
        self.mean_func = mean_func
        self.log_std_func = log_std_func
        self.dim = dim
        self.device = device
        #self.q0 = StandardMultivariateNormal(dim, device=device)  # Use helper class
        self.q0 = torch.distributions.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device)) # Use torch distribution

    def _eps_logrpob(self, eps):
        """
        Compute the log probability of epsilon under the base distribution.

        Args:
            eps (torch.Tensor): Tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Log probability of shape (batch_size,).
        """
        eps = eps.to(self.device)
        return -self.dim * 0.5 * torch.log(torch.pi * torch.tensor(2, device=self.device)) - 0.5 * (eps**2).sum(dim=1)

    def forward_and_log_det(self, z, context=None):
        """
        Perform the forward transformation and compute the log determinant.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            tuple: Transformed tensor of shape (batch_size, dim) and log determinant of shape (batch_size,).
        """
        z, context = z.to(self.device), context.to(self.device)
        ts = context[:, 0]
        t1 = context[:, 1]
        X1 = context[:, 2:]
        log_det = self.log_std_func(ts) * X1.shape[1]
        z_ = z * self.log_std_func(ts).exp().unsqueeze(1) + self.mean_func(ts).unsqueeze(1) * X1
        return z_, log_det

    def inverse(self, z, context=None):
        """
        Perform the inverse transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Inverse transformed tensor of shape (batch_size, dim).
        """
        z, context = z.to(self.device), context.to(self.device)
        ts = context[:, 0]
        t1 = context[:, 1]
        X1 = context[:, 2:]
        z_ = (z - self.mean_func(ts).unsqueeze(1) * X1) / self.log_std_func(ts).exp().unsqueeze(1)
        #log_det = self.std_func(ts).unsqueeze(1).log() * X1.shape[1] # correct this
        return z_

    def sample(self, num_samples=1, context=None):
        """
        Sample from the flow.

        Args:
            num_samples (int): Number of samples to generate.
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            tuple: Sampled tensor of shape (num_samples, dim) and log probability of shape (num_samples,).
        """
        context = context.to(self.device)
        #eps = self.q0.sample(num_samples)  # Sample from standard multivariate normal
        eps = self.q0.sample((num_samples,))  # Sample from standard multivariate normal
        z_, log_det = self.forward_and_log_det(eps, context=context)
        return z_, self.q0.log_prob(eps) - log_det

    def forward(self, z, context=None):
        """
        Perform the forward transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, dim).
        """
        z, context = z.to(self.device), context.to(self.device)
        return self.forward_and_log_det(z, context=context)[0]

    def log_prob(self, z, context=None):
        """
        Compute the log probability of the input tensor.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            torch.Tensor: Log probability of shape (batch_size,).
        """
        z, context = z.to(self.device), context.to(self.device)
        ts = context[:, 0]
        t1 = context[:, 1]
        X1 = context[:, 2:]
        log_det = self.log_std_func(ts) * X1.shape[1]
        eps = (z - self.mean_func(ts).unsqueeze(1) * X1) / self.log_std_func(ts).exp().unsqueeze(1)
        return self._eps_logrpob(eps) - log_det


class Linearflow(Flow):
    """
    Invertible affine transformation WITH context-dependent shift
    """

    def __init__(self, num_channels, std=1, device='cpu'):
        """Constructor
        Args:
          num_channels: Number of channels of the data
        """
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.std = to_tensor(std).to(device)
        self.W = nn.Parameter(torch.eye(num_channels, device=self.device) / self.std)


    def forward(self, z, context=None):
        """
        Perform the forward transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, num_channels).
            context (torch.Tensor): Context tensor of shape (batch_size, num_channels).

        Returns:
            tuple: Transformed tensor of shape (batch_size, num_channels) and log determinant of shape (batch_size,).
        """
        z, context = z.to(self.device), context.to(self.device)
        W = torch.inverse(self.W)
        if self.W.device.type == 'mps':
            log_det = torch.slogdet(self.W.cpu())[1]
        else:
            log_det = torch.slogdet(self.W)[1]
        z_ = z @ W + context
        return z_, log_det

    def inverse(self, z, context=None):
        """
        Perform the inverse transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, num_channels).
            context (torch.Tensor): Context tensor of shape (batch_size, num_channels).

        Returns:
            tuple: Inverse transformed tensor of shape (batch_size, num_channels) and log determinant of shape (batch_size,).
        """
        z, context = z.to(self.device), context.to(self.device)
        W = self.W
        if self.W.device.type == 'mps':
            log_det = torch.slogdet(self.W.cpu())[1]
        else:
            log_det = torch.slogdet(self.W)[1]
        z_ = (z - context) @ W
        return z_, log_det
    
class DDPMflow(nf.flows.Flow):
    """
    Invertible affine transformation WITH context-dependent shift
    """
    def __init__(self, mean_func, std_func, device='cpu'):
        """Constructor
        Args:
          num_channels: Number of channels of the data
        """
        super().__init__()
        self.mean_func = mean_func
        self.std_func = std_func
        self.device = device

    def forward(self, z, context=None):
        """
        Perform the forward transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            tuple: Transformed tensor of shape (batch_size, dim) and log determinant of shape (batch_size,).
        """
        z, context = z.to(self.device), context.to(self.device)
        ts = context[:, 0]
        t1 = context[:, 1]
        X1 = context[:, 2:]
        # print(f"{ts.shape = }, {X1.shape = }")
        # print(f"{self.mean_func(ts).shape = }, {self.diff_func(ts).shape = }")
        # print(f"{z.shape = }")
        # print(f"{(self.diff_func(ts).unsqueeze(1) * z).shape = }")
        # print(f"{(self.mean_func(ts).unsqueeze(1) * X1).shape = }")

        # log_det = torch.slogdet(self.diff_func(ts))[0] # correct this
        # # z_ = sigma(t,z) @ z + m(t,z)*X1
        # mul = torch.einsum("bmk, bk -> bk", self.diff_func(ts) , z)
        # z_ =  mul + self.mean_func(ts).unsqueeze(1) * X1
        log_det = -self.std_func(ts).log() * X1.shape[1]
        z_ = z * self.std_func(ts).unsqueeze(1) + self.mean_func(ts).unsqueeze(1) * X1
        # print(f"{log_det.shape = }")
        return z_, log_det

    def inverse(self, z, context=None):
        """
        Perform the inverse transformation.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim).
            context (torch.Tensor): Context tensor of shape (batch_size, context_dim).

        Returns:
            tuple: Inverse transformed tensor of shape (batch_size, dim) and log determinant of shape (batch_size,).
        """
        z, context = z.to(self.device), context.to(self.device)
        ts = context[:, 0]
        t1 = context[:, 1]
        X1 = context[:, 2:]
        # print(f"{ts.shape = }, {X1.shape = }")
        # print(f"{self.mean_func(ts).shape = }, {self.diff_func(ts).shape = }")
        # print(f"{z.shape = }")
        
        #log_det = torch.slogdet(self.diff_func(ts))[0]
        # sigma(t,z_)^-1 @(z - m(t,z_)*X1) =  z_ 
        #z_ = torch.einsum("bmk, bk -> bk", self.diff_func(ts).inverse() , z - self.mean_func(ts).unsqueeze(1) * X1) 
        z_ = (z - self.mean_func(ts).unsqueeze(1) * X1) / self.std_func(ts).unsqueeze(1)
        log_det = self.std_func(ts).unsqueeze(1).log() * X1.shape[1]
        #print(f"{log_det.shape = }")
        return z_, log_det.squeeze(1)

