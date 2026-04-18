import torch
import torch.nn as nn
import normflows as nf
from torchdyn.core import NeuralODE, NeuralSDE
from torchsde import sdeint
from torchsde import BrownianInterval

#from Prior import GaussianPrior
import matplotlib.pyplot as plt
from ..utils.utils import to_tensor

def manual_euler_sample(drift, diffusion, n_samples=1, init_state=None, ts=None, bm=None, sde_type=True):
    """
    Samples trajectories using the Euler-Maruyama method.

    Args:
        n_samples (int): Number of samples.
        init_state (torch.Tensor): Initial state (shape: [batch_size, state_dim]).
        ts (torch.Tensor): Time steps (shape: [steps]).
        bm (BrownianInterval): Brownian motion object.

    Returns:
        torch.Tensor: Sampled trajectories (shape: [steps, batch_size, state_dim]).
    """
    batch_size = init_state.shape[0]
    states = torch.zeros((ts.shape[0], *init_state.shape), device=init_state.device) 
    states[0] = init_state
    
    for i in range(len(ts)-1):
        dt = ts[i+1] - ts[i]
        states[i+1] = states[i] + drift(states[i], ts[i].unsqueeze(0).repeat(states.shape[1])) * dt
        if sde_type == True:
            diff = diffusion(states[i], ts[i].unsqueeze(0).repeat(states.shape[1]).to(init_state.device))
            if bm is None:
                # Directly sample from a standard normal distribution
                dW_t = torch.randn((batch_size, diff.shape[2]), device=init_state.device) * dt.abs().sqrt()
            else: 
                if ts[i] < ts[i+1]:
                    dW_t = bm(ts[i], ts[i+1])
                else: 
                    dW_t = -bm(ts[i+1], ts[i])
            #stochastic_part = torch.einsum('Bsb, Bb -> Bs', diff, dW_t)
            stochastic_part = (diff @ dW_t.unsqueeze(-1)).squeeze(-1)  # Assuming diff is of shape [batch_size, state_dim, state_dim]
            states[i+1] += stochastic_part
    return states

class SimpleSDE(nn.Module):
    """
    A simple SDE class for defining drift and diffusion functions.

    Args:
        drift (callable): Drift function f(t, x).
        difussion (callable): Diffusion function g(t, x).
        sde_type (str): Type of SDE ("ito" or "stratonovich").
        noise_type (str): Type of noise ("general" or "diagonal").
        device (str): Device to run the computations on.

    Attributes:
        f (callable): Drift function with inputs t (scalar) and x (tensor of shape [batch_size, state_dim]).
        g (callable): Diffusion function with inputs t (scalar) and x (tensor of shape [batch_size, state_dim]).
    """
    def __init__(self, drift, diffusion, sde_type="ito", noise_type="general", device='cpu'):
        super(SimpleSDE, self).__init__()
        self.device = device
        self.drift = drift
        self.diffusion = diffusion

        self.f = lambda t, x: self.drift(x.to(self.device), t.to(self.device).unsqueeze(0).repeat(x.shape[0])).squeeze(0)
        self.g = lambda t, x: self.diffusion(x.to(self.device), t.to(self.device).unsqueeze(0).repeat(x.shape[0])).squeeze(0)
        self.sde_type = sde_type
        self.noise_type = noise_type


class SDE(nn.Module):
    """
    A class for defining and sampling from an SDE.

    Args:
        drift (callable): Drift function f(x, t).
        diffusion (callable): Diffusion function g(x, t).
        prior (Prior): Prior distribution for initial states.
        ts (torch.Tensor): Time steps (shape: [steps]).
        steps (int): Number of time steps.
        t_start (torch.Tensor): Start time (scalar).
        t_end (torch.Tensor): End time (scalar).
        device (str): Device to run the computations on.

    Methods:
        sample_torchsde: Samples trajectories using torchsde (output shape: [steps, batch_size, state_dim]).
        manual_euler_sample: Samples trajectories using Euler-Maruyama (output shape: [steps, batch_size, state_dim]).
    """
    def __init__(self, drift=None, diffusion=None, prior=None, ts=None, steps=17, t_start=torch.tensor(0.), t_end=torch.tensor(1.), device='cpu'):
        super(SDE, self).__init__()
        self.device = device
        self.drift = drift
        self.diffusion = diffusion
        self.prior = prior
        self.t_start = to_tensor(t_start).to(self.device)
        self.t_end = to_tensor(t_end).to(self.device)
        self.steps = steps
        
        if ts is None:
            self.ts = torch.linspace(self.t_start, self.t_end, steps, device=self.device)
        else:
            self.ts = ts.to(self.device)
        #self.torchdyn_neuralSDE = torchdyn.core.NeuralSDE(drift, diffusion)


    def sample_torchsde(self, n_samples=1, init_state=None, ts=None, bm=None):
        """
        Samples trajectories using torchsde.

        Args:
            n_samples (int): Number of samples.
            init_state (torch.Tensor): Initial state (shape: [batch_size, state_dim]).
            ts (torch.Tensor): Time steps (shape: [steps]).
            bm (BrownianInterval): Brownian motion object.

        Returns:
            torch.Tensor: Sampled trajectories (shape: [steps, batch_size, state_dim]).
        """
        if init_state is None:
            init_state = self.prior.sample(n_samples).to(self.device)
        if ts is None:
            ts = self.ts
        mySDE = SimpleSDE(self.drift, self.diffusion, device=self.device)
        return sdeint(mySDE, init_state, ts.to(self.device), bm=bm)
    
    def manual_euler_sample(self, n_samples=1, init_state=None, ts=None, bm=None, sde_type=True):
        """
        Samples trajectories using the Euler-Maruyama method.

        Args:
            n_samples (int): Number of samples.
            init_state (torch.Tensor): Initial state (shape: [batch_size, state_dim]).
            ts (torch.Tensor): Time steps (shape: [steps]).
            bm (BrownianInterval): Brownian motion object.

        Returns:
            torch.Tensor: Sampled trajectories (shape: [steps, batch_size, state_dim]).
        """
        if init_state is None:
            init_state = self.prior.sample(n_samples).to(self.device)
        if ts is None:
            ts = self.ts
        states = manual_euler_sample(self.drift, self.diffusion, n_samples=n_samples, init_state=init_state, ts=ts, bm=bm, sde_type=sde_type)


        return states
