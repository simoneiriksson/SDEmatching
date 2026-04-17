import torch.nn as nn
from torch.func import vmap, grad
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call

import unittest
import torch
import normflows as nf
import matplotlib.pyplot as plt
import matplotlib

class Marginal(nn.Module):
    """
    Marginal distribution class for conditional normalizing flows.

    Args:
        marginal_func (callable): Marginal function (e.g., nf.ConditionalNormalizingFlow).
        diffusion_term (callable): Function returning diffusion matrix (shape: [state_dim, state_dim]).
        condition_mapper (callable): Maps time-data space to conditional space.
        device (str): Device to run the computations on.

    Methods:
        marginal_dt: Computes time derivative of the marginal function (output shape: [batch_size, state_dim]).
        marginal_inv: Computes inverse of the marginal function (output shape: [batch_size, state_dim]).
        sample: Samples from the marginal distribution (output shape: [n_samples, state_dim]).
        log_prob: Computes log-probability of samples (input shape: [batch_size, state_dim]).
    """

    def __init__(self, marginal_func, diffusion_term, condition_mapper, device='cpu', diff_type="torch.autograd.functional.jvp"):
        """
            condition_mapper is a function that maps from time-data space (t, (ts, Xs)) into conditional-space (C)
                - Xs has dimension: [num_time_samples, num_observations, observation_dim]
                - t has dimension num_time_samples
                - for each t, and corresponding series in (ts, Xs), state_mapper returns a condition
                - so output is of size [num_time_samples, conditional-space]
            
            marginal_func can be a nf.ConditionalNormalizingFlow, but any class is ok as long as it has the required methods
                - marginal_func.sample(num_samples=1, context=C)
                - methods required: 
                  - inverse(state, context=conditions)
                  - __call__(epsilon, context = context) = forward taking an epsilon and returning sample
                  - sample(num_samples=conditions.shape[0], context=conditions): sampling epsilon from base distribution and returning result
                  - forward_and_log_det(z=eps, context=conditions): forward taking an epsilon and returning sample and logprob
                  - log_prob(state, context=conditions) return logprob
                  
                properties required: 
                  - base_dist (being a torch distribution)


            diffusion_term is a function that takes values in conditional-space (C) and returns a matrix of size state_dim
        """
        
        # marginal_func can be a nf.ConditionalNormalizingFlow of the type G(epsilon \vert tX), where tX is the concatenation of t and X (if extended_state=True, else =X). 
        # It has to have an inverse with respect to epsilon
        # assert type(marginal_func) == nf.ConditionalNormalizingFlow
        super(Marginal, self).__init__()
        self.device = device
        self.marginal_func = marginal_func.to(self.device)
        self.diffusion_term = diffusion_term.to(self.device)
        self.condition_mapper = condition_mapper
        self.base_dist = marginal_func.q0
        self.diff_type = diff_type
        if type(self.marginal_func) == nf.ConditionalNormalizingFlow: self.is_normflow = True
        else: self.is_normflow = False

    def marginal_dt_(self, epsilon, t, data, data_mask=None):
        """
        Computes the time derivative of the marginal function.

        Args:
            epsilon (torch.Tensor): Latent variable (shape: [batch_size, latent_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Time derivative (shape: [batch_size, state_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        # get the partial derivative dG(t, X, epsilon)/dt 
        # we make a functional forward 
        def marginal_func_wrapped(t, data, epsilon):
            context = self.condition_mapper(t.unsqueeze(0), data.unsqueeze(0), data_mask.unsqueeze(0))
            return self.marginal_func(epsilon.unsqueeze(0), context=context).squeeze(0)
        dG_dt_fn = torch.func.jacfwd(marginal_func_wrapped) # take gradient wrt first parameter (t)
        dG_dt_vmap = vmap(dG_dt_fn) # run it 
        if t.device.type == "mps":
            dG_dt = dG_dt_vmap(t.to("cpu"), data.to("cpu"), data_mask.to("cpu") , epsilon.to("cpu")).to(self.device)
        else:
            dG_dt = dG_dt_vmap(t.to(self.device), data.to(self.device), data_mask.to(self.device), epsilon.to(self.device))
        return dG_dt


    def marginal_dt(self, epsilon, t, data, data_mask=None):
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        # all three gives the same result. I should test which one performs best.
        #return self.marginal_dt_vmap(epsilon, t, data)
        if self.diff_type == "vmap_func":
            return self.marginal_dt_vmap(epsilon, t, data, data_mask)
        elif self.diff_type == "torch.autograd.functional.jvp":
            return self.marginal_dt_autograd_functional_jvp(epsilon, t, data, data_mask)
        elif self.diff_type == "torch.autograd.functional.jvp2":
            return self.marginal_dt_autograd_functional_jvp2(epsilon, t, data, data_mask)
        else: print("ERROR")

    def marginal_dt_autograd_functional_jvp(self, epsilon, t, data, data_mask=None):
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        def marginal_func_wrapped(tt):
            with torch.enable_grad():
                context = self.condition_mapper(tt, data, data_mask)
                return self.marginal_func(epsilon, context=context)
        func_out, jvp = torch.autograd.functional.jvp(func=marginal_func_wrapped, inputs=t, v=torch.ones_like(t), create_graph=True)
        return jvp

    def marginal_dt_autograd_functional_jvp2(self, epsilon, t, data, data_mask=None):
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)

        def marginal_func_wrapped_2(tt, data, epsilon):
            with torch.enable_grad():
                context = self.condition_mapper(tt, data)
                return self.marginal_func(epsilon, context=context)

        func_out, jvp = torch.func.jvp(func=marginal_func_wrapped_2, 
                                primals=(t, data, data_mask, epsilon), 
                                tangents=(torch.ones_like(t), torch.zeros_like(data), torch.zeros_like(data_mask), torch.zeros_like(epsilon)), create_graph=True)
        return jvp


    def marginal_dt_vmap(self, epsilon, t, data, data_mask=None):
        """
        Computes the time derivative of the marginal function.

        Args:
            epsilon (torch.Tensor): Latent variable (shape: [batch_size, latent_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Time derivative (shape: [batch_size, state_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)

        # get the partial derivative dG(t, X, epsilon)/dt 
        # we make a functional forward 
        def marginal_func_wrapped(t, data, data_mask, epsilon):
            context = self.condition_mapper(t.unsqueeze(0), data.unsqueeze(0), data_mask.unsqueeze(0))
            return self.marginal_func(epsilon.unsqueeze(0), context=context).squeeze(0)
        dG_dt_fn = torch.func.jacfwd(marginal_func_wrapped) # take gradient wrt first parameter (t)
        dG_dt_vmap = vmap(dG_dt_fn) # run it 
        if t.device.type == "mps":
            dG_dt = dG_dt_vmap(t.to("cpu"), data.to("cpu"), data_mask.to("cpu"),  epsilon.to("cpu")).to(self.device)
        else:
            dG_dt = dG_dt_vmap(t.to(self.device), data.to(self.device),data_mask.to(self.device), epsilon.to(self.device))
        return dG_dt


    def marginal_inv(self, state, t, data, data_mask=None):
        """
        Computes the inverse of the marginal function.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Inverse (shape: [batch_size, latent_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)

        state, t, data, data_mask = state.to(self.device), t.to(self.device), data.to(self.device), data_mask.to(self.device)
        conditions = self.condition_mapper(t, data, data_mask)
        return self.marginal_func.inverse(state, context=conditions)

    def marginal_inv_and_log_prob(self, state, t, data, data_mask=None):
        """
        Computes the inverse of the marginal function and its log probability.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            tuple: 
                - torch.Tensor: Inverse (shape: [batch_size, latent_dim]).
                - torch.Tensor: Log probability (shape: [batch_size]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        conditions = self.condition_mapper(t, data, data_mask)
        eps = self.marginal_func.inverse(state, context=conditions)
        return eps, self.base_dist.log_prob(eps)

    def sample(self, t, data, data_mask=None, n_samples=1):
        """
        Samples from the marginal distribution.

        Args:
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Sampled tensor (shape: [n_samples, state_dim]).
        """
        assert n_samples == 1, "Multiple samples are not yet implemented"
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        conditions = self.condition_mapper(t, data, data_mask)
        t, data, data_mask = t.to(self.device), data.to(self.device), data_mask.to(self.device)
        return self.marginal_func.sample(num_samples=conditions.shape[0], context=conditions)[0]

    def sample_and_log_prob(self, t, data, data_mask=None, n_samples=1):
        """
        Samples from the marginal distribution and computes log probability.

        Args:
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).
            n_samples (int): Number of samples to generate.

        Returns:
            tuple:
                - torch.Tensor: Sampled tensor (shape: [n_samples, state_dim]).
                - torch.Tensor: Log probability (shape: [n_samples]).
        """
        assert n_samples == 1, "Multiple samples are not yet implemented"
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        conditions = self.condition_mapper(t, data, data_mask)
        t, data, data_mask = t.to(self.device), data.to(self.device), data_mask.to(self.device)
        return self.marginal_func.sample(num_samples=conditions.shape[0], context=conditions)

    def forward_and_log_prob(self, eps, t, data, data_mask=None):
        """
        Performs the forward transformation and computes log probability.

        Args:
            eps (torch.Tensor): Latent variable (shape: [batch_size, latent_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            tuple:
                - torch.Tensor: Transformed tensor (shape: [batch_size, state_dim]).
                - torch.Tensor: Log probability (shape: [batch_size]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        eps, t, data, data_mask = eps.to(self.device), t.to(self.device), data.to(self.device), data_mask.to(self.device)
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        conditions = self.condition_mapper(t, data, data_mask)
        sample, log_det = self.marginal_func.forward_and_log_det(z=eps, context=conditions)
        return sample, log_det + self.base_dist.log_prob(eps)

    def log_prob(self, state, t, data, data_mask=None):
        """
        Computes the log probability of the state.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Log probability (shape: [batch_size]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        state, t, data, data_mask = state.to(self.device), t.to(self.device), data.to(self.device), data_mask.to(self.device)
        conditions = self.condition_mapper(t, data, data_mask)
        return self.marginal_func.log_prob(state, context=conditions)

    def ODEdrift(self, state, t, data, data_mask=None):
        """
        Computes the drift term for the ODE.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Drift term (shape: [batch_size, state_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        state, t, data, data_mask = state.to(self.device), t.to(self.device), data.to(self.device), data_mask.to(self.device)
        eps = self.marginal_inv(state, t, data, data_mask)
        marginal_dt = self.marginal_dt(eps, t, data, data_mask)
        return marginal_dt
    

    def log_prob_grad(self, epsilon, t, data, data_mask=None):
        # all three gives the same result. I should test which one performs best.
        #return self.marginal_dt_vmap(epsilon, t, data)
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        if self.diff_type == "vmap_func":
            return self.log_prob_grad_vmap(epsilon, t, data, data_mask)
        elif self.diff_type == "torch.autograd.functional.jvp":
            return self.log_prob_grad_autograd_functional_jvp(epsilon, t, data, data_mask)
        else: print("ERROR")

    def log_prob_grad_vmap(self, state, t, data, data_mask=None):
        """
        Computes the gradient of the log probability with respect to the state.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Gradient of log probability (shape: [batch_size, state_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        def log_prob_func(state, t, data, data_mask): 
            return self.log_prob(state.unsqueeze(0), t.unsqueeze(0), data.unsqueeze(0), data_mask.unsqueeze(0)).squeeze(0)
        if self.is_normflow:
            # here we cannot use vmap because normflows uses some inplace arithmetics at some point. And that does not work together.
            log_prob_grad_fn = torch.func.jacrev(log_prob_func) # function that returns gradient of log_prob wrt first parameter (state)
            for i in range(len(state)):
                log_prob_grad_ = log_prob_grad_fn(state[i], t[i], data[i], data_mask[i]) # shape (state_dim)
                if i == 0:
                    log_prob_grad_res = torch.zeros([len(state), *log_prob_grad_.shape], device=self.device)
                log_prob_grad_res[i] = log_prob_grad_
            #log_prob_grad_res = log_prob_grad_res[:,0,:]
        else: 
            log_prob_grad_fn = torch.func.jacrev(log_prob_func) # function that returns gradient of log_prob wrt first parameter (state)
            log_prob_grad_res = vmap(log_prob_grad_fn)(state, t, data, data_mask)
        return log_prob_grad_res

    def log_prob_grad_autograd_functional_jvp(self, state, t, data, data_mask):
        def log_prob_func(state): 
            return self.log_prob(state, t, data, data_mask)
        func_out, vjp = torch.autograd.functional.vjp(func=log_prob_func, inputs=state, v=torch.ones(state.shape[0], device=self.device), create_graph=True)
        return vjp


    def SDEbackdrift(self, state, t, data, data_mask=None):
        """
        Computes the backward drift term for the SDE.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Backward drift term (shape: [batch_size, state_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        state, t, data, data_mask = state.to(self.device), t.to(self.device), data.to(self.device), data_mask.to(self.device)
        ODEdrift = self.ODEdrift(state, t, data, data_mask)
        sigma_squared = self.diffusion_term.sigma_squared(state, t)
        sigma_squared_grad = self.diffusion_term.sigma_squared_grad_sum(state, t)
        log_prob_grad = self.log_prob_grad(state, t, data, data_mask)
        SDE_backdrift = ODEdrift - 0.5 * torch.einsum("bkk, bk -> bk", sigma_squared, log_prob_grad) - 0.5 * sigma_squared_grad
        return SDE_backdrift

    def SDEforwarddrift(self, state, t, data, data_mask=None):
        """
        Computes the forward drift term for the SDE.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).
            data (torch.Tensor): Data (shape: [num_time_samples, num_observations, observation_dim]).

        Returns:
            torch.Tensor: Forward drift term (shape: [batch_size, state_dim]).
        """
        if data_mask is None: data_mask = torch.zeros(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)
        state, t, data, data_mask = state.to(self.device), t.to(self.device), data.to(self.device), data_mask.to(self.device)
        ODEdrift = self.ODEdrift(state, t, data, data_mask)
        sigma_squared = self.diffusion_term.sigma_squared(state, t)
        sigma_squared_grad = self.diffusion_term.sigma_squared_grad_sum(state, t)
        log_prob_grad = self.log_prob_grad(state, t, data, data_mask)
        SDEforwarddrift = ODEdrift + 0.5 * torch.einsum("bkk, bk -> bk", sigma_squared, log_prob_grad) + 0.5 * sigma_squared_grad
        return SDEforwarddrift

