import torch
import torch.nn as nn
from SDEMatching.utils import to_tensor

class Diffusion(nn.Module):
    """
    Base class for diffusion models.

    Args:
        device (str): Device to run the computations on.

    Methods:
        forward: Computes the diffusion matrix (output shape: [batch_size, state_dim, brownian_dim]).
        sigma_squared: Computes the squared diffusion matrix (output shape: [batch_size, state_dim, state_dim]).
        sigma_squared_grad_sum: Computes the gradient sum of the squared diffusion matrix (output shape: [batch_size, state_dim]).
    """
    def __init__(self, device='cpu', diff_type="torch.autograd.functional.jvp"):
        super().__init__()
        self.device = device
        self.diff_type = diff_type

    def forward(self, t, x):
        raise NotImplementedError("Diffusion class must implement the forward method.")
    
    # @property
    # def diffusiontype():
    #     raise NotImplementedError("Diffusion class must implement the diffusiontype property.")

    def sigma_squared_grad_sum_vmap(self, state, t):
        def sigma_squared_func(state, t): 
            sigma = self.forward(state.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
            return sigma @ sigma.T
        sigma_squared_grad_fn = torch.func.jacrev(sigma_squared_func)  # function that returns gradient wrt first parameter (state). Shape is [sigma_1, sigma_2, state]
        # That is, the last dimension holds the index of the state vector, with which we differentiate.
        if state.device.type == "mps":
            sigma_squared_grad_state = torch.vmap(sigma_squared_grad_fn)(state.to("cpu"), t.to("cpu")).to(self.device) # evaluate gradient wrt first parameter (state) at (state, t)
        else:
            sigma_squared_grad_state = torch.vmap(sigma_squared_grad_fn)(state.to(self.device), t.to(self.device)) # evaluate gradient wrt first parameter (state) at (state, t)
        # sigma_squared_grad_state has dimensions [num_samples, statedim, statedim, statedim]
        sigma_squared_grad = sigma_squared_grad_state.sum(dim=[2,3])
        return sigma_squared_grad
    
    def sigma_squared_grad_sum_autograd_functional_jvp(self, state, t):
        def sigma_squared_func(state_state):
            with torch.enable_grad():
                sigma = self.forward(state_state, t)
                return torch.einsum("bkm, bcm -> bkc", sigma, sigma)
        func_out, jvp = torch.autograd.functional.jvp(func=sigma_squared_func, 
                                                      inputs=state, 
                                                      v=torch.ones_like(state), 
                                                      create_graph=True)
        return jvp.sum(dim=1)

    def sigma_squared_grad_sum_autograd_functional_jvp2(self, state, t):
        def sigma_squared_func(state_state, tt):
            with torch.enable_grad():
                sigma = self.forward(state_state, tt)
                return torch.einsum("bij, bkj -> bik", sigma, sigma)
        func_out, jvp = torch.func.jvp(func=sigma_squared_func, 
                                primals=(state, t), 
                                tangents=(torch.ones_like(state), torch.zeros_like(t)), create_graph=True)
        return jvp.sum(dim=1)

    def sigma_squared_grad_sum(self, state, t):
        """
        Computes the gradient sum of the squared diffusion matrix.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).

        Returns:
            torch.Tensor: Gradient sum (shape: [batch_size, state_dim]).
        """
        if self.diffusiontype == "matrix":
            if self.diff_type == "vmap_func":
                return self.sigma_squared_grad_sum_vmap(state, t)
            elif self.diff_type == "torch.autograd.functional.jvp":
                return self.sigma_squared_grad_sum_autograd_functional_jvp(state, t)
            elif self.diff_type == "torch.autograd.functional.jvp2":
                return self.sigma_squared_grad_sum_autograd_functional_jvp2(state, t)
            else: AssertionError(f"AAARGH! Wrong {self.diff_type = }")

    def sigma_squared(self, state, t):
        """
        Computes the squared diffusion matrix.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).

        Returns:
            torch.Tensor: Squared diffusion matrix (shape: [batch_size, state_dim, state_dim]).
        """
        if self.diffusiontype == "matrix":
            sigma = self.forward(state.to(self.device), t.to(self.device))
            sigma_squared = torch.einsum("bkm, bcm -> bkc", sigma, sigma) # sigma[0] @ sigma[0].T
            return sigma_squared


# A very simple diffusion class, with fixed homoscedastic difussion
class SimpleDiffusion(Diffusion):
    """
    Simple diffusion class with fixed homoscedastic diffusion.

    Args:
        brownian_dim (int): Dimensionality of the Brownian motion.
        state_dim (int): Dimensionality of the state space.
        log_std (torch.Tensor): Log standard deviation (scalar).
        device (str): Device to run the computations on.

    Methods:
        forward: Computes the diffusion matrix (output shape: [batch_size, state_dim, brownian_dim]).
    """
    def __init__(self, brownian_dim, state_dim, log_std=0.0, device='cpu', trainable=False, diff_type="torch.autograd.functional.jvp"):
        super().__init__(device=device, diff_type=diff_type)
        self.brownian_dim = brownian_dim
        self.state_dim = state_dim
        self.trainable = trainable
        if trainable:
            self.log_std = torch.nn.Parameter(to_tensor(log_std).to(device))
        else:
            self.log_std = to_tensor(log_std).to(device)
        

        if state_dim < brownian_dim:
            self.matrix = torch.column_stack([torch.eye(state_dim), torch.zeros((state_dim, brownian_dim-state_dim))])
        else:
            self.matrix = torch.row_stack([torch.eye(brownian_dim), torch.zeros((state_dim-brownian_dim, brownian_dim))])
        self.diffusiontype = "matrix"
        self.matrix = self.matrix.to(self.device)
        self.log_std = self.log_std.to(self.device)

    def forward(self, state, t):
        """
        Computes the diffusion matrix.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).

            Returns:
            torch.Tensor: Diffusion matrix (shape: [batch_size, state_dim, brownian_dim]).
        """
        return self.matrix.unsqueeze(0).expand(state.shape[0], *self.matrix.shape).to(self.device) * self.log_std.exp()
    

# A very simple diffusion class, with time-controleld difussion
class ScalarDiffusion(Diffusion):
    """
    Diffusion model with time-controlled diffusion.

    Args:
        brownian_dim (int): Dimensionality of the Brownian motion.
        state_dim (int): Dimensionality of the state space.
        std_func (callable): Function returning the standard deviation as a function of time.
        device (str): Device to run the computations on.

    Methods:
        forward: Computes the diffusion matrix (output shape: [batch_size, state_dim, brownian_dim]).
    """
    def __init__(self, brownian_dim, state_dim, log_std_func, device='cpu', diff_type="torch.autograd.functional.jvp"):
        super().__init__(device=device, diff_type=diff_type)
        self.log_std_func = log_std_func
        self.brownian_dim = brownian_dim
        self.state_dim = state_dim
        self.diffusiontype = "matrix"
        if self.state_dim < self.brownian_dim:
            assert 0==1, "state_dim < brownian_dim Not implemented"
            #self.matrix = torch.column_stack([torch.eye(self.state_dim)*std, torch.zeros((self.state_dim, self.brownian_dim-self.state_dim))])
        else:
            if state_dim < brownian_dim:
                self.matrix = torch.column_stack([torch.eye(state_dim), torch.zeros((state_dim, brownian_dim-state_dim))])
            else:
                self.matrix = torch.row_stack([torch.eye(brownian_dim), torch.zeros((state_dim - brownian_dim, brownian_dim))])
        self.matrix = self.matrix.to(self.device)

    def forward(self, state, t):
        """
        Computes the diffusion matrix.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).

        Returns:
            torch.Tensor: Diffusion matrix (shape: [batch_size, state_dim, brownian_dim]).
        """
        std = self.log_std_func(t.to(self.device)).exp()
        return self.matrix.unsqueeze(0).expand(state.shape[0], *self.matrix.shape).to(self.device) * std[:, None, None]
    


# A fully neuural network diffusion class
class NNDiffusion(Diffusion):
    """
    Diffusion model with time-controlled diffusion.

    Args:
        brownian_dim (int): Dimensionality of the Brownian motion.
        state_dim (int): Dimensionality of the state space.
        std_func (callable): Function returning the standard deviation as a function of time.
        device (str): Device to run the computations on.

    Methods:
        forward: Computes the diffusion matrix (output shape: [batch_size, state_dim, brownian_dim]).
    """
    def __init__(self, brownian_dim, state_dim, net, device='cpu', diff_type="torch.autograd.functional.jvp"):
        super().__init__(device=device)
        self.net = net
        self.brownian_dim = brownian_dim
        self.state_dim = state_dim
        self.diffusiontype = "matrix"

    def forward(self, state, t):
        """
        Computes the diffusion matrix.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).

        Returns:
            torch.Tensor: Diffusion matrix (shape: [batch_size, state_dim, brownian_dim]).
        """
        return self.net(state, t).reshape(t.shape[0], self.state_dim, self.state_dim)
    


class FunkyDiffusion(Diffusion):
    """
    Diffusion model with a custom diffusion matrix.

    Args:
        dim (int): Dimensionality of the state space.
        device (str): Device to run the computations on.

    Methods:
        forward: Computes the diffusion matrix (output shape: [batch_size, state_dim, state_dim]).
    """
    def __init__(self, dim, device='cpu'):
        super().__init__(device=device)
        self.dim = dim
        self.diffusiontype = "matrix"

    def forward(self, state, t):
        """
        Computes the diffusion matrix.

        Args:
            state (torch.Tensor): State variable (shape: [batch_size, state_dim]).
            t (torch.Tensor): Time steps (shape: [batch_size]).

        Returns:
            torch.Tensor: Diffusion matrix (shape: [batch_size, state_dim, state_dim]).
        """
        state = state.to(self.device)
        t = t.to(self.device)
        return torch.diag_embed(state) + state[:, :, None].repeat([1, 1, 2])