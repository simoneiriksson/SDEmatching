import torch
import torch.nn as nn
from SDEmatching.utils.utils import torch_seed

class FunctionApproximatorModel(torch.nn.Module):
    """
    A feedforward neural network for function approximation.

    Args:
        num_features (int): Number of input features.
        hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
        num_outputs (int): Number of output features.
        nonlin (torch.nn.Module): Non-linearity to apply between layers.
        seed (int): Random seed for weight initialization.
        initial_zero (bool): Whether to initialize weights and biases to zero.
        device (str): Device to run the computations on.

    Methods:
        forward: Forward pass through the network.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features).

    Output:
        torch.Tensor: Output tensor of shape (batch_size, num_outputs).
    """
    def __init__(self, num_features=1, hidden_layers=[10], num_outputs=1, nonlin=torch.nn.ReLU(), seed=2, initial_zero=False, device='cpu', bias=True):
        super().__init__()
        self.device = device
        self.num_features = num_features
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.nonlin = nonlin
        self.initial_zero = initial_zero
        self.seed = seed
        self.device = device
        self.bias=bias

        if self.num_features == 0:
            self.num_features = 1
            self.input_zero_dim = True
        else:
            self.input_zero_dim = False
        
        if self.num_outputs == 0:
            self.num_outputs = 1
            self.output_zero_dim = True
        else:
            self.output_zero_dim = False
        
        with torch_seed(seed):
            self.layers = [self.num_features] + self.hidden_layers + [self.num_outputs]
            self.hidden_layers = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1], bias=self.bias).to(self.device) for i in range(len(self.layers)-1)])
            self.nonlin = nonlin
            for layer in self.hidden_layers:
                if initial_zero:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight, 0.0)
                else:
                    torch.nn.init.kaiming_normal_(layer.weight)
                #torch.nn.init.constant_(layer.bias, 0.0)
        
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        x = x.to(self.device)
        if self.input_zero_dim:
            if x.dim() == 1: # if we are in the case that the input dimension is 0 and we are batched, 
                x = x.unsqueeze(-1) # add a batch dimension
            elif x.dim() == 0: # if we are in the case that the input dimension is 0 and we are not batched,
                x = x.unsqueeze(0).unsqueeze(-1) # add a batch dimensionx
        
        for layer in self.hidden_layers[:-1]:
            x = self.nonlin(layer(x))
        x = self.hidden_layers[-1](x)
        if self.output_zero_dim:
            if x.dim() == 2:  # if we are in the case that the output dimension is 1 and we are batched, 
                x = x.squeeze(-1) # remove the data dimension
            elif x.dim() == 1:  # if we are in the case that the output dimension is 1 and we are not batched,
                x = x.squeeze(-1)  # remove the data dimension
        return x
    

class FunctionApproximatorModel_drift(FunctionApproximatorModel):
    """
    A specialized function approximator for drift functions.

    Args:
        num_features (int): Number of input features (excluding time).
        hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
        num_outputs (int): Number of output features.
        nonlin (torch.nn.Module): Non-linearity to apply between layers.
        seed (int): Random seed for weight initialization.
        initial_zero (bool): Whether to initialize weights and biases to zero.
        device (str): Device to run the computations on.

    Methods:
        forward: Forward pass through the network.

    Input:
        state (torch.Tensor): State tensor of shape (batch_size, state_dim).
        ts (torch.Tensor): Time tensor of shape (batch_size).

    Output:
        torch.Tensor: Output tensor of shape (batch_size, num_outputs).
    """
    def __init__(self, num_features=1, hidden_layers=[10], num_outputs=1, nonlin=torch.nn.ReLU(), seed=2, initial_zero=False, device='cpu', time_independent=False, bias=False):
        if not time_independent: num_features += 1
        super().__init__(num_features=num_features, hidden_layers=hidden_layers, num_outputs=num_outputs, nonlin=nonlin, seed=seed, initial_zero=initial_zero, device=device, bias=bias)
        self.time_independent = time_independent

    def forward(self, state, ts):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).
            ts (torch.Tensor): Time tensor of shape (batch_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        state, ts = state.to(self.device), ts.to(self.device)
        if self.time_independent:
            x_ = state
        else:
            x_ = torch.cat([ts.unsqueeze(1), state], dim=1)
        return super().forward(x_)
    

class FunctionApproximatorModel_diffusion(FunctionApproximatorModel):
    """
    A specialized function approximator for diffusion functions.

    Args:
        num_features (int): Number of input features (excluding time).
        hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
        num_outputs (int): Number of output features.
        nonlin (torch.nn.Module): Non-linearity to apply between layers.
        seed (int): Random seed for weight initialization.
        initial_zero (bool): Whether to initialize weights and biases to zero.
        device (str): Device to run the computations on.

    Methods:
        forward: Forward pass through the network.

    Input:
        state (torch.Tensor): State tensor of shape (batch_size, state_dim).
        ts (torch.Tensor): Time tensor of shape (batch_size).

    Output:
        torch.Tensor: Output tensor of shape (batch_size, num_outputs).
    """
    def __init__(self, num_features=1, hidden_layers=[10], num_outputs=1, nonlin=torch.nn.ReLU(), seed=2, initial_zero=False, device='cpu', time_independent=False, diagonal=False, bias=False):
        self.diagonal = diagonal
        self.time_independent = time_independent
        self.num_outputs = num_outputs
        if not time_independent: num_features += 1
        if diagonal == True: num_outputs=num_outputs**2
        super().__init__(num_features=num_features, hidden_layers=hidden_layers, num_outputs=num_outputs**2, nonlin=nonlin, seed=seed, initial_zero=initial_zero, device=device, bias=bias)

    def forward(self, state, ts):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).
            ts (torch.Tensor): Time tensor of shape (batch_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        state, ts = state.to(self.device), ts.to(self.device)
        if self.time_independent:
            x_ = state
        else:
            x_ = torch.cat([ts.unsqueeze(1), state], dim=1)
        out = super().forward(x_)
        if self.diagonal:
            return torch.diag(out)
        else: 
            return out.reshape(ts.shape[0], self.num_outputs, self.num_outputs)

class FunctionApproximatorModel_condition_mapper(FunctionApproximatorModel):
    """
    A specialized function approximator for mapping conditions.

    Methods:
        forward: Forward pass through the network.

    Input:
        t (torch.Tensor): Time tensor of shape (batch_size).
        x (torch.Tensor): Data tensor of shape (batch_size, num_observations, observation_dim).
        ts (torch.Tensor): Time series tensor of shape (batch_size, num_time_samples).

    Output:
        torch.Tensor: Output tensor of shape (batch_size, num_outputs).
    """
    def forward(self, t, x, ts):
        """
        Forward pass through the network.

        Args:
            t (torch.Tensor): Time tensor of shape (batch_size).
            x (torch.Tensor): Data tensor of shape (batch_size, num_observations, observation_dim).
            ts (torch.Tensor): Time series tensor of shape (batch_size, num_time_samples).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        t, x, ts = t.to(self.device), x.to(self.device), ts.to(self.device)
        x_ = torch.cat([t.unsqueeze(1), ts[:, 0].unsqueeze(1), x[:, 0, :]], dim=1)
        return super().forward(x_)