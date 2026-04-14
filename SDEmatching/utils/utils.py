
import torch
from contextlib import contextmanager

@contextmanager
def torch_seed(seed):
    """
    A context manager to temporarily set the random seed in PyTorch.
    
    Args:
        seed (int): The seed value to use within the context.
    """
    # Save the current random state
    random_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        # Restore the previous random state
        torch.set_rng_state(random_state)

def to_tensor(data):
    """
    Converts the input to a torch tensor with the same values.
    """
    return torch.tensor(data) if not isinstance(data, torch.Tensor) else data
