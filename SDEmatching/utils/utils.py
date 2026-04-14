
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


def mask_and_pad(list_of_timeseries, list_of_state_timeseries, state_dim, device):
    num_series = len(list_of_timeseries)
    emissions_longest_len = max([len(timeseries) for timeseries in list_of_timeseries])
    emissions_padded = torch.zeros(num_series, emissions_longest_len, state_dim+1, device=device)
    emissions_mask = torch.ones(num_series, emissions_longest_len, dtype=torch.bool, device=device)  # True = padding

    for i, ts in enumerate(list_of_timeseries):
        #print(f"Series {i} length: {len(ts)}")
        n = len(ts)
        emissions_padded[i, :n] = ts
        emissions_mask[i, :n] = False  # real observations

    data = emissions_padded.detach().clone()
    return data, emissions_mask