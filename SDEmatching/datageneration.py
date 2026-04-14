import torch
# import torch.nn as nn
# import normflows as nf
# from torchdyn.core import NeuralODE, NeuralSDE
# from torchsde import sdeint
# from torchsde import BrownianInterval

# from Prior import GaussianPrior
# from SDE import SimpleDiffusion, SimpleDrift, SimpleSDE, SDE
# from Emission import Emission, DistEmission, GaussianEmission, NFEmission
import matplotlib.pyplot as plt
from SDEMatching.utils import torch_seed




def SDEdatagenerator(SDE, emission_dist, time_sampler, num_series=7, mean_num_ts=10, same_ts=False, 
                     num_ts_samples=None, device='cpu', seed=2):
    """
    A function that takes:
    1) a drift function
    2) a diffusion function
    3) a prior
    4) an emission distribution
    5) time sampling distributioun

    And returns a list of time series, where each time series is of the form [ts, X]. 
    """  
    SDE.to(device)
    original_ts = SDE.ts.to(device)
    list_of_emission_timeseries = []
    list_of_state_timeseries = []

    with torch_seed(seed):
        if same_ts:
            if num_ts_samples is None:
                num_samples = int(torch.distributions.Poisson(mean_num_ts).sample().item())
            else:
                num_samples = num_ts_samples
            emission_ts = time_sampler.sample((num_samples,)).to(device).sort()[0]  # sample timesteps for emissions. These can be different from the timesteps in SDE.ts

        for i in range(num_series):
            if not same_ts:
                if num_ts_samples is None:
                    num_samples = int(torch.distributions.Poisson(mean_num_ts).sample().item())
                else:
                    num_samples = num_ts_samples
                emission_ts = time_sampler.sample((num_samples,)).to(device).sort()[0]
            ts, t_order = torch.cat([original_ts, emission_ts]).sort()  # add timesteps where the emissions should be sampled
            SDE.ts = ts
            state_samples = SDE.manual_euler_sample(1).to(device)[:, 0, :]  # simulate the SDE
            list_of_state_timeseries.append(state_samples[(t_order < original_ts.shape[0]).nonzero()[:, 0]]) # filter out the timesteps which corresponds to the original SDE.ts
            state_samples_ = state_samples[(t_order >= original_ts.shape[0]).nonzero()[:, 0]] # filter out only the time steps where the emissions are sampled
            emissions = emission_dist.sample(state_samples_, n_samples=1)[0, :, :].to(device)
            list_of_emission_timeseries.append(torch.column_stack([emission_ts.unsqueeze(1), emissions]))
        SDE.ts = original_ts
    return list_of_emission_timeseries, list_of_state_timeseries

