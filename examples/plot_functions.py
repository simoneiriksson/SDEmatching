import torch
import matplotlib
from matplotlib import pyplot as plt

def plot_parameter_history(true_parameter_dict, saved_parameter_dict, step_no):
    cmap = matplotlib.colormaps["viridis"]
    
    fig, axes = plt.subplots(1, len(true_parameter_dict), figsize=(5*len(true_parameter_dict), 5), squeeze=False)
    for ax_no, (key, val_tensor) in enumerate(saved_parameter_dict.items()):
        #print(key)
        for i in range(val_tensor.shape[1]):
            #print(f"{key}, {i}", {val_tensor[:,i]})
            color=cmap(i/val_tensor.shape[1])
            axes[0, ax_no].plot(val_tensor[:,i].detach().cpu(), color=color)
        axes[0, ax_no].set_title(f"{key}")

    for ax_no, (key, val_tensor) in enumerate(true_parameter_dict.items()):
        #print(key)
        #print(val_tensor)
        for i in range(val_tensor.shape[1]):
            #print(f"{key}, {i}", {val_tensor[:,i]})
            color=cmap(i/val_tensor.shape[1])
            axes[0, ax_no].hlines(val_tensor[:,i].detach().cpu(), xmin=0, xmax=step_no, color=color)
    return fig, axes

def plot_marginal(sdeproblem, data, state_dim, axes=None, fig=None, true_states=None, true_states_ts=None, num_timesteps=30,
                  num_samples=100, max_num_data=10, device="cpu", timeseries_separate_plots=True, observation_dim=None, data_mask=None):
    with torch.no_grad():
        eps1 = sdeproblem.marginal.base_dist.sample((num_samples,))
        if observation_dim is None:
            observation_dim = state_dim
        num_data = min(data.shape[0], max_num_data)
        series_length = data.shape[1]
        ts_plot = torch.linspace(true_states_ts.min(), true_states_ts.max(), num_timesteps, device=device)

        # extending the timestep tensor, data tensor and eps tensor  
        ts_plt_extended = ts_plot[:, None, None].repeat([1, num_samples, num_data])
        ts_plt_extended.shape # num_timesteps, num_samples, num_data
        ts_plt_extended_flat = ts_plt_extended.reshape([num_timesteps* num_samples* num_data])
        ts_plt_extended_flat.shape

        data_extended = data[None, None, :num_data, ...].repeat([num_timesteps, num_samples] + [1] * data.dim()).to(device)
        data_extended.shape  # num_timesteps, num_samples, num_data, series_length, observation_dim+1
        data_extended_flat = data_extended.reshape([num_timesteps * num_samples * num_data, series_length, observation_dim+1])
        data_extended_flat.shape

        eps1_extended = eps1[None, :, None, :].repeat([num_timesteps, 1, num_data, 1]).to(device)
        eps1_extended.shape # num_timesteps, num_samples, num_data, statedim
        eps1_extended_flat = eps1_extended.reshape([num_timesteps * num_samples * num_data , state_dim])
        eps1_extended_flat.shape

        marginal_samples, logprob = sdeproblem.marginal.forward_and_log_prob(eps1_extended_flat, ts_plt_extended_flat, data_extended_flat)
        marginal_samples.shape
        marginal_samples_reshaped = marginal_samples.reshape([num_timesteps, num_samples, num_data, state_dim])
        marginal_samples_reshaped.shape

        # now for the SDE simulation    
        starting_points = true_states[:num_data,0,:] # we depart from the true state of the SDE. Using only first num_data
        simulation_samples = sdeproblem.SDE.manual_euler_sample(1, init_state=starting_points)


        cmap = matplotlib.colormaps["viridis"]
        marginal_samples_mean = marginal_samples_reshaped[:, :, :, :].mean(dim=1).cpu().detach()
        marginal_samples_std = marginal_samples_reshaped[:, :, :, :].std(dim=1).cpu().detach()
        if axes==None: 
            if timeseries_separate_plots:
                fig, axes=plt.subplots(state_dim, num_data, figsize=(10,5), sharey=True)
            else:
                fig, axes=plt.subplots(1, state_dim, figsize=(10,5))
        for dimension in range(state_dim):
            for i in range(num_data):
                if timeseries_separate_plots:
                    this_ax = axes[dimension][i]
                else:
                    this_ax = axes[dimension]
                color = cmap(i/float(num_data))

                this_ax.plot(ts_plot.detach().cpu(), marginal_samples_mean[:,i,dimension].detach().cpu(), alpha=.1, c=color)  # plot mean marginal distribution
                this_ax.fill_between(ts_plot.detach().cpu(), (marginal_samples_mean[:,i,dimension] - marginal_samples_std[:,i,dimension]*1.94).detach().cpu(), 
                                    (marginal_samples_mean[:,i,dimension] + marginal_samples_std[:,i,dimension]*1.94).detach().cpu(), alpha=.5, color=color)  # plot std bands for marginal distribution
                if dimension < observation_dim:
                    this_ax.plot(data[i,~data_mask[i],0].detach().cpu(), data[i,~data_mask[i],dimension+1].detach().cpu(), color=color, linewidth=0, marker="x") # plot the sampled data poitns
                this_ax.plot(true_states_ts.detach().cpu(), true_states[i,:,dimension].T.detach().cpu(), marker="", c=color) # plot the true underlying SDE
                #this_ax.plot(sdeproblem.SDE.ts.detach().cpu(), simulation_samples[:,i, dimension].detach().cpu(), c=color, linestyle=":")
            #this_ax.set_title(f"samples from marginal distribution, {dimension} dimension")
        
    return fig, axes