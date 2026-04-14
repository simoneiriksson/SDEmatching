from SDEmatching.core.SDE import SDE
from SDEmatching.core.Marginal import Marginal
import torch


# SDE problem class
class SDEproblem(torch.nn.Module):
    def __init__(self, drift=None, diffusion=None, prior=None, marginal_func=None, condition_mapper=None, emission=None, time_dist=None, t_start=None, t_end=None, device=None):
        super(SDEproblem, self).__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.prior = prior
        self.marginal_func = marginal_func
        self.condition_mapper = condition_mapper
        self.emission = emission
        self.time_dist = time_dist
        self.t_start = t_start
        self.t_end = t_end
        self.device = device
        self.marginal = Marginal(self.marginal_func, self.diffusion, self.condition_mapper, device=device)
        self.SDE = SDE(drift, diffusion, prior, steps=100, t_start=t_start, t_end=t_end)
    
    def ELBO(self, data_batch):
        this_batch_size = len(data_batch)
        # finding prior loss
        t0_batch = torch.ones(this_batch_size, device=self.device) * self.t_start
        z_0, z_0_logprob = self.marginal.sample_and_log_prob(t0_batch, data_batch, 1)  # get sample and corresponding logprob from marginal distribution at t=t_start 
        prior_logprob = self.prior.log_prob(z_0)  # Get logprob of sample in the prior distribution
        prior_loss = z_0_logprob - prior_logprob

        # finding diffusion loss
        t_diff = self.time_dist.sample((this_batch_size,)).to(self.device)
        eps1 = self.marginal.base_dist.sample((this_batch_size,)) # shape (batch, state_dim)
        cond = self.condition_mapper(t_diff, data_batch) # shape (batch, state_dim * 2)
        z_diff, z_diff_logprob = self.marginal.forward_and_log_prob(eps1, t_diff, data_batch) # shape (batch, state_dim), (batch)
        #diffusion_inverse = mySDEproblem.diffusion(z_diff, t_diff).inverse()  # shape=(batch, state_dim, state_dim)
        diffusion_matrix = self.diffusion(z_diff, t_diff)  # shape=(batch, state_dim, state_dim)
        fd1 = self.marginal.SDEforwarddrift(z_diff, t_diff, data_batch)  # shape (batch, state_dim)
        #fd1 = mySDEproblem.marginal.SDEbackdrift(z_diff, t_diff, data_batch)  # shape (batch, state_dim)
        da = self.drift(z_diff, t_diff)  # shape (batch, state_dim)
        difference = fd1 - da  # shape (batch, state_dim)
        lower_cholesky = torch.linalg.cholesky(diffusion_matrix)  # shape=(batch, state_dim, state_dim)
        if self.device == "mps":
            # torch.cholesky_solve is not implemented for MPS, so we use torch.linalg.solve instead
            diffusion_inverse = torch.linalg.inv(diffusion_matrix)  # shape=(batch, state_dim, state_dim)
            residuals = torch.bmm(diffusion_inverse, difference.unsqueeze(2)).squeeze(2)  # shape (batch, state_dim)
        else:
            residuals =torch.cholesky_solve(difference.unsqueeze(2), lower_cholesky, upper=False).squeeze(2)  # shape=(batch, state_dim)
        #residuals = torch.bmm(diffusion_inverse, difference.unsqueeze(2)).squeeze(2)  # shape (batch, state_dim)
        diffusion_loss = .5 * (residuals**2).sum(dim=1)   # shape (batch)

        # finding reconstruction loss
        rec_ns = torch.randint(low=0, high=data_batch.shape[1], size=(this_batch_size,))
        batch_indices = torch.arange(this_batch_size)
        data_batch_slice = data_batch[batch_indices,rec_ns]
        z_rec, z_rec_logprob = self.marginal.sample_and_log_prob(data_batch_slice[:,0], data_batch, 1)
        x_batch = data_batch_slice[:,1:]
        reconstruction_loss = -self.emission.log_prob(x_batch, z_rec) * data_batch.shape[1]
        return diffusion_loss, prior_loss, reconstruction_loss

