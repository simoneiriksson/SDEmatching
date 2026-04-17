import torch
import torch.nn as nn
class PeriodicKernel(nn.Module):
    """
    Periodic kernel: k(t,t') = sigma_f² * exp(-2 sin²(π(t-t')/p) / l²)

    Hyperparameters are per obs_dim, learned in log space.

    Methods:
        forward(t1, t2):           (obs_dim, B, N, M)  cross-covariance
        K_obs(t, data_mask):       (obs_dim, B, T, T)  noisy kernel matrix
        dK_dt1(t1, t2):            (obs_dim, B, N, M)  derivative w.r.t. t1
        d2K_dt1dt2_diag(t):        (obs_dim,)          prior variance of f'(t)
    """
    def __init__(self, obs_dim, device='cpu'):
        super().__init__()
        self.obs_dim     = obs_dim
        self.device      = device
        self.log_l       = nn.Parameter(torch.zeros(obs_dim))
        self.log_sigma_f = nn.Parameter(torch.zeros(obs_dim))
        self.log_sigma_n = nn.Parameter(torch.zeros(obs_dim))
        self.log_p       = nn.Parameter(torch.zeros(obs_dim))  # log period

    @property
    def l(self):       return self.log_l.exp()
    @property
    def sigma_f(self): return self.log_sigma_f.exp()
    @property
    def sigma_n(self): return self.log_sigma_n.exp()
    @property
    def p(self):       return self.log_p.exp()

    def forward(self, t1, t2):
        """
        K(t1, t2): (obs_dim, B, N, M)
        t1: (B, N), t2: (B, M)
        """
        diff = t1.unsqueeze(2) - t2.unsqueeze(1)               # (B, N, M)
        u    = torch.pi * diff.unsqueeze(0) / self.p.view(-1,1,1,1)      # (obs_dim, B, N, M)
        return self.sigma_f.view(-1,1,1,1)**2 * torch.exp(
            -2 * torch.sin(u)**2 / self.l.view(-1,1,1,1)**2
        )                                                        # (obs_dim, B, N, M)

    def K_obs(self, t, data_mask):
        """
        K(T,T) + noise, with masked entries inflated on diagonal.
        t:         (B, T)
        data_mask: (B, T)  1=masked, 0=valid
        """
        B, T    = t.shape
        K       = self.forward(t, t)                             # (obs_dim, B, T, T)
        eye     = torch.eye(T, device=self.device)
        noise   = (self.sigma_n.view(-1,1,1,1)**2 + 1e-6) * eye
        INF     = 1e6
        inflate = torch.diag_embed(
            INF * data_mask.float().unsqueeze(0).expand(self.obs_dim,-1,-1)
        )
        return K + noise + inflate                               # (obs_dim, B, T, T)

    def dK_dt1(self, t1, t2):
        """
        dk/dt1: (obs_dim, B, N, M)
        Derivative of k w.r.t. t1.

        dk/dt1 = k(t1,t2) * (-2/l²) * sin(2u) * (π/p)
        where u = π(t1-t2)/p
        """
        diff = t1.unsqueeze(2) - t2.unsqueeze(1)               # (B, N, M)
        u    = torch.pi * diff.unsqueeze(0) / self.p.view(-1,1,1,1)      # (obs_dim, B, N, M)
        K    = self.forward(t1, t2)                             # (obs_dim, B, N, M)
        return K * (-2.0 / self.l.view(-1,1,1,1)**2) * torch.sin(2 * u) * (
            torch.pi / self.p.view(-1,1,1,1)
        )                                                        # (obs_dim, B, N, M)

    def d2K_dt1dt2_diag(self, t):
        """
        d²k/dt1 dt2 at t1=t2: prior variance of f'(t).
        = sigma_f² * 4π²/(p²l²)
        Returns (obs_dim,)
        """
        return self.sigma_f**2 * 4 * torch.pi**2 / (self.p**2 * self.l**2)    

class RBFKernel(nn.Module):
    def __init__(self, obs_dim, device='cpu'):
        super().__init__()
        self.obs_dim     = obs_dim
        self.device      = device
        self.log_l       = nn.Parameter(torch.zeros(obs_dim))
        self.log_sigma_f = nn.Parameter(torch.zeros(obs_dim))
        self.log_sigma_n = nn.Parameter(torch.zeros(obs_dim))

    @property
    def l(self):       return self.log_l.exp()
    @property
    def sigma_f(self): return self.log_sigma_f.exp()
    @property
    def sigma_n(self): return self.log_sigma_n.exp()

    def forward(self, t1, t2):
        """K(t1, t2): (obs_dim, B, N, M). t1: (B, N), t2: (B, M)"""
        diff = t1.unsqueeze(2) - t2.unsqueeze(1)                 # (B, N, M)
        rbf  = torch.exp(
            -0.5 * (diff.unsqueeze(0) / self.l.view(-1,1,1,1))**2
        )                                                         # (obs_dim, B, N, M)
        return self.sigma_f.view(-1,1,1,1)**2 * rbf

    def K_obs(self, t, data_mask):
        """
        K(T,T) + noise, with masked entries inflated on diagonal.
        t:         (B, T)
        data_mask: (B, T)  1=masked/padding, 0=valid   <- same convention as input
        """
        B, T   = t.shape
        K      = self.forward(t, t)                               # (obs_dim, B, T, T)
        eye    = torch.eye(T, device=self.device)                 # (T, T)
        noise  = (self.sigma_n.view(-1,1,1,1)**2 + 1e-6) * eye   # (obs_dim, 1, T, T)
        INF    = 1e6
        inflate = torch.diag_embed(
            INF * data_mask.float().unsqueeze(0).expand(self.obs_dim,-1,-1)
        )                                                         # (obs_dim, B, T, T)
        return K + noise + inflate

    def dK_dt1(self, t1, t2):
        """dk/dt1: (obs_dim, B, N, M). t1: (B, N), t2: (B, M)"""
        diff = t1.unsqueeze(2) - t2.unsqueeze(1)                 # (B, N, M)
        K    = self.forward(t1, t2)                               # (obs_dim, B, N, M)
        return -(diff.unsqueeze(0) / self.l.view(-1,1,1,1)**2) * K

    def d2K_dt1dt2_diag(self, t):
        """Prior variance of f'(t). Returns (obs_dim,)"""
        return self.sigma_f**2 / self.l**2


class GPLatentModel(nn.Module):
    def __init__(self, obs_dim, kernel=None, device='cpu', linearmapping=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.device  = device
        self.kernel  = kernel if kernel is not None else RBFKernel(obs_dim, device)
        self.linearmapping = linearmapping
        if linearmapping:
            self.W = nn.Parameter(torch.eye(2 * obs_dim))

    def forward(self, t_star, data, data_mask):
        """
        t_star:    (B,)
        data:      (B, T, 1+obs_dim)
        data_mask: (B, T)  1=masked, 0=valid
        """
        B, T, _ = data.shape

        t_obs = data[:, :, 0]                                     # (B, T)
        x_obs = data[:, :, 1:]                                    # (B, T, obs_dim)

        # Zero out masked observations so they don't contribute to alpha
        valid = (data_mask == 0).float()                          # (B, T)  1=valid
        x_obs = x_obs * valid.unsqueeze(2)                        # (B, T, obs_dim)

        # --- Kernel matrices ---
        # Pass data_mask directly (1=masked convention)
        K_TT  = self.kernel.K_obs(t_obs, data_mask)              # (obs_dim, B, T, T)

        t_star_ = t_star.unsqueeze(1)                             # (B, 1)
        k_sT  = self.kernel.forward(t_star_, t_obs).squeeze(2)   # (obs_dim, B, T)
        dk_sT = self.kernel.dK_dt1(t_star_, t_obs).squeeze(2)    # (obs_dim, B, T)

        # Zero out masked entries in cross-covariances
        k_sT  = k_sT  * valid.unsqueeze(0)                       # (obs_dim, B, T)
        dk_sT = dk_sT * valid.unsqueeze(0)                       # (obs_dim, B, T)

        k_ss   = self.kernel.sigma_f**2                           # (obs_dim,)
        d2k_ss = self.kernel.d2K_dt1dt2_diag(t_star)             # (obs_dim,)

        # --- Cholesky solve ---
        L     = torch.linalg.cholesky(K_TT)                      # (obs_dim, B, T, T)
        x_T   = x_obs.permute(2, 0, 1).unsqueeze(3)              # (obs_dim, B, T, 1)
        alpha = torch.cholesky_solve(x_T, L).squeeze(3)          # (obs_dim, B, T)

        # --- Posterior means ---
        mu     = (k_sT  * alpha).sum(dim=2)                      # (obs_dim, B)
        mu_dot = (dk_sT * alpha).sum(dim=2)                      # (obs_dim, B)

        # --- Posterior variances ---
        v     = torch.cholesky_solve(k_sT.unsqueeze(3),  L).squeeze(3)  # (obs_dim, B, T)
        v_dot = torch.cholesky_solve(dk_sT.unsqueeze(3), L).squeeze(3)  # (obs_dim, B, T)

        var     = (k_ss.view(-1,1)   - (k_sT  * v    ).sum(dim=2)).clamp(min=1e-6)
        var_dot = (d2k_ss.view(-1,1) - (dk_sT * v_dot).sum(dim=2)).clamp(min=1e-6)

        # --- Reshape to (B, obs_dim) ---
        mu        = mu.T
        mu_dot    = mu_dot.T
        sigma     = var.sqrt().T
        sigma_dot = var_dot.sqrt().T

        # Linear transform
        if self.linearmapping:
            # Stack mu and mu_dot: (B, 2*obs_dim)
            gp_out = torch.cat([mu, mu_dot], dim=1)

            # Apply linear map to get latent state mean
            latent_mean = gp_out @ self.W.T                          # (B, 2*obs_dim)

            # For the log_sigma, scale by absolute value of W diagonal
            # (uncertainty transforms linearly too)
            gp_log_sigma = torch.cat([sigma.log(), sigma_dot.log()], dim=1)
            # Transform variance: Var[Wx] = W Var[x] W^T
            # For simplicity, just scale sigma by |W| diagonal
            latent_log_sigma = gp_log_sigma + self.W.abs().diagonal().log()
        else:
            latent_mean = torch.cat([mu, mu_dot], dim=1)           # (B, 2*obs_dim)
            latent_log_sigma = torch.cat([sigma.log(), sigma_dot.log()], dim=1)  # (B, 2*obs_dim)

        return torch.cat([latent_mean, latent_log_sigma], dim=1)

