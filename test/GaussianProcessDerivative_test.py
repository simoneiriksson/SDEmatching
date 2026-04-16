import pytest
import torch
import torch.nn as nn
import numpy as np
from SDEmatching.models.GaussianProcessDerivative import RBFKernel, GPLatentModel, PeriodicKernel
import math

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def device():
    return 'cpu'

@pytest.fixture
def obs_dim():
    return 2

@pytest.fixture
def rbf_kernel(obs_dim, device):
    return RBFKernel(obs_dim=obs_dim, device=device)

@pytest.fixture
def gp_model(obs_dim, device):
    kernel = RBFKernel(obs_dim=obs_dim, device=device)
    return GPLatentModel(obs_dim=obs_dim, kernel=kernel, device=device)

@pytest.fixture
def simple_data():
    """Simple pendulum-like data: B=3, T=10, obs_dim=2"""
    torch.manual_seed(42)
    B, T, obs_dim = 3, 10, 2
    t     = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)  # (B, T)
    x     = torch.sin(2 * np.pi * t)                             # (B, T)
    x     = x.unsqueeze(2).expand(-1, -1, obs_dim)               # (B, T, obs_dim)
    x     = x + 0.1 * torch.randn_like(x)                        # add noise
    data  = torch.cat([t.unsqueeze(2), x], dim=2)                 # (B, T, 1+obs_dim)
    mask  = torch.zeros(B, T)                                     # all valid
    t_star = torch.tensor([0.5, 0.5, 0.5])                        # (B,)
    return data, mask, t_star

@pytest.fixture
def masked_data():
    """Data with variable length sequences via masking: B=3, T=10, obs_dim=2"""
    torch.manual_seed(42)
    B, T, obs_dim = 3, 10, 2
    t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    x    = torch.sin(2 * np.pi * t).unsqueeze(2).expand(-1, -1, obs_dim)
    x    = x + 0.1 * torch.randn_like(x)
    data = torch.cat([t.unsqueeze(2), x], dim=2)

    # Different valid lengths per batch element: 10, 7, 5
    mask = torch.zeros(B, T)
    mask[1, 7:] = 1   # batch element 1: only 7 valid
    mask[2, 5:] = 1   # batch element 2: only 5 valid

    t_star = torch.tensor([0.5, 0.5, 0.5])
    return data, mask, t_star


# ============================================================
# RBFKernel tests
# ============================================================

class TestRBFKernel:

    def test_output_shape_forward(self, rbf_kernel):
        """K(t1, t2) should be (obs_dim, B, N, M)"""
        B, N, M = 3, 10, 8
        t1 = torch.randn(B, N)
        t2 = torch.randn(B, M)
        K  = rbf_kernel(t1, t2)
        assert K.shape == (rbf_kernel.obs_dim, B, N, M)

    def test_output_shape_K_obs(self, rbf_kernel):
        """K_obs should be (obs_dim, B, T, T)"""
        B, T = 3, 10
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        K    = rbf_kernel.K_obs(t, mask)
        assert K.shape == (rbf_kernel.obs_dim, B, T, T)

    def test_output_shape_dK_dt1(self, rbf_kernel):
        """dK_dt1 should be (obs_dim, B, N, M)"""
        B, N, M = 3, 10, 8
        t1 = torch.randn(B, N)
        t2 = torch.randn(B, M)
        dK = rbf_kernel.dK_dt1(t1, t2)
        assert dK.shape == (rbf_kernel.obs_dim, B, N, M)

    def test_output_shape_d2K_diag(self, rbf_kernel):
        """d2K_dt1dt2_diag should be (obs_dim,)"""
        t_star = torch.tensor(0.5)
        d2K    = rbf_kernel.d2K_dt1dt2_diag(t_star)
        assert d2K.shape == (rbf_kernel.obs_dim,)

    def test_kernel_symmetry(self, rbf_kernel):
        """K(t1, t2) should equal K(t2, t1).transpose"""
        B, T = 3, 10
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        K    = rbf_kernel(t, t)
        assert torch.allclose(K, K.transpose(2, 3), atol=1e-6)

    def test_kernel_positive_definite(self, rbf_kernel):
        """K_obs should be positive definite (all eigenvalues > 0)"""
        B, T = 2, 8
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        K    = rbf_kernel.K_obs(t, mask)
        # Check via Cholesky (will raise if not PD)
        for d in range(rbf_kernel.obs_dim):
            for b in range(B):
                torch.linalg.cholesky(K[d, b])

    def test_kernel_diagonal_is_sigma_f_squared(self, rbf_kernel):
        """K(t, t) diagonal should equal sigma_f^2 (before noise)"""
        B, T = 2, 5
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        K    = rbf_kernel(t, t)                                   # no noise
        diag = torch.diagonal(K, dim1=2, dim2=3)                  # (obs_dim, B, T)
        expected = rbf_kernel.sigma_f**2                           # (obs_dim,)
        assert torch.allclose(diag, expected.view(-1,1,1).expand_as(diag), atol=1e-6)

    def test_masked_entries_inflated(self, rbf_kernel):
        """Masked entries should have large diagonal values in K_obs"""
        B, T = 2, 5
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        mask[0, -2:] = 1   # last 2 entries of batch 0 are masked
        K    = rbf_kernel.K_obs(t, mask)
        # Masked diagonal entries should be >> unmasked
        assert K[0, 0, -1, -1] > 1e4
        assert K[0, 0, -2, -2] > 1e4
        assert K[0, 0,  0,  0] < 10    # unmasked should be normal

    def test_derivative_numerical(self, rbf_kernel):
        """dK_dt1 should match numerical differentiation"""
        B  = 1
        t1 = torch.tensor([[0.5]], requires_grad=False)
        t2 = torch.tensor([[0.3, 0.7]])
        eps = 1e-4

        t1_plus  = t1 + eps
        t1_minus = t1 - eps
        K_plus   = rbf_kernel(t1_plus,  t2)
        K_minus  = rbf_kernel(t1_minus, t2)
        dK_num   = (K_plus - K_minus) / (2 * eps)                # (obs_dim, B, 1, M)

        dK_anal  = rbf_kernel.dK_dt1(t1, t2)                     # (obs_dim, B, 1, M)
        assert torch.allclose(dK_num, dK_anal, atol=1e-4)

    def test_d2K_numerical(self, rbf_kernel):
        """d2K_dt1dt2_diag should match numerical second derivative"""
        t_star = torch.tensor(0.5)
        eps    = 1e-4
        B      = 1

        t1      = t_star.view(1, 1)
        t2_plus  = (t_star + eps).view(1, 1)
        t2_minus = (t_star - eps).view(1, 1)

        dK_plus  = rbf_kernel.dK_dt1(t1, t2_plus ).squeeze()     # (obs_dim,)
        dK_minus = rbf_kernel.dK_dt1(t1, t2_minus).squeeze()     # (obs_dim,)
        d2K_num  = (dK_plus - dK_minus) / (2 * eps)

        d2K_anal = rbf_kernel.d2K_dt1dt2_diag(t_star)
        assert torch.allclose(d2K_num, d2K_anal, atol=1e-4)

    def test_hyperparameters_positive(self, rbf_kernel):
        """l, sigma_f, sigma_n should always be positive"""
        assert (rbf_kernel.l       > 0).all()
        assert (rbf_kernel.sigma_f > 0).all()
        assert (rbf_kernel.sigma_n > 0).all()

    def test_gradients_flow(self, rbf_kernel):
        """Gradients should flow through kernel to hyperparameters"""
        B, T = 2, 5
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        K    = rbf_kernel.K_obs(t, mask)
        K.sum().backward()
        assert rbf_kernel.log_l.grad       is not None
        assert rbf_kernel.log_sigma_f.grad is not None
        assert rbf_kernel.log_sigma_n.grad is not None


# ============================================================
# GPLatentModel tests
# ============================================================

class TestGPLatentModel:

    def test_output_shape(self, gp_model, simple_data):
        """Output should be (B, 4*obs_dim)"""
        data, mask, t_star = simple_data
        out = gp_model(t_star, data, mask)
        B   = data.shape[0]
        assert out.shape == (B, 4 * gp_model.obs_dim)

    def test_output_shape_masked(self, gp_model, masked_data):
        """Output shape should be correct even with masked inputs"""
        data, mask, t_star = masked_data
        out = gp_model(t_star, data, mask)
        B   = data.shape[0]
        assert out.shape == (B, 4 * gp_model.obs_dim)

    def test_output_is_finite(self, gp_model, simple_data):
        """Output should not contain NaN or Inf"""
        data, mask, t_star = simple_data
        out = gp_model(t_star, data, mask)
        assert torch.isfinite(out).all()

    def test_output_is_finite_masked(self, gp_model, masked_data):
        """Output should not contain NaN or Inf with masked inputs"""
        data, mask, t_star = masked_data
        out = gp_model(t_star, data, mask)
        assert torch.isfinite(out).all()

    def test_log_sigma_is_finite(self, gp_model, simple_data):
        """log_sigma outputs should be finite (variances must be > 0)"""
        data, mask, t_star = simple_data
        out        = gp_model(t_star, data, mask)
        obs_dim    = gp_model.obs_dim
        log_sig    = out[:, 2*obs_dim:]                            # (B, 2*obs_dim)
        assert torch.isfinite(log_sig).all()

    def test_gradients_flow(self, gp_model, simple_data):
        """Gradients should flow through GPLatentModel to kernel hyperparameters"""
        data, mask, t_star = simple_data
        out = gp_model(t_star, data, mask)
        out.sum().backward()
        assert gp_model.kernel.log_l.grad       is not None
        assert gp_model.kernel.log_sigma_f.grad is not None
        assert gp_model.kernel.log_sigma_n.grad is not None

    def test_masked_does_not_affect_output(self, gp_model):
        """
        Adding extra masked observations should not change the output.
        """
        torch.manual_seed(0)
        B, T, obs_dim = 1, 5, gp_model.obs_dim
        t      = torch.linspace(0, 1, T).view(1, T)
        x      = torch.randn(B, T, obs_dim)
        data   = torch.cat([t.unsqueeze(2).expand(B,-1,-1), x], dim=2)
        mask   = torch.zeros(B, T)
        t_star = torch.tensor([0.5])

        out_original = gp_model(t_star, data, mask)

        # Append extra masked observations
        T2      = 3
        t_extra = torch.linspace(0, 1, T2).view(1, T2)
        x_extra = torch.randn(B, T2, obs_dim)
        data_extra = torch.cat([t_extra.unsqueeze(2).expand(B,-1,-1), x_extra], dim=2)
        data_padded = torch.cat([data, data_extra], dim=1)
        mask_padded = torch.cat([mask, torch.ones(B, T2)], dim=1)

        out_padded = gp_model(t_star, data_padded, mask_padded)

        assert torch.allclose(out_original, out_padded, atol=1e-4)

    def test_posterior_mean_close_to_observations(self, gp_model):
        """With low noise and short lengthscale, posterior mean at an
        observation time should be close to the observed value."""
        with torch.no_grad():
            gp_model.kernel.log_sigma_n.fill_(-5.0)   # very low noise
            gp_model.kernel.log_l.fill_(-2.0)          # short lengthscale: l ~ 0.13
            gp_model.kernel.log_sigma_f.fill_(0.0)

        B, T, obs_dim = 1, 10, gp_model.obs_dim
        t      = torch.linspace(0, 1, T).view(1, T)
        x      = torch.sin(2 * np.pi * t).unsqueeze(2).expand(B, -1, obs_dim).clone()
        data   = torch.cat([t.unsqueeze(2).expand(B,-1,-1), x], dim=2)
        mask   = torch.zeros(B, T)

        t_star      = t[0, 5].unsqueeze(0)
        out         = gp_model(t_star, data, mask)
        mu_pos      = out[:, :obs_dim]
        x_at_t_star = x[0, 5, :]

        assert torch.allclose(mu_pos[0], x_at_t_star, atol=1e-2)
        
    def test_custom_kernel(self, obs_dim, device):
        """Should work with a custom kernel that implements the interface"""
        class ConstantKernel(nn.Module):
            """Trivial kernel: k(t,t') = sigma_f^2 for all t, t'"""
            def __init__(self, obs_dim, device):
                super().__init__()
                self.obs_dim     = obs_dim
                self.device      = device
                self.log_sigma_f = nn.Parameter(torch.zeros(obs_dim))
                self.log_sigma_n = nn.Parameter(torch.zeros(obs_dim))

            @property
            def sigma_f(self): return self.log_sigma_f.exp()
            @property
            def sigma_n(self): return self.log_sigma_n.exp()

            def forward(self, t1, t2):
                B, N = t1.shape
                M    = t2.shape[1]
                return self.sigma_f.view(-1,1,1,1)**2 * torch.ones(
                    self.obs_dim, B, N, M, device=self.device
                )

            def K_obs(self, t, valid_mask):
                B, T   = t.shape
                K      = self.forward(t, t)
                eye    = torch.eye(T, device=self.device)
                noise  = (self.sigma_n.view(-1,1,1,1)**2 + 1e-6) * eye
                INF    = 1e6
                invalid = (1 - valid_mask).float()
                inflate = torch.diag_embed(
                    INF * invalid.unsqueeze(0).expand(self.obs_dim,-1,-1)
                )
                return K + noise + inflate

            def dK_dt1(self, t1, t2):
                B, N = t1.shape
                M    = t2.shape[1]
                return torch.zeros(self.obs_dim, B, N, M, device=self.device)

            def d2K_dt1dt2_diag(self, t):
                return torch.zeros(self.obs_dim, device=self.device)

        model  = GPLatentModel(obs_dim=obs_dim, kernel=ConstantKernel(obs_dim, device), device=device)
        B, T   = 3, 8
        t      = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        x      = torch.randn(B, T, obs_dim)
        data   = torch.cat([t.unsqueeze(2).expand(B,-1,-1), x], dim=2)
        mask   = torch.zeros(B, T)
        t_star = torch.tensor([0.5, 0.5, 0.5])
        out    = model(t_star, data, mask)

        assert out.shape == (B, 4 * obs_dim)
        assert torch.isfinite(out).all()


class TestPeriodicKernel:

    @pytest.fixture
    def periodic_kernel(self, obs_dim, device):
        kernel = PeriodicKernel(obs_dim=obs_dim, device=device)
        with torch.no_grad():
            kernel.log_p.fill_(math.log(math.pi))  # p = π
        return kernel

    def test_output_shape_forward(self, periodic_kernel):
        B, N, M = 3, 10, 8
        t1 = torch.randn(B, N)
        t2 = torch.randn(B, M)
        K  = periodic_kernel(t1, t2)
        assert K.shape == (periodic_kernel.obs_dim, B, N, M)

    def test_output_shape_K_obs(self, periodic_kernel):
        B, T = 3, 10
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        K    = periodic_kernel.K_obs(t, mask)
        assert K.shape == (periodic_kernel.obs_dim, B, T, T)

    def test_output_shape_dK_dt1(self, periodic_kernel):
        B, N, M = 3, 10, 8
        t1 = torch.randn(B, N)
        t2 = torch.randn(B, M)
        dK = periodic_kernel.dK_dt1(t1, t2)
        assert dK.shape == (periodic_kernel.obs_dim, B, N, M)

    def test_output_shape_d2K_diag(self, periodic_kernel):
        t_star = torch.tensor(0.5)
        d2K    = periodic_kernel.d2K_dt1dt2_diag(t_star)
        assert d2K.shape == (periodic_kernel.obs_dim,)

    def test_kernel_symmetry(self, periodic_kernel):
        B, T = 3, 10
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        K    = periodic_kernel(t, t)
        assert torch.allclose(K, K.transpose(2, 3), atol=1e-6)

    def test_kernel_positive_definite(self, periodic_kernel):
        B, T = 2, 8
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        K    = periodic_kernel.K_obs(t, mask)
        for d in range(periodic_kernel.obs_dim):
            for b in range(B):
                torch.linalg.cholesky(K[d, b])

    def test_periodicity(self, periodic_kernel):
        """k(t, t') should equal k(t, t' + p)"""
        with torch.no_grad():
            periodic_kernel.log_p.fill_(0.0)   # p = 1.0
        B  = 1
        t1 = torch.tensor([[0.3]])
        t2 = torch.tensor([[0.7]])
        t2_shifted = t2 + 1.0                  # shift by one period
        K1 = periodic_kernel(t1, t2)
        K2 = periodic_kernel(t1, t2_shifted)
        assert torch.allclose(K1, K2, atol=1e-6)

    def test_derivative_numerical(self, periodic_kernel):
        """dK_dt1 should match numerical differentiation"""
        B   = 1
        t1  = torch.tensor([[0.5]])
        t2  = torch.tensor([[0.3, 0.7]])
        eps = 1e-4
        K_plus  = periodic_kernel(t1 + eps, t2)
        K_minus = periodic_kernel(t1 - eps, t2)
        dK_num  = (K_plus - K_minus) / (2 * eps)
        dK_anal = periodic_kernel.dK_dt1(t1, t2)
        assert torch.allclose(dK_num, dK_anal, atol=1e-4)

    def test_d2K_numerical(self, periodic_kernel):
        """d2K_dt1dt2_diag should match numerical second derivative"""
        t_star   = torch.tensor(0.5)
        eps      = 1e-4
        t1       = t_star.view(1, 1)
        t2_plus  = (t_star + eps).view(1, 1)
        t2_minus = (t_star - eps).view(1, 1)
        dK_plus  = periodic_kernel.dK_dt1(t1, t2_plus ).squeeze()
        dK_minus = periodic_kernel.dK_dt1(t1, t2_minus).squeeze()
        d2K_num  = (dK_plus - dK_minus) / (2 * eps)
        d2K_anal = periodic_kernel.d2K_dt1dt2_diag(t_star)
        assert torch.allclose(d2K_num, d2K_anal, atol=1e-4)

    def test_hyperparameters_positive(self, periodic_kernel):
        assert (periodic_kernel.l       > 0).all()
        assert (periodic_kernel.sigma_f > 0).all()
        assert (periodic_kernel.sigma_n > 0).all()
        assert (periodic_kernel.p       > 0).all()

    def test_gradients_flow(self, periodic_kernel):
        B, T = 2, 5
        t    = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, T)
        K    = periodic_kernel.K_obs(t, mask)
        K.sum().backward()
        assert periodic_kernel.log_l.grad       is not None
        assert periodic_kernel.log_sigma_f.grad is not None
        assert periodic_kernel.log_sigma_n.grad is not None
        assert periodic_kernel.log_p.grad       is not None