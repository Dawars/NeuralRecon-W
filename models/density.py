"""
This implementation is borrowed from VolSDF https://github.com/lioryariv/volsdf
"""
import torch.nn as nn
import torch


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            self.register_parameter(p, nn.Parameter(torch.tensor(params_init[p])))


    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta(sdf)

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self, device_tensor):
        beta = self.beta.abs() + self.beta_min
        return beta.to(device_tensor.device)


class LogisticDensity(Density):
    def __init__(self, params_init={}):
        super().__init__(params_init=params_init)

    def density_func(self, prev_sdf, next_sdf, inv_s=None):
        if inv_s is None:
            inv_s = self.get_invs(prev_sdf)

        prev_cdf = torch.sigmoid(prev_sdf * inv_s)
        next_cdf = torch.sigmoid(next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf

        return p, c

    def get_invs(self, device_tensor):
        inv_s = torch.ones([len(device_tensor), 1]).to(device_tensor.device) * torch.exp(self.variance * 10.0)
        return inv_s


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)