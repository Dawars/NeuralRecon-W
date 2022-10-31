import abc
import torch


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = weights.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples, device=device
        )
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1], device=device) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class RaySampler(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, n_samples, perturb):
        super(torch.nn.Module).__init__()
        super().__init__()
        self.n_samples = n_samples
        self.perturb = perturb

    @abc.abstractmethod
    def get_z_vals(self, rays_o, rays_d, near, far):
        pass


class UniformSampler(RaySampler):
    def __init__(self, n_samples, perturb):
        super().__init__(n_samples, perturb)

    def get_z_vals(self, rays_o, rays_d, near, far):
        device = rays_o.device
        batch_size = len(rays_o)

        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        # z_vals = near + (far - near) * z_vals[None, :]
        z_vals = near * (1. - z_vals[None, :]) + far * (z_vals[None, :])
        # https://cs.stackexchange.com/a/59650

        if self.perturb > 0:
            t_rand = torch.rand([batch_size, 1], device=device) - 0.5
            z_vals = z_vals + (far - near) * t_rand * 2.0 / self.n_samples  # todo fix behind camera

        return z_vals


class NeuSSampler(RaySampler):
    def __init__(self, n_samples, n_importance, perturb, neuconw, density_model, up_sample_steps, s_val_base):
        super().__init__(n_samples, perturb)

        self.n_importance = n_importance

        self.neuconw = neuconw
        self.density_model = density_model

        self.up_sample_steps = up_sample_steps
        self.s_val_base = s_val_base

        self.uniform_sampler = UniformSampler(n_samples, self.perturb)

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, step):
        """
        Up sampling give a fixed inv_s
        """
        device = sdf.device
        batch_size, n_samples = z_vals.shape
        pts = (
                rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        )  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        dist = next_z_vals - prev_z_vals

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------

        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat(
            [torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1
        )
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5

        # Eq 13
        p, c = self.density_model.density_func(prev_esti_sdf, next_esti_sdf, inv_s)
        alpha = (p + 1e-5) / (c + 1e-5)

        # transient alpha
        # alpha = alpha_s + alpha_t
        weights = (
                alpha
                * torch.cumprod(
            torch.cat(
                [torch.ones([batch_size, 1], device=device), 1.0 - alpha + 1e-7], -1
            ),
            -1,
        )[:, :-1]
        )

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        # if self.save_step_sample:
        #     # # start points
        #     # self.save_samples_step(pts[:, :-1][weights < 0.1].view(-1, 3), f"{step}_{0.0}_{0.1}")
        #     # # end points
        #     # self.save_samples_step(pts[:, :-1][((weights > 0.1) & (weights < 0.9))].view(-1, 3), f"{step}_{0.1}_{0.9}")
        #     # self.save_samples_step(pts[:, :-1].reshape(-1, 3), f"{step}_all")
        #
        #     # colored
        #     self.save_samples_step(
        #         pts[:, :-1].reshape(-1, 3),
        #         weights.reshape(
        #             -1,
        #         ),
        #         f"{step}_colored",
        #     )
        #
        #     pts_new = rays_o[:, None, :] + rays_d[:, None, :] * z_samples[..., :, None]
        #     self.save_samples_step(
        #         pts_new.reshape(-1, 3),
        #         torch.zeros_like(z_samples).reshape(
        #             -1,
        #         ),
        #         f"{step}",
        #         dir_name="new_z",
        #     )
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            _n_rays, _n_samples, _ = pts.size()
            new_sdf = self.neuconw.sdf(pts).reshape(_n_rays, _n_samples)  # todo why inference here?
            # # print("cat_z_vals ", new_sdf.size(), sdf.size())
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = (
                torch.arange(batch_size)[:, None]
                .expand(batch_size, n_samples + n_importance)
                .reshape(-1)
            )
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def get_z_vals(self, rays_o, rays_d, near, far):
        batch_size = len(rays_o)

        # uniform samples
        z_vals = self.uniform_sampler.get_z_vals(rays_o, rays_d, near, far)

        # upsample inside voxel
        if self.n_importance > 0:
            with torch.no_grad():
                pts = (
                        rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                )  # N_rays, N_samples, 3
                sdf = self.neuconw.sdf(pts).reshape(batch_size, self.n_samples)
                for i in range(self.up_sample_steps):
                    inv_s = 64 * 2 ** (self.s_val_base + i)
                    new_z_vals = self.up_sample(
                        rays_o,
                        rays_d,
                        z_vals,
                        sdf,
                        self.n_importance // self.up_sample_steps,
                        inv_s,
                        i,
                    )
                    z_vals, sdf = self.cat_z_vals(
                        rays_o,
                        rays_d,
                        z_vals,
                        new_z_vals,
                        sdf,
                        last=(i + 1 == self.up_sample_steps),
                    )

        return z_vals


class ErrorBoundSampler(RaySampler):
    def __init__(self, neuconw, density_model,perturb,
                 n_samples=64, n_samples_eval=128, n_samples_extra=32,
                 eps=0.1, beta_iters=10, max_total_iters=5, add_tiny=0.0):
        """
        n_samples=64:
        n_samples_eval=128:
        n_samples_extra=32:
        """
        super().__init__(n_samples, perturb)

        self.n_samples_eval = n_samples_eval
        self.uniform_sampler = UniformSampler(n_samples_eval, perturb)  # todo how many?

        self.n_samples_extra = n_samples_extra

        self.neuconw = neuconw
        self.density_model = density_model

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.add_tiny = add_tiny

    def get_z_vals(self, rays_o, rays_d, near, far):
        device = rays_o.device

        beta0 = self.density_model.get_beta(rays_o).detach()

        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(rays_o, rays_d, near, far)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = rays_o.unsqueeze(1) + samples.unsqueeze(2) * rays_d.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                samples_sdf = self.neuconw.sdf(points_flat)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1, device=device)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = self.density_model(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat([dists, torch.tensor([1e10], device=device).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=device), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                ''' Sample more points proportional to the current error bound'''

                N = self.n_samples_eval

                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:, :-1] ** 2.) / (
                        4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                ''' Sample the final sample set to be used in the volume rendering integral '''

                N = self.n_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

            # Invert CDF  # todo unify with sample_pdf
            if (not_converge and total_iters < self.max_total_iters) or (self.perturb <= 0):
                u = torch.linspace(0., 1., steps=N, device=device).unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N], device=device)
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        if self.n_samples_extra > 0:
            if self.perturb > 0:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.n_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.n_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points for the eikonal loss  # todo check
        # idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],), device=device)
        # z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        return z_vals  #, z_samples_eik

    def get_error_bound(self, beta, sdf, z_vals, dists, d_star):
        device = z_vals.device
        density = self.density_model.density_func(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros([dists.shape[0], 1], device=device), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]
