import torch

from intdom_dgp_lmc.means.mean_z_identity_x import _MeanZ_IdentityX


class LFM_MeanZ_IdentityX_ODE1(_MeanZ_IdentityX):
    def __init__(self, lfm_kernel, vff_kernel, batch_shape=torch.Size([])):
        super().__init__(vff_kernel, batch_shape=batch_shape)
        self.lfm_kernel = lfm_kernel

    def forward(self, z):
        nu, a, b, lamb, sig2, z_cos_block, z_sin_block = self._prepare_params(z)
        gama = self.lfm_kernel.gama.unsqueeze(-1).unsqueeze(-1)

        if nu == 0.5:
            offset = (gama * (b + a) + 2.) * (lamb * (b - a) + 2.) / (4. * sig2)
            cos_mean = (gama * (b + a) + 2.) / (2. * sig2)
            cos_mean = cos_mean.expand_as(z_cos_block[..., 1:, :])
            sin_mean = lamb * gama * (a - b) / (2. * sig2 * z_sin_block)
            mean = torch.cat([offset, cos_mean, sin_mean], dim=-2) / gama
            return mean.squeeze(-1)

        elif nu == 1.5:
            offset = (gama * (b + a) + 2.) * (lamb * (b - a) + 4.) / (8. * sig2)
            cos_mean = (gama * (b + a) + 2.) / (2. * sig2)
            cos_mean = cos_mean.expand_as(z_cos_block[..., 1:, :])
            sin_mean = gama * lamb * (a - b) / (4. * sig2 * z_sin_block)
            sin_mean = sin_mean + gama * (4. + (b - a) * lamb) * z_sin_block / (4. * sig2 * lamb.square())
            mean = torch.cat([offset, cos_mean, sin_mean], dim=-2) / gama  # [t, 2M+1]
            return mean.squeeze(-1)

        elif nu == 2.5:
            offset = 3. * (gama * (b + a) + 2.) * (lamb * (b - a) + 6.) / (32. * sig2)
            cos_mean = (3. * (gama * (a + b) + 2.) * (3. * lamb.square() - z_cos_block[..., 1:, :].square()) /
                        (16. * lamb.square() * sig2))
            sin_mean = 3. * lamb * (a - b) * gama / (16. * sig2 * z_sin_block)
            sin_mean = sin_mean + 3. * gama * (16. + lamb * (b - a)) * z_sin_block / (16. * lamb.square() * sig2)
            mean = torch.cat([offset, cos_mean, sin_mean], dim=-2) / gama
            return mean.squeeze(-1)





