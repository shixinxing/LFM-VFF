import torch
from gpytorch.means import Mean
from intdom_dgp.kernels.intdom_gp_kernels import VFFKernel


class _MeanZ_IdentityX(Mean):
    def __init__(self, vff_kernel: VFFKernel, batch_shape=torch.Size([])):
        super().__init__()
        self.vff_kernel = vff_kernel
        self.batch_shape = batch_shape

        self.kernel_batch_shape = self.vff_kernel.batch_shape  # [(D_out)]
        if len(self.kernel_batch_shape) != 0:
            assert self.batch_shape == self.kernel_batch_shape

    def _prepare_params(self, z):
        nu = self.vff_kernel.nu
        a = self.vff_kernel.a
        b = self.vff_kernel.b
        lamb = torch.sqrt(2. * nu) / self.vff_kernel.lengthscale
        outputscale = self.vff_kernel.outputscale
        sig2 = outputscale.view(outputscale.shape + torch.Size([1, 1]))  # [(D_out), 1, 1]

        z_cos_block = self.vff_kernel.vff_cos_expand_batch.unsqueeze(-1)  # [(D_out), M(+1), 1]
        z_sin_block = self.vff_kernel.vff_sin_expand_batch.unsqueeze(-1)
        return nu, a, b, lamb, sig2, z_cos_block, z_sin_block


class MeanZ_IdentityX(_MeanZ_IdentityX):
    def forward(self, z):
        """
        :param z: [(D_out), M, D_in]
        :return: prior mean(z) [D_out, M]
        """
        nu, a, b, lamb, sig2, z_cos_block, z_sin_block = self._prepare_params(z)

        if nu == 0.5:
            offset = (b + a) * (lamb * (b - a) + 2.) / (4. * sig2)
            cos_mean = (b + a) / (2. * sig2)
            cos_mean = cos_mean.expand_as(z_cos_block[..., 1:, :])
            sin_mean = lamb * (a - b) / (2. * sig2 * z_sin_block)
            mean = torch.cat([offset, cos_mean, sin_mean], dim=-2).squeeze(-1)
            return mean.expand(*self.batch_shape, mean.shape[-1])

        elif nu == 1.5:
            offset = (b + a) * (lamb * (b - a) + 4.) / (8. * sig2)
            cos_mean = (b + a) / (2. * sig2)
            cos_mean = cos_mean.expand_as(z_cos_block[..., 1:, :])
            sin_mean = lamb * (a - b) / (4. * sig2 * z_sin_block)
            sin_mean = sin_mean + (4. + (b - a) * lamb) * z_sin_block / (4. * sig2 * lamb.square())
            mean = torch.cat([offset, cos_mean, sin_mean], dim=-2).squeeze(-1)
            return mean.expand(*self.batch_shape, mean.shape[-1])

        elif nu == 2.5:
            offset = 3. * (b + a) * (lamb * (b - a) + 6.) / (32. * sig2)
            cos_mean = (3. * (a + b) * (3. * lamb.square() - z_cos_block[..., 1:, :].square()) /
                        (16. * lamb.square() * sig2))
            sin_mean = 3. * lamb * (a - b) / (16. * sig2 * z_sin_block)
            sin_mean = sin_mean + 3. * (16. + lamb * (b - a)) * z_sin_block / (16. * lamb.square() * sig2)
            mean = torch.cat([offset, cos_mean, sin_mean], dim=-2).squeeze(-1)
            return mean.expand(*self.batch_shape, mean.shape[-1])





