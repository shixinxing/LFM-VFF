from typing import Optional
import torch
from gpytorch.kernels import Kernel
from linear_operator import to_linear_operator

from dlfm_vff.kernels.rff_lfm_kernels import LFMMatern_RFF_Kernel1d


class LFMMatern_RFF_Cross_Kernel1d(Kernel):
    """
    Approximate cross-covariance K*G ~ Kvf by Random Fourier Features
    All kernels (GKG, KG, K) must share the same random weights.
    """
    def __init__(
            self,
            lfm_rff_kernel: LFMMatern_RFF_Kernel1d,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.lfm_rff_kernel = lfm_rff_kernel  # K_ff
        self.scaled_matern_rff_kernel = lfm_rff_kernel.scaled_matern_rff_kernel
        self.nu = lfm_rff_kernel.nu
        self.lower_bound = lfm_rff_kernel.lower_bound

    @property
    def batch_shape(self):
        return self.lfm_rff_kernel.batch_shape

    @property
    def num_samples(self):
        return self.scaled_matern_rff_kernel.base_kernel.num_samples

    @property
    def randn_weights(self):  # [t, 1, D]
        return self.scaled_matern_rff_kernel.base_kernel.randn_weights

    def __call__(self, z: torch.Tensor, x: Optional[torch.Tensor] = None, compute='real', **params):
        """
        z: [(S), t, M, 1] in the latent force domain; x: [(S), t, N, 1] in the LFM domain
        :return: Kvf: [(S), t, M, N]
        """
        if x is None or x.size(-1) != 1 or z.size(-1) != 1:
            raise ValueError

        varphi_z = self.scaled_matern_rff_kernel.base_kernel._featurize(z, normalize=True)  # noqa
        sig2 = self.scaled_matern_rff_kernel.outputscale.unsqueeze(-1).unsqueeze(-1)
        varphi_z = varphi_z * torch.sqrt(sig2)

        varphi_x = self.lfm_rff_kernel._featurize(x, normalize=True, compute=compute)  # noqa
        return to_linear_operator(varphi_z @ varphi_x.mT)

    def forward(*args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    from gpytorch.kernels import ScaleKernel, MaternKernel
    from dlfm_vff.kernels.rff_matern_kernels import Matern_RFF_Kernel
    from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d
    from dlfm_vff.kernels.rff_lfm_kernels import LFMMatern_RFF_Kernel1d
    from dlfm_vff.kernels.lfm_intdom_kernels import LFM_InterDomainKernels

    def test_kernel(nu=0.5, num_samples=100, batch_shape=torch.Size([]), S=3):
        kernel = ScaleKernel(MaternKernel(nu=nu, batch_shape=batch_shape))
        kernel.base_kernel.lengthscale = torch.rand(*batch_shape, 1, 1) * 2
        kernel.outputscale = torch.rand(*batch_shape) * 5
        lfm_kernel = LFM_MaternKernel1d(kernel)   # GKG
        lfm_kernel.alpha = torch.rand(*batch_shape) * 2
        lfm_kernel.beta = torch.rand(*batch_shape) * 0.1
        kg = LFM_InterDomainKernels(lfm_kernel)  # KG

        kernel_rff = ScaleKernel(Matern_RFF_Kernel(nu, num_samples, batch_shape))
        kernel_rff.base_kernel.lengthscale = kernel.base_kernel.lengthscale
        kernel_rff.outputscale = kernel.outputscale
        lfm_kernel_rff = LFMMatern_RFF_Kernel1d(kernel_rff, integral_lower_bound=None)  # GKG
        lfm_kernel_rff.alpha = lfm_kernel.alpha
        lfm_kernel_rff.beta = lfm_kernel.beta
        kg_rff = LFMMatern_RFF_Cross_Kernel1d(lfm_kernel_rff)  # KG

        z = torch.randn(*batch_shape, 4, 1) * 5
        x = torch.randn(*batch_shape, 5, 1) * 3
        analytic_kg = kg(z, x).to_dense()
        approximate_kg_real = kg_rff(z, x, compute='real').to_dense()
        approximate_kg_complex = kg_rff(z, x, compute='complex').to_dense()

        print(f"===== nu:{nu}, batch_shape:{batch_shape} =====")
        print(f"z (shape: {z.shape}): \n{z}\n")
        print(f"x (shape: {x.shape}): \n{x}\n")
        print(f"analytic_KG ({analytic_kg.shape}): \n{analytic_kg}\n")
        print(f"approximate_KG_real ({approximate_kg_real.shape}): \n{approximate_kg_real}\n")

        delta = approximate_kg_complex - approximate_kg_real
        print(f"complex compute - real compute: {torch.max(torch.abs(delta))} ")

        delta = approximate_kg_real - analytic_kg
        if torch.linalg.norm(analytic_kg) < 1e-5:
            raise ValueError
        relative_error = torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(analytic_kg)
        print(f"relative error: {relative_error}, delta max abs: {torch.max(torch.abs(delta))})")

        # check shape
        xx = torch.randn(S, *batch_shape, 6, 1)
        re_kg = kg(z, xx).to_dense()
        re_kg_rff = kg_rff(z, xx, compute='real').to_dense()
        delta = re_kg_rff - re_kg
        print(f"check shape: z: {z.shape}, x: {xx.shape},\n ")
        print(f"analytical KG(z,x): {re_kg.shape}, rff KG(z, x): {re_kg_rff.shape}\n")
        print(f"relative error: error: {torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(re_kg)}"
              f"delta (max_abs): {torch.max(torch.abs(delta))}")

    torch.set_default_dtype(torch.float64)
    # test_kernel(nu=0.5, num_samples=1000000, batch_shape=torch.Size([2]))
    # test_kernel(nu=1.5, num_samples=1000000, batch_shape=torch.Size([2]))
    test_kernel(nu=2.5, num_samples=100000, batch_shape=torch.Size([2]))








