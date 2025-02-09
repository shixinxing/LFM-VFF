from typing import Optional
import math

import torch
from torch import Tensor

from linear_operator.operators import LowRankRootLinearOperator, RootLinearOperator, MatmulLinearOperator

from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d


class LFMMatern_RFF_Kernel1d(LFM_MaternKernel1d):
    """
    approximate RFF G*K*G kernel by Fast kernel approximation
    """
    def __init__(
            self, scaled_matern_rff_kernel,
            ode_type='ode1', integral_lower_bound: Optional[float] = None,  # None stands for - inf
            **kwargs
    ):
        super().__init__(scaled_matern_rff_kernel, ode_type, **kwargs)
        self.scaled_matern_rff_kernel = scaled_matern_rff_kernel
        self.lower_bound = integral_lower_bound
        self.nu = scaled_matern_rff_kernel.base_kernel.nu

    @property
    def num_samples(self) -> int:
        return self.scaled_matern_rff_kernel.base_kernel.num_samples

    @property
    def randn_weights(self):  # [l, 1, D]
        return self.scaled_matern_rff_kernel.base_kernel.randn_weights

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, compute='real', **params):
        """
        x1, x2: [..., (S), l, N, 1]
        adapted from gpytorch RFFKernel
        """
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self._featurize(x1, normalize=False, compute=compute)
        if not x1_eq_x2:
            z2 = self._featurize(x2, normalize=False, compute=compute)
        else:
            z2 = z1
        D = float(self.num_samples)
        if diag:
            return (z1 * z2).sum(-1) / D
        if x1_eq_x2:
            if z1.size(-1) < z2.size(-2):
                return LowRankRootLinearOperator(z1 / math.sqrt(D))
            else:
                return RootLinearOperator(z1 / math.sqrt(D))
        else:
            return MatmulLinearOperator(z1 / D, z2.transpose(-1, -2))

    def _featurize(self, x: Tensor, normalize: bool = False, compute='real'):
        """
        x: [..., N, 1]
        return real random features {\varphi: [t, N, 2D]} of GKG kernel
        """
        beta = self.beta.unsqueeze(-1).unsqueeze(-1)
        gama = self.gama.unsqueeze(-1).unsqueeze(-1)
        sig2 = self.outputscale.unsqueeze(-1).unsqueeze(-1)
        A = self.lower_bound

        features = self.scaled_matern_rff_kernel.base_kernel._featurize(x, normalize)  # noqa, [cos(x), sin(x)]
        cos_wx, sin_wx = features[..., :self.num_samples], features[..., self.num_samples:]
        W = self.randn_weights / self.lengthscale.mT  # [t, 1, D]

        if compute == 'real':
            # using sin/cos to compute random features
            denomi = torch.square(gama) + torch.square(W)

            varphi_real = gama * cos_wx + W * sin_wx
            varphi_real = varphi_real / denomi

            varphi_img = gama * sin_wx - W * cos_wx
            varphi_img = varphi_img / denomi

            if A is None:  # -infty
                varphi = torch.cat([varphi_real, varphi_img], dim=-1)   # z: [t, N, 2D]
                return varphi * torch.sqrt(sig2) / beta

            nume = W * torch.sin(A * W) + gama * torch.cos(A * W)
            varphi_real = varphi_real - torch.exp(gama * (A - x)) * nume / denomi
            nume = gama * torch.sin(A * W) - W * torch.cos(A * W)
            varphi_img = varphi_img - torch.exp(gama * (A - x)) * nume / denomi
            varphi = torch.cat([varphi_real, varphi_img], dim=-1)
            return varphi * torch.sqrt(sig2) / beta

        elif compute == 'complex':
            dtype = torch.complex64 if torch.get_default_dtype() == torch.float32 else torch.complex128
            D = torch.as_tensor(self.num_samples, dtype=torch.get_default_dtype())
            j = torch.as_tensor(1.j, dtype=dtype)
            exp_iwt = torch.exp(W * x * j)
            denomi = gama + j * W

            if A is None:
                varphi = exp_iwt / denomi
                if normalize:  # real computation is already normalized but complex case isn't
                    varphi = varphi / torch.sqrt(D)
                return torch.cat([varphi.real, varphi.imag], dim=-1) * torch.sqrt(sig2) / beta

            nume = exp_iwt - torch.exp(gama * (A - x) + W * A * j)
            varphi = nume / denomi
            if normalize:
                varphi = varphi / torch.sqrt(D)
            return torch.cat([varphi.real, varphi.imag], dim=-1) * torch.sqrt(sig2) / beta

        else:
            raise ValueError(f"wrong compute argument: {compute}.")


if __name__ == '__main__':
    from vffdeeplfm.kernels.rff_matern_kernels import Matern_RFF_Kernel
    from gpytorch.kernels import MaternKernel, ScaleKernel

    def test_kernel(nu=0.5, num_samples=100, batch_shape=torch.Size([]), diag=False):
        scaled_rff_kernel = ScaleKernel(Matern_RFF_Kernel(nu, num_samples, batch_shape))
        scaled_rff_kernel.base_kernel.lengthscale = torch.rand(*batch_shape, 1, 1) * 2
        scaled_rff_kernel.outputscale = torch.rand(*batch_shape) * 5

        scaled_matern_kernel = ScaleKernel(MaternKernel(nu=nu, batch_shape=batch_shape))
        scaled_matern_kernel.base_kernel.lengthscale = scaled_rff_kernel.base_kernel.lengthscale
        scaled_matern_kernel.outputscale = scaled_rff_kernel.outputscale

        lfm_rff_kernel = LFMMatern_RFF_Kernel1d(scaled_rff_kernel, integral_lower_bound=None)
        lfm_rff_kernel.alpha = torch.rand(*batch_shape)
        lfm_rff_kernel.beta = torch.rand(*batch_shape)

        lfm_kernel = LFM_MaternKernel1d(scaled_matern_kernel)
        lfm_kernel.alpha = lfm_rff_kernel.alpha
        lfm_kernel.beta = lfm_rff_kernel.beta

        x1 = torch.randn(*batch_shape, 4, 1) * 3
        x2 = torch.randn(*batch_shape, 5, 1) * 2
        K_truth = lfm_kernel(x1, x2, diag=diag).to_dense()
        K_rff_real = lfm_rff_kernel(x1, x2, compute='real', diag=diag).to_dense()
        K_rff_complex = lfm_rff_kernel(x1, x2, compute='complex', diag=diag).to_dense()

        print(f"===== nu:{nu}, batch_shape:{batch_shape}, "
              f"lower bound: {lfm_rff_kernel.lower_bound} =====")
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}\n")
        print(f"GKG_truth (shape: {K_truth.shape}): \n{K_truth}\n")
        print(f"GKG_rff_real (shape: {K_rff_real.shape}): \n{K_rff_real}\n")
        print(f"GKG_rff_complex (shape: {K_rff_complex.shape}): \n{K_rff_complex}\n")
        delta = K_rff_complex - K_rff_real
        print(f"delta(complex - real) abs_max: {torch.max(delta)})\n")

        delta = K_truth - K_rff_real
        if torch.linalg.norm(K_truth) < 1e-5:
            raise ValueError
        relative_error = torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(K_truth)
        print(f"relatetive error: {relative_error}, abs_max: {torch.max(torch.abs(delta))}")


    torch.set_default_dtype(torch.float64)
    # test_kernel(nu=0.5, num_samples=1000000, batch_shape=torch.Size([2]), diag=False)
    # test_kernel(nu=1.5, num_samples=1000000, batch_shape=torch.Size([2]))
    test_kernel(nu=2.5, num_samples=1000000, batch_shape=torch.Size([2]), diag=False)













