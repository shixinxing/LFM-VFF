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

