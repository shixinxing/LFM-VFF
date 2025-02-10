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




