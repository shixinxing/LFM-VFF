import torch
from torch import Tensor

from gpytorch.kernels import Kernel
from linear_operator import to_linear_operator

from intdom_dgp.kernels.intdom_gp_kernels import VFFKernel
from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d
from dlfm_vff.kernels.kernel_functions.func_MKG_matern12 import lfm_matern12_cross_cov
from dlfm_vff.kernels.kernel_functions.func_MKG_matern32 import lfm_matern32_cross_cov
from dlfm_vff.kernels.kernel_functions.func_MKG_matern52 import lfm_matern52_cross_cov
from dlfm_vff.kernels.kernel_functions.func_GK_matern12 import lfm_matern12_green_k
from dlfm_vff.kernels.kernel_functions.func_GK_matern32 import lfm_matern32_green_k
from dlfm_vff.kernels.kernel_functions.func_GK_matern52 import lfm_matern52_green_k


class LFM_VFF_InterDomainKernel(Kernel):
    """
    K_{vf} induced by VFFs and LFMs
    :return: [(S, l), 2M+1, 2M(+1)]
    """
    def __init__(self,
                 lfm_kernel: LFM_MaternKernel1d,
                 vff_kernel: VFFKernel,
                 **kwargs):
        super().__init__(**kwargs)   # This will set the batch shape equal to that of base Matern kernel
        self.lfm_kernel = lfm_kernel
        self.vff_kernel = vff_kernel
        self.nu = self.vff_kernel.nu

    @property
    def batch_shape(self) -> torch.Size:
        return self.lfm_kernel.batch_shape

    def __call__(self, x: Tensor, **params):
        """
        :param: x: [(S), l, N, d=1], with z:[l, 2M+1, d=1]
        :return: linear operator with [(S), l, 2M+1, N]
        """
        assert x.size(-1) == 1
        a, b = self.vff_kernel.a, self.vff_kernel.b
        lamb = self.lfm_kernel.lamda
        beta = self.lfm_kernel.beta.unsqueeze(-1).unsqueeze(-1)
        gama = self.lfm_kernel.gama.unsqueeze(-1).unsqueeze(-1)

        x_T = x.transpose(-1, -2)  # [(S, l) d=1, N]
        z_cos_block = self.vff_kernel.vff_cos_expand_batch.unsqueeze(-1)  # [(l), M+1, 1]
        if self.nu == 0.5:
            Kvf = lfm_matern12_cross_cov(z_cos_block, x_T, a, b, gama, beta, lamb)
        elif self.nu == 1.5:
            Kvf = lfm_matern32_cross_cov(z_cos_block, x_T, a, b, gama, beta, lamb)
        elif self.nu == 2.5:
            Kvf = lfm_matern52_cross_cov(z_cos_block, x_T, a, b, gama, beta, lamb)
        else:
            raise ValueError(f"Unknown nu: {self.nu}.")
        return to_linear_operator(Kvf)  # [...,t, 2M+1, N]

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class LFM_InterDomainKernels(Kernel):
    """
    Kuf by conventional latent force model: k (z, x) * G
    """
    def __init__(self, lfm_kernel: LFM_MaternKernel1d, **kwargs):
        super(LFM_InterDomainKernels, self).__init__(**kwargs)
        self.lfm_kernel = lfm_kernel
        self.nu = lfm_kernel.nu
        self.ode_type = lfm_kernel.ode_type

    @property
    def batch_shape(self) -> torch.Size:
        return self.lfm_kernel.batch_shape

    def __call__(self, z: Tensor, x: Tensor, **params):  # noqa
        """
        :param z, x: [(S), D_in, N, 1]
        :return: k (z, x) * G
        """
        # first compute G * k (x (t1), z (t2))
        assert z.size(-1) == 1 and z.size(-1) == 1
        lamb = self.lfm_kernel.lamda
        gama = self.lfm_kernel.gama.unsqueeze(-1).unsqueeze(-1)
        beta = self.lfm_kernel.beta.unsqueeze(-1).unsqueeze(-1)
        outputscale = self.lfm_kernel.outputscale.unsqueeze(-1).unsqueeze(-1)

        distance = self.covar_dist(x, z, diag=False, square_dist=False, **params)  # |t1 - t2|
        if len(distance.shape) < len(gama.shape):  # just for test, this will never happen in practice
            brdcst_shape = torch.broadcast_shapes(gama.shape, distance.shape)
            distance = distance.expand(*brdcst_shape)

        if self.nu == 0.5:
            output = lfm_matern12_green_k(x, z, distance, gama, beta, outputscale, lamb)
        elif self.nu == 1.5:
            output = lfm_matern32_green_k(x, z, distance, gama, beta, outputscale, lamb)
        elif self.nu == 2.5:
            output = lfm_matern52_green_k(x, z, distance, gama, beta, outputscale, lamb)
        else:
            raise ValueError(f"Wrong nu: {self.nu}.")

        return to_linear_operator(output.mT)

    def forward(*args, **kwargs):
        raise NotImplementedError

