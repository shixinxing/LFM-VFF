from typing import Optional
import torch

from gpytorch.kernels import ScaleKernel, MaternKernel

from intdom_dgp.models.layer_vff_full import IntDom_GPLayer_Full
from intdom_dgp.kernels.intdom_gp_kernels import VFFKernel
from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d
from dlfm_vff.kernels.lfm_intdom_kernels import LFM_VFF_InterDomainKernel


class LFM_Layer_Full(IntDom_GPLayer_Full):
    """
    Extends IntDom_GPLayer_Full to LFM kernels;
    The variational `S` in q(u) is dense, not blockdiag
    """
    def __init__(
            self,
            input_dims: int, output_dims: int,
            a, b, M, nu,
            mean_x_type='zero', mean_z_type='zero', ode_type='ode1',
            whitened: str = 'none',
            kernel_sharing_across_output_dims=True,
            has_kernel_noise=False,
            boundary_learnable=False, jitter_val: Optional[float] = None
    ):
        self.ode_type = ode_type
        super().__init__(
            input_dims, output_dims, a, b, M, nu, mean_x_type, mean_z_type, whitened,
            kernel_sharing_across_output_dims, has_kernel_noise, boundary_learnable, jitter_val
        )  # will get Z: [D_in * (2M+1), 1]
        # alias
        self.lfm_kernel, self.vff_kernel, self.cross_kernel = self.kxx, self.kzz, self.kzx
        delattr(self, 'scaled_matern_kernel')

    def create_variational_strategy(self, inducing_points, jitter_val):
        return super().create_variational_strategy(inducing_points, jitter_val)

    def build_mean_module(self):
        return super().build_mean_module()

    # override
    def build_kernels(self, a, b, M, nu, boundary_learnable):
        """
        :return: Kernels with batch shape [D_in], haven't broadcast to `D_out`
        """
        scaled_matern_kernel = ScaleKernel(MaternKernel(
            nu=nu, batch_shape=torch.Size([self.input_dims])
        ))
        lfm_kernel = LFM_MaternKernel1d(scaled_matern_kernel, ode_type=self.ode_type)
        vff_kernel = VFFKernel(a, b, M, scaled_matern_kernel, boundary_learnable)
        cross_kernel = LFM_VFF_InterDomainKernel(lfm_kernel, vff_kernel)
        return lfm_kernel, vff_kernel, cross_kernel

