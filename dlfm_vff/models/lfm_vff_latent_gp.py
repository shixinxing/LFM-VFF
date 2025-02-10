import torch
from gpytorch.kernels import ScaleKernel, MaternKernel

from intdom_dgp_lmc.models.layer_vff_latent_gp import VFF_LatentGP
from intdom_dgp.kernels.intdom_gp_kernels import VFFKernel
from intdom_dgp_lmc.models.model_id_utils import _build_mean_module
from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d
from dlfm_vff.kernels.lfm_intdom_kernels import LFM_VFF_InterDomainKernel


class LFM_VFF_LatentGP(VFF_LatentGP):
    """
    Latent LFMs induced by VFFs before LMC layers
    """
    def __init__(
            self,
            input_dims: int,
            a, b, M, nu,
            mean_x_type='zero', mean_z_type='zero', ode_type='ode1',  # or `with x`
            whitened: str = 'none',
            kernel_sharing_across_dims=False,
            has_kernel_noise=False,
            boundary_learnable=False, jitter_val=None
    ):
        self.ode_type = ode_type
        super().__init__(
            input_dims, a, b, M, nu, mean_x_type, mean_z_type,
            whitened, kernel_sharing_across_dims, has_kernel_noise, boundary_learnable, jitter_val
        )
        # alias
        delattr(self, 'scaled_matern_kernel')
        self.lfm_kernel, self.vff_kernel, self.cross_kernel = self.kxx, self.kzz, self.kzx

    def build_mean_module(self):
        if self.ode_type == 'ode1':
            return _build_mean_module(self, self.mean_x_type, self.mean_z_type, model_type='LFM_ode1')
        else:
            raise NotImplementedError

    def build_kernels(
            self,
            input_dims,
            a, b, M, nu,
            kernel_sharing_across_dims, boundary_learnable
    ):
        scaled_matern_kernel = ScaleKernel(MaternKernel(
            nu=nu,
            batch_shape=torch.Size([]) if kernel_sharing_across_dims else torch.Size([input_dims])
        ))
        lfm_kernel = LFM_MaternKernel1d(
            scaled_matern_kernel,
            ode_type=self.ode_type
        )
        vff_kernel = VFFKernel(
            a, b, M, scaled_matern_kernel, boundary_learnable
        )
        cross_kernel = LFM_VFF_InterDomainKernel(
            lfm_kernel, vff_kernel
        )
        return lfm_kernel, vff_kernel, cross_kernel

    def create_variational_strategy(self, inducing_points, jitter_val):
        return super().create_variational_strategy(inducing_points, jitter_val)


