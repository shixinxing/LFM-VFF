from typing import Optional
import torch
from gpytorch.kernels import ScaleKernel
from gpytorch.utils.memoize import cached

from intdom_dgp.models._intdom_latent_gp import _Intdom_LatentGP
from intdom_dgp_lmc.models.model_id_utils import _build_mean_module
from intdom_dgp_lmc.models.layer_vff_latent_gp import VFF_LatentGP

from dlfm_vff.kernels.rff_matern_kernels import Matern_RFF_Kernel
from dlfm_vff.kernels.rff_lfm_kernels import LFMMatern_RFF_Kernel1d
from dlfm_vff.kernels.rff_lfm_cross_kernels import LFMMatern_RFF_Cross_Kernel1d


class LFM_RFF_LatentGP(_Intdom_LatentGP):
    """
    LFM kernels are approximated using RFFs; inducing variables are place in latent force domain
    """
    def __init__(
            self,
            input_dims: int,
            nu,
            inducing_points: torch.Tensor,  # [D_in, M, 1], used for create vs
            num_samples=100,
            whitened: str = 'none',
            mean_x_type='zero', ode_type='ode1',  # or `with x`
            learning_inducing_locations: bool = True,
            kernel_sharing_across_dims=False, integral_lower_bound: Optional[float] = None,
            has_kernel_noise=False, jitter_val=None
    ):
        self.ode_type = ode_type
        self.num_samples = num_samples
        self.learning_inducing_locations = learning_inducing_locations
        self.lower_bound = integral_lower_bound

        lfm_rff_kernel, scaled_matern_rff_kernel, cross_kernel = self.build_kernels(
            input_dims, nu, kernel_sharing_across_dims
        )

        super().__init__(
            input_dims, lfm_rff_kernel, scaled_matern_rff_kernel, cross_kernel,
            inducing_points,
            mean_x_type=mean_x_type, mean_z_type='zero',  # ⚠️ force prior mean(Z) = 0
            whitened=whitened, kernel_sharing_across_dims=kernel_sharing_across_dims,
            has_kernel_noise=has_kernel_noise, jitter_val=jitter_val
        )
        # alias
        self.lfm_rff_kernel, self.scaled_matern_rff_kernel, self.cross_kernel = self.kxx, self.kzz, self.kzx

    def build_kernels(
            self,
            input_dims: int, nu,
            kernel_sharing_across_dims
    ):
        base_matern_rff_kernel = Matern_RFF_Kernel(
            nu, num_samples=self.num_samples,
            batch_shape=torch.Size([]) if kernel_sharing_across_dims else torch.Size([input_dims])
        )
        scaled_matern_rff_kernel = ScaleKernel(base_matern_rff_kernel)
        lfm_rff_kernel = LFMMatern_RFF_Kernel1d(
            scaled_matern_rff_kernel,
            ode_type=self.ode_type, integral_lower_bound=self.lower_bound
        )
        cross_kernel = LFMMatern_RFF_Cross_Kernel1d(lfm_rff_kernel)
        return lfm_rff_kernel, scaled_matern_rff_kernel, cross_kernel

    def create_variational_strategy(self, inducing_points, jitter_val):  # Z: [(D=l), 2M+1, 1]
        assert self.whitened == 'none' or self.whitened == 'cholesky'
        return VFF_LatentGP.create_variational_strategy(self, inducing_points, jitter_val)

    def build_mean_module(self):
        return _build_mean_module(self, mean_x_type=self.mean_x_type, mean_z_type=self.mean_z_type, model_type=None)

    @cached(name='covar_z_memo', ignore_args=True)
    def covar_z(self, z, *args, **kwargs):
        return self.scaled_matern_rff_kernel(z)

    @cached(name='covar_zx_memo', ignore_args=False)
    def covar_zx(self, z, x, compute='real'):
        return self.cross_kernel(z, x, compute=compute)







