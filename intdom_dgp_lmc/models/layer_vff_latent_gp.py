from typing import Optional
import torch
from gpytorch.utils.memoize import cached

from dgp.inference.chol_variational_dist import CholeskyVariationalDist_ZeroInit

from intdom_dgp.models._intdom_latent_gp import _Intdom_LatentGP
from intdom_dgp_lmc.models.model_id_utils import _build_mean_module

from intdom_dgp.linear_op_utils.block_diag_Kuu import BlockDiagKuu

from intdom_dgp_lmc.inference.variational_strategy_idgp import (
    IDGP_WhitenedVariationalStrategy, IDGP_UnWhitenedVariationalStrategy
)


class VFF_LatentGP(_Intdom_LatentGP):
    """
    Inter-domain GP models for multidimensional inputs, providing independent GPs as outputs before LMC
    """
    def __init__(
            self,
            input_dims: int,
            a, b, M, nu,
            mean_x_type='zero', mean_z_type='zero',  # or 'with-x'
            whitened: str = 'none',
            kernel_sharing_across_dims=False,
            has_kernel_noise=False,
            boundary_learnable=False,
            jitter_val: Optional[float] = None  # past in vs, if None, jitter value: float: 1e-4; double: 1e-6
    ):
        self.nu = nu
        self.learning_inducing_locations = False
        scaled_matern_kernel, vff_kernel, cross_kernel = self.build_kernels(
            input_dims, a, b, M, nu, kernel_sharing_across_dims, boundary_learnable
        )
        inducing_points = torch.cat([vff_kernel.vff_cos_expand_batch, vff_kernel.vff_sin_expand_batch], dim=-1)
        inducing_points = inducing_points.unsqueeze(-1)  # [(l), 2M+1, 1]

        super().__init__(
            input_dims, kxx=scaled_matern_kernel, kzz=vff_kernel, kzx=cross_kernel,
            inducing_points=inducing_points,
            mean_x_type=mean_x_type, mean_z_type=mean_z_type,
            whitened=whitened,
            kernel_sharing_across_dims=kernel_sharing_across_dims,
            has_kernel_noise=has_kernel_noise, jitter_val=jitter_val
        )
        # alias
        self.scaled_matern_kernel, self.vff_kernel, self.cross_kernel = self.kxx, self.kzz, self.kzx

    def create_variational_strategy(self, inducing_points, jitter_val):  # Z: [(D=l), 2M+1, 1]
        if self.whitened == 'none' or self.whitened == 'cholesky':
            num_inducing = inducing_points.size(-2)  # 2M+1
        else:
            raise NotImplementedError("Unknown whitened strategy.")

        variational_distribution = CholeskyVariationalDist_ZeroInit(  # q(u): [l, 2M+1(+1)]
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([self.num_latents]),
            mean_init_std=1e-3,  # will initialize p(u) to q(u), which does not conform to DSVI paper
            force_mean_to_zero=False
        )
        if self.whitened == 'none':
            return IDGP_UnWhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learning_inducing_locations=self.learning_inducing_locations, jitter_val=jitter_val
            )
        elif self.whitened == 'cholesky':
            return IDGP_WhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learning_inducing_locations=self.learning_inducing_locations, jitter_val=jitter_val
            )
        else:
            raise NotImplementedError("Unknown whitened Variational Strategy.")

    def build_mean_module(self):
        return _build_mean_module(self, mean_x_type=self.mean_x_type, mean_z_type=self.mean_z_type, model_type='IDGP')

    def build_kernels(
            self,
            input_dims,
            a, b, M, nu,
            kernel_sharing_across_dims, boundary_learnable
    ):
        raise NotImplementedError

    @cached(name='covar_z_memo', ignore_args=False)
    def covar_z(self, z) -> BlockDiagKuu:  # [(l), 2M+1, 2M+1]
        return self.vff_kernel()  # noqa

    @cached(name='covar_zx_memo', ignore_args=False)
    def covar_zx(self, z, x):  # [l, 2M+1, N]
        return self.cross_kernel(x)




