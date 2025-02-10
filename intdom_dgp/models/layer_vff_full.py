from typing import Optional
import torch

from gpytorch.means import ZeroMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.memoize import cached
from linear_operator.operators import BlockDiagLinearOperator

from dgp.inference.chol_variational_dist import CholeskyVariationalDist_ZeroInit
from intdom_dgp.models._intdom_latent_gp import _Intdom_LatentGP
from intdom_dgp.linear_op_utils.block_diag_Kuu import BlockDiagKuu
from intdom_dgp.inference.variational_strategy_idgp_full import IDGP_Whitened_VariationalStrategy_Full


class IntDom_GPLayer_Full(_Intdom_LatentGP):
    """
    Each output dim has M * D_in inducing variables;
    The variational covar `S` in  q(u) = N(m, S) is dense, not block diag;
    """
    def __init__(
            self,
            input_dims: int, output_dims: int,
            a, b, M, nu,
            mean_x_type='zero', mean_z_type='zero',  # `linear` or `zero`
            whitened: str = 'none',
            kernel_sharing_across_output_dims=True,
            has_kernel_noise=False,
            boundary_learnable=False,
            jitter_val: Optional[float] = None
    ):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.mean_module, self.mean_z_module = None, None
        assert kernel_sharing_across_output_dims, NotImplementedError

        self.nu = nu
        self.learning_inducing_locations = False
        scaled_matern_kernel, vff_kernel, cross_kernel = self.build_kernels(a, b, M, nu, boundary_learnable)
        inducing_points = torch.cat([vff_kernel.vff_cos_expand_batch, vff_kernel.vff_sin_expand_batch], dim=-1)
        # Z: [D_in * (2M+1), 1], used as the argument in `create_variational_strategy`
        inducing_points = inducing_points.expand(self.input_dims, -1).view(-1, 1)

        super().__init__(
            input_dims, kxx=scaled_matern_kernel, kzz=vff_kernel, kzx=cross_kernel,
            inducing_points=inducing_points,
            mean_x_type=mean_x_type, mean_z_type=mean_z_type,
            whitened=whitened,
            kernel_sharing_across_dims=False,  # here refers to the input dims
            has_kernel_noise=has_kernel_noise, jitter_val=jitter_val
        )
        # alias
        self.scaled_matern_kernel, self.vff_kernel, self.cross_kernel = self.kxx, self.kzz, self.kzx

    def create_variational_strategy(self, inducing_points, jitter_val):  # Z: [D_in * (2M+1), 1]
        if self.whitened == 'none' or self.whitened == 'cholesky':
            num_inducing = inducing_points.size(-2)  # D_in * (2M+1)
        else:
            raise NotImplementedError("Unknown whitened strategy.")

        variational_distribution = CholeskyVariationalDist_ZeroInit(     # q(u): [D_out, D_in * (2M+1)]
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([self.output_dims]),
            mean_init_std=1e-3,  # will initialize p(u) to q(u)
            force_mean_to_zero=False
        )
        if self.whitened == 'none':
            raise NotImplementedError
        elif self.whitened == 'cholesky':
            return IDGP_Whitened_VariationalStrategy_Full(
                self, inducing_points, variational_distribution, jitter_val=jitter_val
            )
        else:
            raise NotImplementedError("Unknown whitened Variational Strategy.")

    def build_mean_module(self):
        if self.mean_x_type == 'zero':
            self.mean_module = ZeroMean(batch_shape=torch.Size([self.output_dims]))
        elif self.mean_x_type == 'linear':
            # The weights will be initialized later when build deep models
            self.mean_module = LinearMean(input_size=self.input_dims,
                                          batch_shape=torch.Size([self.output_dims]),
                                          bias=False)  # [D_out, input_size=D_in, 1]
        else:
            raise NotImplementedError(f"invalid mean_type for x: {self.mean_module}.")

        if self.mean_z_type == 'zero':
            self.mean_z_module = ZeroMean(batch_shape=torch.Size([self.output_dims]))
        else:
            raise NotImplementedError(f"invalid mean_type for z: {self.mean_z_type}.")

    def build_kernels(self, a, b, M, nu, boundary_learnable):
        raise NotImplementedError

    def forward(self, x, output_form='Full', full_cov=True):
        if x is None:
            assert output_form == 'VFF'  # used to get prior p(u)
        else:
            assert x.size(-1) == 1 and x.size(-3) == self.input_dims

        z = self.variational_strategy.inducing_points  # [D_in * (2M + 1), 1]
        assert z.shape == torch.Size([self.input_dims * self.vff_kernel.num_features, 1])

        if output_form == 'VFF':   # p(u), used in Variational Strategy and  q(u) initialization
            mean_z = self.mf_z(z)  # [D_out, D_in * (2M+1)]
            covar_z = self.covar_z(z)  # [D_in, 2M+1, 2M+1]
            # covar_z = covar_z.add_jitter(settings.variational_cholesky_jitter.value())  # optional, usually psd
            covar_z = BlockDiagLinearOperator(covar_z, block_dim=-3)  # [D_in * (2M+1), D_in * (2M+1)]
            covar_z = covar_z.expand(*mean_z.shape[:-1], *covar_z.shape[-2:])  # expand to `D_out`
            return MultivariateNormal(mean_z, covar_z)  # q(u): MVN [D_out, D_in * (2M+1)]

        elif output_form == 'Full':
            mean_x = self.mf_x(x.transpose(-1, -3))  # [(S), D_out, N]
            mean_z = self.mf_z(z)  # cached [D_out, D_in * (2M+1)]
            covar_x = self.covar_x_with_noise(x, diag=not full_cov)    # [(S), D_in, N, N]
            covar_zz, covar_zx = self.covar_z(z), self.covar_zx(z, x)  # [D_in, 2M+1, 2M+1] / [S, D_in, 2M+1, N]
            covar_x, covar_zx = covar_x.unsqueeze(-4), covar_zx.unsqueeze(-4)
            covar_x = covar_x.expand(*covar_x.shape[:-4], self.output_dims, *covar_x.shape[-3:])
            covar_zx = covar_zx.expand(*covar_zx.shape[:-4], self.output_dims, *covar_zx.shape[-3:])
            assert covar_zz.size(-3) == self.input_dims and covar_zx.size(-3) == self.input_dims
            return mean_x, mean_z, covar_x, covar_zz, covar_zx

        elif output_form == 'Prior':
            mean_x = self.mf_x(x)
            covar_x = self.covar_x_with_noise(x, diag=not full_cov)
            covar_x = covar_x.sum(dim=-3).unsqueeze(-3)  # [(S), 1, N, N]
            covar_x = covar_x.expand(covar_x.shape[:-3], self.output_dims, covar_x.shape[-2:])
            return MultivariateNormal(mean_x, covar_x)  # [(S), D_out, N]
        else:
            raise ValueError(f"Invalid output_form: {output_form}.")

    @cached(name='covar_z_memo', ignore_args=False)
    def covar_z(self, z) -> BlockDiagKuu:  # [D_in, 2M+1, 2M+1]
        return self.vff_kernel()  # noqa

    @cached(name='covar_zx_memo', ignore_args=False)
    def covar_zx(self, z, x):  # [D_in, 2M+1, N]
        return self.cross_kernel(x)




