from typing import Optional
import torch
from torch import Tensor

from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy, _VariationalDistribution
from gpytorch.settings import _linalg_dtype_cholesky
from linear_operator import LinearOperator
from linear_operator.operators import SumLinearOperator, MatmulLinearOperator, DiagLinearOperator


class IDGP_Whitened_VariationalStrategy_Full(VariationalStrategy):
    def __init__(
            self,
            model: ApproximateGP,
            inducing_points: Tensor,  # [D_in * (2M + 1), 1]
            variational_distribution: _VariationalDistribution,  # q(u): [D_out, D_in * (2M + 1)]
            jitter_val: Optional[float] = None
    ):
        super().__init__(model, inducing_points, variational_distribution,
                         learn_inducing_locations=False, jitter_val=jitter_val)

    def __call__(self, x: Tensor, prior=False, full_cov=False,
                 **kwargs) -> MultivariateNormal:  # x: [(S), D_in, N, 1]
        if prior:
            return self.model.foward(x, output_form='Prior', full_cov=full_cov)
        return super().__call__(x, prior=False, full_cov=full_cov, **kwargs)

    @property
    def prior_distribution(self) -> MultivariateNormal:
        prior_dist = super().prior_distribution
        assert prior_dist.batch_shape == torch.Size([self.model.output_dims])
        return prior_dist

    def kl_divergence(self) -> Tensor:
        res = super().kl_divergence()
        assert res.shape == torch.Size([self.model.output_dims])
        return res

    def forward(
            self, x: Tensor,  # [(S), D_in, N, 1]
            inducing_points: Tensor,  # [..., D_in * (2M +1), 1]
            inducing_values: Tensor,  # [D_out, D_in * (2M+1)]
            variational_inducing_covar: Optional[LinearOperator] = None,   # [D_out, D_in * (2M+1), D_in * (2M+1)]
            full_cov: bool = False,
            **kwargs
    ):
        """
        m(x): [(S), D_out, N], m(z): [D_out, D_in * (2M+1)]
        Kxx: [(S), D_out, D_in, N, N], Kzz: [D_in, 2M+1, 2M+1], Kzx: [(S), D_out, D_in, 2M+1, N]
        """
        mean_x, mean_z, Kxx, Kzz, Kzx = self.model.forward(x, output_form='Full', full_cov=full_cov)
        Kxx, Kzz = Kxx.add_jitter(self.jitter_val), Kzz.add_jitter(self.jitter_val)
        L = self._cholesky_factor(Kzz)  # [D_in, 2M, 2M]
        # Kzz^{-1/2}Kzx: [D_in, 2M, 2M] \times [D_out, D_in, 2M, N] -> [D_out, D_in, 2M, N]
        KzzInv_Kzx = L.solve(Kzx.to_dense().type(_linalg_dtype_cholesky.value())).to(x.dtype)
        KzzInv_Kzx = KzzInv_Kzx.reshape(*KzzInv_Kzx.shape[:-3], -1, x.size(-2))  # [D_out, D_in*(2M+1), N]

        predictive_mean = (KzzInv_Kzx.mT @ inducing_values.unsqueeze(-1)).squeeze(-1) + mean_x  # [D_out, N]

        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)  # I: [D_out, D_in*(2M+1), D_in*(2M+1)]
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if full_cov:
            predictive_covar = SumLinearOperator(
                Kxx.sum(dim=-3), MatmulLinearOperator(KzzInv_Kzx.mT, middle_term @ KzzInv_Kzx)
            )
        else:
            assert isinstance(Kxx, DiagLinearOperator)
            S_K_Kzx = middle_term @ KzzInv_Kzx
            diag_part = torch.sum(KzzInv_Kzx * S_K_Kzx, dim=-2)
            Kxx_diag = DiagLinearOperator(Kxx.diagonal(dim1=-1, dim2=-2).sum(-2))  # [D_out, N, N]
            predictive_covar = Kxx_diag + DiagLinearOperator(diag_part)  # Kxx is also Diag LOP
        # Return MVN with mean [D_out, N], cov [D_out, N, N] or DiagLOP([D_out, N])
        return MultivariateNormal(predictive_mean, predictive_covar)




