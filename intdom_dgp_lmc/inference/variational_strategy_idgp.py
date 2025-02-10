from typing import Optional
import torch
from torch import Tensor
from gpytorch.variational import VariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.variational import _VariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils import cached
from linear_operator.operators import LinearOperator, SumLinearOperator

from dgp.inference.variational_strategy_dsvi_dgp import _DSVI_DGP_VariationalStrategy
from intdom_dgp.inference.kl import blockdiag_kl_divergence_inv_quad_logdet, kl_divergence_whitened


class _IDGPVariationalStrategy(VariationalStrategy):
    """
    Base abstract class of variational strategies before LMC for all inter-domain GP models;
    :return: independent MVN with shape [(S), l=D, N]
    """
    def __init__(
            self,
            model: ApproximateGP,
            inducing_points: Tensor,  # [l, 2M+1, 1]
            variational_distribution: _VariationalDistribution,
            learning_inducing_locations=False,  # this can be true in analytic LFMs
            jitter_val: Optional[float] = None
    ):
        super().__init__(model, inducing_points, variational_distribution,
                         learn_inducing_locations=learning_inducing_locations,
                         jitter_val=jitter_val)

    def __call__(self, x: Tensor, prior=False, full_cov=False, **kwargs) -> MultivariateNormal:
        # get x: [(S), D=l, N, 1]
        if prior:
            return self.model.forward(x, output_form='Prior', full_cov=full_cov)
        # `full_cov` and kwargs finally go to `self.forward(...)` below
        return super().__call__(x, prior=False, full_cov=full_cov, **kwargs)

    def _get_matrix(self, x, full_cov=False):
        mean_x, mean_z, covar_x, covar_zz, covar_zx = self.model.forward(x, output_form='Full', full_cov=full_cov)
        return mean_x, mean_z, covar_x, covar_zz, covar_zx

    @staticmethod
    def compute_covar_return_posterior(
            middle_term, KzzInv_Kzx, Kxx, predictive_mean, full_cov=False
    ):
        return _DSVI_DGP_VariationalStrategy.compute_covar_return_posterior(
            middle_term, KzzInv_Kzx, Kxx, predictive_mean, full_cov=full_cov
        )


class IDGP_UnWhitenedVariationalStrategy(_IDGPVariationalStrategy):
    """
    discard whitening in order to utilize structure of K_uu
    """
    @property
    @cached(name='prior_distribution_memo')
    def prior_distribution(self) -> MultivariateNormal:  # for KL and q(v) initialization
        p = self.model.forward(None, output_form='VFF')  # [(l), 2M+1]
        num_latents = self.model.num_latents
        return p.expand(batch_size=torch.Size([num_latents]))  # [l, 2M+1]

    def kl_divergence(self) -> Tensor:  # _VariationalStrategy.kl_divergence() doesn't utilize Kvv's structure
        # MultivariateNormal mean: [l, N], covar: [l, N, N]
        q = self.variational_distribution
        p = self.prior_distribution
        assert q.batch_shape == p.batch_shape
        res = blockdiag_kl_divergence_inv_quad_logdet(q, p)
        assert res.shape == torch.Size([self.model.num_latents])
        return res

    def forward(
            self, x: Tensor,  # [l, N, 1]
            inducing_points: Tensor,  # [l, 2M+1, 1], Z will broadcast to X when call `_variational_strategy`
            inducing_values: Tensor,  # [l, 2M+1]
            variational_inducing_covar: Optional[LinearOperator] = None,  # [t, N, N]
            full_cov: bool = False
    ) -> MultivariateNormal:
        mean_x, mean_z, Kxx, Kvv, Kvf = self._get_matrix(x, full_cov=full_cov)
        Kvv = Kvv.add_jitter(self.jitter_val)  # BlockDiagKuu + jitter, may not have batch dim
        Kxx = Kxx.add_jitter(self.jitter_val)  # Kxx can be DiagLOP

        # Compute K_ZZ^{-1} K_ZX utilizing the covariance structure
        KzzInv_Kzx = torch.linalg.solve(Kvv, Kvf)  # overridden by LOP.solve()

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1} (m - \mu_Z) + \mu_X, \mu_Z != 0
        mean_diff = (inducing_values - mean_z).unsqueeze(-1)
        predictive_mean = (KzzInv_Kzx.transpose(-1, -2) @ mean_diff).squeeze(-1) + mean_x

        # Compute the covariance of q(f)
        # K_XX + K_XZ K_ZZ^{-1} (S - K_ZZ) K_ZZ^{-1} K_ZX
        middle_term = Kvv.mul(-1)  # AddedDiag LOP, no Low-Rank
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        return self.compute_covar_return_posterior(
            middle_term, KzzInv_Kzx, Kxx, predictive_mean, full_cov=full_cov
        )


class IDGP_WhitenedVariationalStrategy(_IDGPVariationalStrategy):
    """
    use conventional whitening by Cholesky decomposition;
    p(u') = N(0, I), q(u') = N(m', S'), u = Lu' + m_z
    """
    def kl_divergence(self) -> Tensor:
        res = kl_divergence_whitened(self.variational_distribution)
        assert res.shape == torch.Size([self.model.num_latents])
        return res

    def forward(
            self, x: Tensor,
            inducing_points: Tensor,
            inducing_values: Tensor,
            variational_inducing_covar: Optional[LinearOperator] = None,
            full_cov: bool = False,
            **kwargs
    ) -> MultivariateNormal:
        test_mean, _, data_data_covar, induc_induc_covar, induc_data_covar = self._get_matrix(x, full_cov=full_cov)
        data_data_covar = data_data_covar.add_jitter(self.jitter_val)
        # Whitening uses Cholesky decomposition, essentially using torch.linalg.cholesky_ex()
        L = self._cholesky_factor(induc_induc_covar.add_jitter(self.jitter_val))
        interp_term = L.solve(induc_data_covar.to_dense()).to(x.dtype)  # Kvv^{-1/2}Kvf

        predictive_mean = (
                (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1)
                + test_mean
        )
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        return self.compute_covar_return_posterior(
            middle_term, interp_term, data_data_covar, predictive_mean, full_cov=full_cov
        )


