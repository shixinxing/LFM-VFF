from typing import Optional
import torch
from torch import Tensor

from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import _linalg_dtype_cholesky
from gpytorch.utils import cached
from linear_operator.operators import (
    LinearOperator, SumLinearOperator, MatmulLinearOperator, DiagLinearOperator
)


class _DSVI_DGP_VariationalStrategy(VariationalStrategy):
    """
    base DSVI variational strategy with option `full_cov` and `prior`;
    especially suited for non-inter-domain models, which means Kzx, Kzz have the same computation as Kxx
    """
    def __call__(self, x: Tensor, prior=False, full_cov=False, **kwargs) -> MultivariateNormal:
        if prior:
            return self.model.forward(x, full_cov=full_cov)
        return super().__call__(x, prior=False, full_cov=full_cov, **kwargs)

    def _get_matrix(self, x, inducing_points: Tensor, full_cov=False):
        # separately get Kzz to prevent z's expansion along dim `S`, otherwise Kzz will have extra batch dim `S`
        prior_x = self.model.forward(x, full_cov=full_cov)
        mean_x = prior_x.mean
        Kxx = prior_x.lazy_covariance_matrix

        prior_z = self.model.forward(inducing_points, full_cov=True)
        mean_z = prior_z.mean
        Kzz = prior_z.lazy_covariance_matrix
        Kzx = self.model.scaled_kernel(inducing_points, x)

        return mean_x, mean_z, Kxx, Kzz, Kzx

    def _get_matrix_full(self, x, inducing_points: Tensor, full_cov=False):
        # in this case, x and z have been expanded along batch dim in _vs.__call__;
        full_inputs = torch.cat([x, inducing_points], dim=-2)
        full_outputs = self.model.forward(full_inputs, full_cov=True)
        full_mean = full_outputs.mean
        full_covar = full_outputs.lazy_covariance_matrix

        num_inducing = inducing_points.size(-2)
        mean_x = full_mean[..., :-num_inducing]
        mean_z = full_mean[..., -num_inducing:]

        Kxx = full_covar[..., :-num_inducing, :-num_inducing]
        Kzz = full_covar[..., -num_inducing:, -num_inducing:]
        Kzx = full_covar[..., -num_inducing:, :-num_inducing]

        return mean_x, mean_z, Kxx, Kzz, Kzx

    @staticmethod
    def compute_covar_return_posterior(
            middle_term, KzzInv_Kzx, Kxx, predictive_mean, full_cov=False
    ):
        if full_cov:
            predictive_covar = SumLinearOperator(
                Kxx,
                MatmulLinearOperator(KzzInv_Kzx.transpose(-1, -2), middle_term @ KzzInv_Kzx),
            )
        else:
            assert isinstance(Kxx, DiagLinearOperator)
            S_K_Kzx = middle_term @ KzzInv_Kzx
            diag_part = torch.sum(KzzInv_Kzx * S_K_Kzx, dim=-2)
            predictive_covar = Kxx + DiagLinearOperator(diag_part)  # Kxx is also Diag LOP
        # Return MVN with mean [t, N], cov [t, N, N] or DiagLOP([t, N])
        return MultivariateNormal(predictive_mean, predictive_covar)


class DSVI_DGP_UnWhitenedVariationalStrategy(_DSVI_DGP_VariationalStrategy):
    """
    Un-whitened Deep GPs, conform to the DSVI paper; assume zero prior mean for Z;
    :return: MVN with shape [(S), t=D_out, N]
    """
    @property
    @cached(name='prior_distribution_memo')
    def prior_distribution(self) -> MultivariateNormal:
        p = self.model.forward(self.inducing_points, full_cov=True)
        output_dims = self.model.output_dims
        return p.expand(batch_size=torch.Size([output_dims]))  # expand along D_out, get [D_out, M]

    def kl_divergence(self) -> Tensor:
        q = self.variational_distribution
        p = self.prior_distribution
        assert q.batch_shape == p.batch_shape
        kl = torch.distributions.kl.kl_divergence(q, p)
        assert kl.shape == torch.Size([self.model.output_dims])
        return kl

    def forward(
            self,
            x: Tensor,  # [S, N, D_in]
            inducing_points: Tensor,  # [S, M, D_in], has been broadcast to x in `_Variational_Strategy`
            inducing_values: Tensor,  # [D_out, M]
            variational_inducing_covar: Optional[LinearOperator] = None,  # q(u), [D_out, M, M]
            full_cov: bool = False,
            **kwargs
    ):
        mean_x, mean_z, Kxx, Kzz, Kzx = self._get_matrix(
            x, self.inducing_points, full_cov=full_cov  # to save computation by not using the broadcast Z
        )
        Kzz = Kzz.add_jitter(self.jitter_val)
        Kxx = Kxx.add_jitter(self.jitter_val)

        L = self._cholesky_factor(Kzz)
        KzzInv_Kzx = torch.cholesky_solve(Kzx.to_dense(), L.to_dense(), upper=False)  # L.to_dense? L.solve?

        mean_diff = inducing_values - mean_z
        predictive_mean = (KzzInv_Kzx.transpose(-1, -2) @ mean_diff.unsqueeze(-1)).squeeze(-1) + mean_x
        middle_term = Kzz.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        return self.compute_covar_return_posterior(middle_term, KzzInv_Kzx, Kxx, predictive_mean, full_cov=full_cov)


class DSVI_DGP_WhitenedVariationalStrategy(_DSVI_DGP_VariationalStrategy):
    """
    adapted from gpytorch.variational.VariationalStrategy with Whitening;
    :return: MVN with shape [(S), t=D_out, N]
    """
    def forward(
            self,
            x: Tensor,  # [S, N, D_in]
            inducing_points: Tensor,  # [S, M, D_in]
            inducing_values: Tensor,  # [D_out, M]
            variational_inducing_covar: Optional[LinearOperator] = None,  # [D_out, M, M]
            full_cov: bool = False,
            **kwargs
    ) -> MultivariateNormal:
        test_mean, _, data_data_covar, induc_induc_covar, induc_data_covar = self._get_matrix(
            x, self.inducing_points, full_cov=full_cov
        )  # use `self.inducing_points` instead of `inducing_points` to prevent expansion along `S`
        data_data_covar = data_data_covar.add_jitter(self.jitter_val)
        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        L = self._cholesky_factor(induc_induc_covar.add_jitter(self.jitter_val))
        interp_term = L.solve(induc_data_covar.to_dense().type(_linalg_dtype_cholesky.value())).to(x.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} m' + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)  # -I
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)  # S - I

        MVN = self.compute_covar_return_posterior(  # noqa
            middle_term, interp_term, data_data_covar,
            predictive_mean, full_cov=full_cov
        )

        return MVN


