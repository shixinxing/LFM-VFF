from typing import Union
import torch
from torch.distributions import Normal

from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch import settings

from linear_operator.operators import (
    RootLinearOperator, KroneckerProductLinearOperator, BlockDiagLinearOperator, to_linear_operator
)


class LMCLayer(Module):
    def __init__(
            self,
            input_dims: int, output_dims: int,
            init_scheme='identity'
    ):
        super(LMCLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.init_scheme = init_scheme
        if init_scheme == 'rand':
            lmc_coefficients = torch.rand(input_dims, output_dims)
        elif init_scheme == 'randn':
            lmc_coefficients = torch.randn(input_dims, output_dims)
        elif init_scheme == 'identity':
            if input_dims == output_dims:
                lmc_coefficients = torch.eye(input_dims)
            else:
                lmc_coefficients = torch.rand(input_dims, output_dims)
        else:
            raise NotImplementedError(f"{init_scheme} not implemented.")
        self.register_parameter('lmc_coefficients', torch.nn.Parameter(lmc_coefficients))

    def forward(
            self,
            input_dist: MultivariateNormal,
            get_samples: bool = True,
            full_cov: bool = False,
    ) -> Union[MultivariateNormal, torch.Tensor]:
        """
        :param input_dist: MVN q(f) with mean [S, l, N], cov [S, l, N, N]
        :param get_samples: call `_sample`, otherwise the other
        :param full_cov: whether to consider correlations along dim `N`
        :return: Multitask MVN q(g) with mean [S, N, t], cov [S, t*N, t*N] or samples from it
        """
        if get_samples:
            return self._sample(input_dist, full_cov=full_cov)
        else:
            return self._density(input_dist, full_cov=full_cov)

    def _sample(self, input_dist: MultivariateNormal, full_cov: bool = False):
        """
        :param input_dist: MVN q(f) [S, l, N], cov [S, l, N, N] or DiagLOP
        :param full_cov: whether to consider correlations along dim `N`
        :return: samples from Multitask MVN q(g) with mean [S, N, t]
        """
        if not full_cov:
            input_tensors = Normal(loc=input_dist.mean, scale=input_dist.variance.sqrt()).rsample()
        else:
            input_tensors = input_dist.rsample(sample_shape=torch.Size([]))  # Samples before LMC have shape [S, l, N]
        output_samples = input_tensors.mT @ self.lmc_coefficients
        # Note we keep the correlation along output-dim `t` by the matmul above.
        return output_samples

    def _density(self, input_dist: MultivariateNormal, full_cov: bool = False):
        """
        :param input_dist: MVN with mean [S, l, N], cov [S, l, N, N] or DiagLOP
        :return: Multitask MVN q(g) with mean [S, N, t], cov [S, N, t*N, t*N] after LMC
        """
        latent_mean = input_dist.mean
        mean = latent_mean.mT @ self.lmc_coefficients

        latent_covar = input_dist.lazy_covariance_matrix  # independent [S, l, N, N]
        if not full_cov:
            latent_covar_diag = latent_covar.diagonal(dim1=-2, dim2=-1).mT  # [S, l, N] -> [S, N, l]
            covar_out = (  # [S, N, t, t], W^T @ diag @ W
                (self.lmc_coefficients.mT * latent_covar_diag.unsqueeze(-2)) @ self.lmc_coefficients
            )
            covar = BlockDiagLinearOperator(to_linear_operator(covar_out), block_dim=-3)
            return MultitaskMultivariateNormal(mean, covar, interleaved=True)
        else:
            lmc_factor = RootLinearOperator(self.lmc_coefficients.unsqueeze(-1))
            covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(-3)
            covar = covar.add_jitter(settings.variational_cholesky_jitter.value(self.lmc_coefficients.dtype))  # make pd
            return MultitaskMultivariateNormal(mean, covar)
