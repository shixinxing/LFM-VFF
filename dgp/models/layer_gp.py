from typing import Optional, Union
import torch
from torch.distributions import Normal

from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.models import ApproximateGP
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.settings import num_likelihood_samples
from gpytorch import settings
from linear_operator.operators import BlockDiagLinearOperator

from dgp.inference.variational_strategy_dsvi_dgp import (
    DSVI_DGP_UnWhitenedVariationalStrategy, DSVI_DGP_WhitenedVariationalStrategy
)
from dgp.inference.chol_variational_dist import CholeskyVariationalDist_ZeroInit
from dgp.models.model_utils import _build_kernel_noise, _add_kernel_noise


class DSVI_GPLayer(DeepGPLayer):
    """
    Deep GP Layers with DSVI: [(S), N, D_in] -> [(S), N, D_out]
    Kernel's `batch_shape` controls whether to share the kernel across the output_dims:
    sharing: batch_shape = torch.Size([]) - default; otherwise batch_shape = torch.Size([D_out]) ;
    This layer assumes zero mean for inducing variables.
    """
    def __init__(
            self,
            input_dims, output_dims,
            mean_function: Mean,
            scaled_kernel: Kernel,  # kernel's batch_shape will indicate whether kernel shares across D_out
            Z: torch.Tensor,  # [M, D_in], shared across dims
            whitened: str = 'none',  # or 'cholesky'
            has_kernel_noise=True, learning_inducing_locations=True,
            jitter_val: Optional[float] = None
    ):
        self.input_dims, self.output_dims = input_dims, output_dims
        self.kernel_batch_shape = scaled_kernel.batch_shape  # can be torch.Size([]) or torch.Size([D_out])
        if len(self.kernel_batch_shape):
            assert self.kernel_batch_shape == torch.Size([output_dims])
        self.whitened = whitened
        self.has_kernel_noise = has_kernel_noise
        self.learning_inducing_locations = learning_inducing_locations

        variational_strategy = self.create_variational_strategy(Z, jitter_val)
        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = mean_function
        self.scaled_kernel = scaled_kernel
        self.build_kernel_noise()

    def create_variational_strategy(self, inducing_points, jitter_val):
        variational_distribution = CholeskyVariationalDist_ZeroInit(
            inducing_points.size(-2),
            batch_shape=torch.Size([self.output_dims]), mean_init_std=0.,
            force_mean_to_zero=False
        )
        if self.whitened == 'cholesky':
            variational_strategy = DSVI_DGP_WhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=self.learning_inducing_locations, jitter_val=jitter_val
            )
        elif self.whitened == 'none':
            variational_strategy = DSVI_DGP_UnWhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=self.learning_inducing_locations, jitter_val=jitter_val
            )
        else:
            raise ValueError(f"Unknown whitened {self.whitened}")
        return variational_strategy

    @property
    def device(self) -> torch.device:
        return self.scaled_kernel.device

    def build_kernel_noise(self):
        return _build_kernel_noise(self, noise_sharing_across_dims=True)

    def add_kernel_noise(self, covar_x_or_z, full_cov):
        return _add_kernel_noise(self, covar_x_or_z, full_cov)

    def forward(self, x, full_cov=False):  # used in VS.forward() or `prior=True`
        mean_x = self.mean_module(x)   # [(S), D_out, N+(M)]
        covar_x = self.scaled_kernel(x, diag=not full_cov)  # [(S), D_out, N+(M), N+(M)]
        # ⚠️ which means we also add kernel noise to Kzz
        covar_x = self.add_kernel_noise(covar_x, full_cov=full_cov)
        covar_x = covar_x.add_jitter(settings.cholesky_jitter.value(torch.get_default_dtype()))
        covar_x = covar_x.expand(*mean_x.shape[:-1], *covar_x.shape[-2:])  # in case the kernel shares across D_out
        return MultivariateNormal(mean_x, covar_x)

    def __call__(
            self, inputs: Union[torch.Tensor, MultitaskMultivariateNormal], *other_inputs,
            full_cov=False, are_samples=False,
            prior=False, **kwargs
    ) -> MultitaskMultivariateNormal:
        """
        :param inputs: [N, D_in] data for the first layer
                  [S, N, D_in] Multitask MVN or data for the other layers
        :param other_inputs: usually concatenate x: [N, D_in]
        :return: Multitask MVN with mean [S, N, D_out], cov: BlockDiagLOP or DiagLOP (AddedDiag if with kernel noise)
                (may get un-blocked DiagLOP when `full_cov=False`)
        """
        _, output_dist = self.propagate(inputs, *other_inputs, full_cov=full_cov,
                                        are_samples=are_samples, prior=prior, **kwargs)
        return output_dist

    def propagate(self, inputs: Union[torch.Tensor, MultitaskMultivariateNormal], *other_inputs,
                  full_cov=False, are_samples=False,
                  prior=False, **kwargs) -> tuple:
        """
        additionally save samples at intermediate layers
        :return: tuple(input samples [S, N, t], Multitask MVN: [S, N, t]);
        samples input to the first layer don't have dim `S`
        """
        deterministic_inputs = not are_samples
        if isinstance(inputs, MultitaskMultivariateNormal):
            if not full_cov:
                inputs = Normal(loc=inputs.mean, scale=inputs.variance.sqrt()).rsample()
            else:
                inputs = inputs.rsample()
            deterministic_inputs = False
        input_samples = inputs.detach().clone()
        # concatenate
        if len(other_inputs):
            processed_inputs = [
                inp.unsqueeze(0).expand(num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]
            inputs = torch.cat([inputs] + processed_inputs, dim=-1)

        inputs = inputs.unsqueeze(-3)
        inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])  # for each output GP

        output_dist = ApproximateGP.__call__(
            self, inputs, full_cov=full_cov, prior=prior, **kwargs
        )   # get MVN [(S), t, N]
        # transform MVN to Multitask MVN
        mean = output_dist.mean.transpose(-1, -2)  # [(S), N, t]
        covar = output_dist.lazy_covariance_matrix  # LOP([(S) t, N, N]) or DiagLOP([(S), t, N])
        covar = BlockDiagLinearOperator(covar, block_dim=-3)  # ⚠️ may get flattened Diag [tN, tN] if covar is DiagLOP
        output_dist = MultitaskMultivariateNormal(mean, covar, interleaved=False)

        if deterministic_inputs:
            output_dist = output_dist.expand(torch.Size([num_likelihood_samples.value()]) + output_dist.batch_shape)
        return input_samples, output_dist


