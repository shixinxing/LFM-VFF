from typing import Union
import torch
from torch import Tensor
from torch.distributions import Normal

from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.settings import num_likelihood_samples
from linear_operator.operators import BlockDiagLinearOperator

from intdom_dgp.models.layer_vff_full import IntDom_GPLayer_Full
from intdom_dgp.models.layer_normalization import (
    Affine_SameAsInput_Normalization, Batch_MaxMin_Normalization, MaxMin_MovingAverage_LayerNormalizationMaxMin,
    BatchNorm1d_LayerNormalization, Tanh_Normalization
)


class Norm_IntDom_GPLayer(Module):
    """
    Normalization layer -> IDGP layer, similar to IDGP_LMC_Layer
    """
    def __init__(
            self,
            idgp_layer: IntDom_GPLayer_Full,
            normalization=None  # before the layer
    ):
        super().__init__()
        self.normalization = normalization
        self.idgp_layer = idgp_layer

        self.input_dims = idgp_layer.input_dims
        self.output_dims = idgp_layer.output_dims

    @property
    def device(self) -> torch.device:
        return self.idgp_layer.device

    def layer_normalize(self, f: Tensor) -> Tensor:
        if self.normalization is None:
            return f
        if isinstance(
                self.normalization,
                (MaxMin_MovingAverage_LayerNormalizationMaxMin, Batch_MaxMin_Normalization)
        ):
            f_nor = self.normalization(f, training=self.training)
        elif isinstance(
                self.normalization,
                (Affine_SameAsInput_Normalization, BatchNorm1d_LayerNormalization, Tanh_Normalization)
        ):
            f_nor = self.normalization(f)
        else:
            raise NotImplementedError
        return f_nor

    def propagate(
            self, inputs: Union[Tensor, MultivariateNormal], *other_inputs,
            full_cov=False, are_samples=False, prior=False, **kwargs
    ) -> tuple:
        """
        :return: (input_samples, normalized_input_samples: [S, N, D_in], output_dist Multitask MVN: [S, N, D_out])
        """
        deterministic_inputs = not are_samples
        if isinstance(inputs, MultitaskMultivariateNormal):
            if not full_cov:
                inputs = Normal(loc=inputs.mean, scale=inputs.variance.sqrt()).rsample()
            else:
                inputs = inputs.rsample()
            deterministic_inputs = False
        unnormalized_input_samples = inputs.detach().clone()

        D_in = inputs.size(-1)
        if len(other_inputs):
            processed_inputs = [
                inp.unsqueeze(0).expand(num_likelihood_samples.value(), *inp.size())
                for inp in other_inputs
            ]
            inputs = torch.cat([inputs] + processed_inputs, dim=-1)  # [S, N, t + t2]
            assert self.input_dims != inputs.size(-1)

        if not deterministic_inputs and self.normalization is not None:
            inputs = self.layer_normalize(inputs)
        normalized_input_samples = inputs.detach().clone()[..., :D_in]  # [(S), N, D_in]

        q_f = self.idgp_layer(inputs, full_cov=full_cov, prior=prior, **kwargs)  # get MVN [(S), D_out, N]
        # transform MVN to Multitask MVN
        mean, covar = q_f.mean.mT, q_f.lazy_covariance_matrix
        # Output dims are independent, q(f)'s covar can be BlockDiag or Diag
        covar = BlockDiagLinearOperator(covar, block_dim=-3)
        output_dist = MultitaskMultivariateNormal(mean, covar, interleaved=False)
        if deterministic_inputs:
            output_dist = output_dist.expand(torch.Size([num_likelihood_samples.value()]) + output_dist.batch_shape)

        return unnormalized_input_samples, normalized_input_samples, output_dist

    def forward(
            self,
            inputs: Union[Tensor, MultitaskMultivariateNormal], *other_inputs,
            full_cov=False, are_samples=False, prior=False, **kwargs
    ) -> Union[MultitaskMultivariateNormal, Tensor]:
        """
        inputs: unnormalized [N, D_in] data for the first layer
                  [S, N, D_in] Multitask MVN or data for the other layers
        other_inputs: usually concatenate x: [N, D_in]
        :return: Multitask MVN with mean [S, N, D_out], cov: BlockDiagLOP or DiagLOP (AddedDiag if with kernel noise)
                (may get un-blocked DiagLOP when `full_cov=False`)
        """
        _, _, output_dist = self.propagate(inputs, *other_inputs, full_cov=full_cov,
                                           are_samples=are_samples, prior=prior, **kwargs)
        return output_dist
