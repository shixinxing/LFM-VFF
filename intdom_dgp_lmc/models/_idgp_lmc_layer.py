import torch
from torch import Tensor
from typing import Union

from gpytorch.module import Module
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.settings import num_likelihood_samples

from intdom_dgp.models._intdom_latent_gp import _Intdom_LatentGP
from intdom_dgp_lmc.models.layer_lmc import LMCLayer
from dgp.models.layer_gp import DSVI_GPLayer


class _IDGP_LMC_Layer(Module):
    """
    wrap up Latent Inter-domain GPs for one-dim inputs followed by LMC layer,
    consider layer composition and intermediate normalization before it in this class:
    normalization -> Latent GPs -> LMC
    """
    def __init__(
            self,
            latent_idgp: Union[_Intdom_LatentGP, DSVI_GPLayer],
            lmc_layer: LMCLayer,
            normalization=None,  # before the layer
    ):
        super().__init__()
        self.normalization = normalization
        self.latent_idgp = latent_idgp
        self.lmc_layer = lmc_layer

        self.input_dims = lmc_layer.input_dims
        self.output_dims = lmc_layer.output_dims

    @property
    def device(self):
        return self.latent_idgp.device

    def layer_normalize(self, f: Tensor) -> Tensor:
        if self.normalization is None:
            return f
        else:
            raise NotImplementedError  # need to implement in subclasses

    def forward(
            self,
            inputs: Tensor, *other_inputs,
            get_samples: bool = False,
            full_cov=False, are_first_layer_inputs=False, prior=False
    ) -> Union[MultitaskMultivariateNormal, Tensor]:
        """
        :param: inputs: unnormalized input points [N, D] for the first layer and [S, N, D] for intermediate layers
        :param: full_cov: if True, the covariance is KroneckerProductDiagLOP, otherwise BlockDiagLOP
        :return: Multitask MVN [S, N, t] or samples from it
        """
        if len(other_inputs):  # concatenate at intermediate layer, expand according to num_samples
            processed_inputs = [
                inp.unsqueeze(0).expand(num_likelihood_samples.value(), *inp.size())
                for inp in other_inputs
            ]
            inputs = torch.cat([inputs] + processed_inputs, dim=-1)  # [S, N, t + t2]
            if self.input_dims != inputs.size(-1):   # check input dims and num GP latents
                raise ValueError(f"The number of the new input dims {inputs.size(-1)} doesn't match!")

        if not are_first_layer_inputs and self.normalization is not None:
            inputs = self.layer_normalize(inputs)

        q_f = self.latent_idgp(inputs, full_cov=full_cov, prior=prior)
        if are_first_layer_inputs:
            # expand to S samples for the first-layer output Multitask MVN [N, t] whose batch_shape=[]
            q_f = q_f.expand(torch.Size([num_likelihood_samples.value()]) + q_f.batch_shape)
        q_g_or_samples = self.lmc_layer(q_f, get_samples=get_samples, full_cov=full_cov)
        return q_g_or_samples  # return Multitask MVN [S, N, t] or samples from it

    def propagate(
            self, inputs: torch.Tensor, *other_inputs,
            full_cov=False, are_first_layer_inputs=False, prior=False
    ) -> tuple:
        """
        :return: (normalized_input_samples, output_dist, output_samples)
        """
        D_in = inputs.size(-1)
        if len(other_inputs):
            processed_inputs = [
                inp.unsqueeze(0).expand(num_likelihood_samples.value(), *inp.size())
                for inp in other_inputs
            ]
            inputs = torch.cat([inputs] + processed_inputs, dim=-1)  # [S, N, t + t2]
            assert self.input_dims != inputs.size(-1)

        if not are_first_layer_inputs and self.normalization is not None:
            inputs = self.layer_normalize(inputs)
        normalized_input_samples = inputs.detach().clone()[..., :D_in]

        q_f = self.latent_idgp(inputs, full_cov=full_cov, prior=prior)
        if are_first_layer_inputs:
            # expand to S samples for the first-layer output Multitask MVN [N, t] whose batch_shape=[]
            q_f = q_f.expand(torch.Size([num_likelihood_samples.value()]) + q_f.batch_shape)
        output_dist = self.lmc_layer(q_f, get_samples=False, full_cov=full_cov)
        output_samples = self.lmc_layer(q_f, get_samples=True, full_cov=full_cov)
        return normalized_input_samples, output_dist, output_samples
