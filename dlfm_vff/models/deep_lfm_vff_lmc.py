from typing import Optional

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor
from torch.nn import ModuleList
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from intdom_dgp_lmc.models.deep_iddgp_lmc import IDDeepGP_VFF_LMC
from intdom_dgp_lmc.models.model_id_utils import create_normalization_from_dict
from dgp.models.model_utils import create_dim_list
from dlfm_vff.models.lfm_lmc_layer import LFM_VFF_LMC_Layer


class DeepLFM_VFF_LMC(IDDeepGP_VFF_LMC):
    def __init__(
            self,
            input_dims, output_dims, hidden_dims, num_layers,
            a, b, M, nu,
            whitened='none', mean_z_type='with-x', ode_type='ode1',
            hidden_normalization_info: Optional[dict] = None,
            kernel_sharing_across_dims: bool = False,
            has_kernel_noise=True,
            boundary_learnable=False,
            with_concat=False, jitter_val: Optional[float] = None,
    ):
        input_dims_list, output_dims_list = create_dim_list(input_dims, output_dims, hidden_dims,
                                                            num_layers, with_concat)
        self.hidden_normalization_info = hidden_normalization_info
        self.ode_type = ode_type

        layers = ModuleList([])
        for i in range(num_layers):
            if i == 0 or hidden_normalization_info is None:  # no normalization before the 1st layer
                layer_normalization = None
            else:
                layer_normalization = create_normalization_from_dict(
                    hidden_normalization_info, num_features=input_dims_list[i]
                )

            mean_x_type = 'linear-fix-identity'
            if i == num_layers - 1:  # The last layer has different settings
                mean_x_type = 'zero'  # zero mean
                mean_z_type = 'zero'
                has_kernel_noise = False  # no kernel noise
            if whitened != 'none':
                mean_z_type = 'zero'

            lfm_lmc_layer = LFM_VFF_LMC_Layer(
                input_dims_list[i], output_dims_list[i],
                a, b, M, nu,
                mean_x_type=mean_x_type, mean_z_type=mean_z_type,  ode_type=self.ode_type,
                whitened=whitened,
                normalization=layer_normalization,
                kernel_sharing_across_dims=kernel_sharing_across_dims,
                has_kernel_noise=has_kernel_noise,
                boundary_learnable=boundary_learnable,
                jitter_val=jitter_val
            )
            layers.append(lfm_lmc_layer)

        super(IDDeepGP_VFF_LMC, self).__init__()
        self.layers = layers
        self.likelihood = MultitaskGaussianLikelihood(output_dims, has_global_noise=(output_dims != 1))

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.with_concat = with_concat if num_layers != 1 else False

    def forward(self, x: Tensor, full_cov=False, S=1, prior=False) -> MultitaskMultivariateNormal:
        return super().forward(x, full_cov, S, prior)

    def propagate(self, x, full_cov=False, S=1, prior=False):
        return super().propagate(x, full_cov, S, prior)

    @torch.no_grad()
    def predict_measure(self, test_loader, y_mean=None, y_std=None, S=10):
        return super().predict_measure(test_loader, y_mean, y_std, S)


