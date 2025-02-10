from typing import Optional

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor
from torch.nn import ModuleList
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from intdom_dgp_lmc.models.deep_iddgp_lmc import IDDeepGP_VFF_LMC
from dgp.models.model_utils import create_dim_list
from dlfm_vff.models.lfm_lmc_layer import LFM_RFF_LMC_Layer


class DeepLFM_RFFLMC(IDDeepGP_VFF_LMC):
    def __init__(
            self,
            input_dims, output_dims, hidden_dims, num_layers,
            M, nu,
            Z_init=None,  # [D_in, M, 1]
            num_samples=100,
            whitened: str = 'none',
            kernel_sharing_across_dims: bool = False,
            has_kernel_noise=True, learning_inducing_locations=True,
            integral_lower_bound: Optional[float] = None,
            with_concat=False, jitter_val: Optional[float] = None,
    ):
        input_dims_list, output_dims_list = create_dim_list(input_dims, output_dims, hidden_dims,
                                                            num_layers, with_concat)

        layers = ModuleList([])
        for i in range(num_layers):
            mean_x_type = 'linear-fix-identity'
            if i == num_layers - 1:  # The last layer has different settings
                mean_x_type = 'zero'  # zero mean
                has_kernel_noise = False  # no kernel noise

            if Z_init is None:  # TODO
                Z_init = torch.rand(input_dims_list[i], M, 1)
            else:
                Z_init = torch.as_tensor(Z_init, dtype=torch.get_default_dtype())

            lfm_lmc_layer = LFM_RFF_LMC_Layer(
                input_dims_list[i], output_dims_list[i],
                nu, Z_init, num_samples,
                mean_x_type=mean_x_type,
                whitened=whitened,
                kernel_sharing_across_dims=kernel_sharing_across_dims,
                integral_lower_bound=integral_lower_bound,
                has_kernel_noise=has_kernel_noise,
                learning_inducing_locations=learning_inducing_locations,
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
        """
        :return: [(output_dist, output_samples), ...]
        """
        res_all_layers = super().propagate(x, full_cov=full_cov, S=S, prior=prior)
        res_all_layers_new = []
        for input_nor, output_dist, output_sample in res_all_layers:
            res_all_layers_new.append((output_dist, output_sample))
        return res_all_layers_new

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        """
        re-organize the predictions at all layers
        :return: (Fmeans: [N, t]; Fvars: [N, t, t]; samples: [S, N, t])
        """
        res_all_layers = self.propagate(x, full_cov=full_cov, S=S, **kwargs)
        Fmeans, Fvars, Foutputs = [], [], []
        for output_dist, output_samples in res_all_layers:
            mixture_means, mixture_covars = self._compute_mixture_mean_covar(output_dist)
            Fmeans.append(mixture_means)  # [N, t]
            Fvars.append(mixture_covars)  # [N, t, t]

            Foutputs.append(output_samples)

        return Fmeans, Fvars, Foutputs


