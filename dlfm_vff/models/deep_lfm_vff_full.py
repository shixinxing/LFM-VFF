from typing import Optional
import numpy as np
import torch
from torch.nn import ModuleList
from torch.distributions import Normal

from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.settings import num_likelihood_samples
from linear_operator.operators import DiagLinearOperator

from dgp.models.model_utils import create_dim_list, predict_mixture_mean_and_covar
from dgp.models.deep_gp import DSVI_DeepGP
from intdom_dgp_lmc.models.model_id_utils import create_normalization_from_dict
from intdom_dgp.models.layer_vff_normalization import Norm_IntDom_GPLayer
from dlfm_vff.models.layer_lfm_vff_full import LFM_Layer_Full


class DeepLFM_VFF_Full(DSVI_DeepGP):
    def __init__(
            self,
            input_dims, output_dims, hidden_dims, num_layers,
            a, b, M, nu,
            X_running: Optional[np.ndarray],  # used to determine the mean function at hidden layers
            whitened: str = 'cholesky', mean_z_type='zero', ode_type='ode1',
            hidden_normalization_info: Optional[dict] = None,
            kernel_sharing_across_output_dims=True, has_kernel_noise=True, boundary_learnable=False,
            with_concat=False, jitter_val=None
    ):
        input_dims_list, output_dims_list = create_dim_list(input_dims, output_dims, hidden_dims,
                                                            num_layers, with_concat)
        self.hidden_normalization_info = hidden_normalization_info
        self.ode_type = ode_type

        layers = ModuleList([])
        for i in range(num_layers):
            if i == 0 or hidden_normalization_info is None:
                layer_normalization = None
            else:
                layer_normalization = create_normalization_from_dict(
                    hidden_normalization_info, num_features=input_dims_list[i]
                )

            mean_x_type = 'linear'
            if i == num_layers - 1:
                mean_x_type, mean_z_type, has_kernel_noise = 'zero', 'zero', False
            if whitened != 'none':
                mean_z_type = 'zero'

            dim_in, dim_out = input_dims_list[i], output_dims_list[i]
            lfm_layer = LFM_Layer_Full(
                dim_in, dim_out, a, b, M, nu,
                mean_x_type, mean_z_type, ode_type, whitened,
                kernel_sharing_across_output_dims, has_kernel_noise, boundary_learnable, jitter_val=jitter_val
            )
            # adjust weights of the hidden mean function
            if i != num_layers - 1:
                weights = lfm_layer.mean_module.weights
                weights.requires_grad = False
                if dim_in == dim_out:
                    lfm_layer.mean_module.initialize(weights=torch.eye(dim_in).unsqueeze(-1))
                else:
                    if dim_in > dim_out:
                        _, _, VT = np.linalg.svd(X_running, full_matrices=False)  # VT: [..., D_in]
                        W = VT[:dim_out, :].T  # [D_in, D_out]
                    else:
                        W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, dim_out - dim_in))], axis=1)
                    W_tensor = torch.as_tensor(W, dtype=weights.dtype, device=weights.device)
                    lfm_layer.mean_module.initialize(weights=W_tensor.mT.unsqueeze(-1))
                    X_running = X_running.dot(W)

            layer = Norm_IntDom_GPLayer(
                lfm_layer, normalization=layer_normalization
            )
            layers.append(layer)

        super(DSVI_DeepGP, self).__init__()  # use DeepGP to initialize
        self.layers = layers
        self.likelihood = MultitaskGaussianLikelihood(output_dims, has_global_noise=(output_dims != 1))

        self.input_dims, self.output_dims, self.hidden_dims = input_dims, output_dims, hidden_dims
        self.num_layers = num_layers
        self.with_concat = with_concat

    def forward(self, x, full_cov=False, S=1, **kwargs):
        return super().forward(x, full_cov=full_cov, S=S, **kwargs)

    def propagate(self, x, full_cov=False, S=1, **kwargs):  # override
        """
        :return: [(unnor_input_samples, normed_input_samples, output_dist), (...), ...] with shape [S, N, t]
        """
        inputs = x.clone() if self.with_concat else None
        res_all_layers = []
        with num_likelihood_samples(S):
            for i, layer in enumerate(self.layers):
                if self.num_layers == 1:
                    res_all_layers.append(layer.propagate(x, full_cov=full_cov, **kwargs))
                    return res_all_layers
                elif i == 0:
                    unnor_input_samples, normed_input_samples, x = layer.propagate(x, full_cov=full_cov, **kwargs)
                else:
                    pre_x = (x, inputs) if self.with_concat else (x,)
                    unnor_input_samples, normed_input_samples, x = layer.propagate(*pre_x, full_cov=full_cov, **kwargs)

                res_all_layers.append((unnor_input_samples, normed_input_samples, x))
        return res_all_layers

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):  # override
        """
        re-organize the predictions at all layers and compute the mixture mean and covariance
        :return: (normalized inputs Fs: [S, N, t]; Fmeans: [N, t]; Fvars: [N, t, t]; output samples: [S, N, t])
        """
        res_all_layers = self.propagate(x, full_cov=full_cov, S=S, **kwargs)
        Finputs, Fmeans, Fvars, Fs = [], [], [], []
        for unnor_input_samples, nor_input_samples, output_dist in res_all_layers:
            Finputs.append(nor_input_samples)  # [S, N, t]

            component_means = output_dist.mean.transpose(-2, -3)  # [N, S, t]
            component_covs = output_dist.variance.transpose(-2, -3)  # [N, S, t]
            component_covs = DiagLinearOperator(component_covs)  # [N, S, t, t]
            mixture_means, mixture_covars = predict_mixture_mean_and_covar(component_means, component_covs)

            Fmeans.append(mixture_means)  # [N, t]
            Fvars.append(mixture_covars)  # [N, t, t]
            Fs.append(unnor_input_samples)
        if full_cov:
            samples_final_layer = output_dist.rsample()
        else:
            samples_final_layer = Normal(loc=output_dist.mean, scale=output_dist.variance.sqrt()).rsample()
        del Fs[0]
        Fs.append(samples_final_layer)
        return Finputs, Fmeans, Fvars, Fs




