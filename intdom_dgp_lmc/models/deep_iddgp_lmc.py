from typing import Optional
import torch
from torch import Tensor
from torch.nn import ModuleList

from gpytorch.models.deep_gps import DeepGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.settings import num_likelihood_samples

from dgp.models.model_utils import create_dim_list, unsqueeze_y_stat, predict_mixture_mean_and_covar
from dgp.models.deep_gp import DSVI_DeepGP

from intdom_dgp_lmc.models.layer_idgp_lmc import IDGP_LMC_Layer
from intdom_dgp_lmc.models.model_id_utils import create_normalization_from_dict
from intdom_dgp.linear_op_utils.extract_diag_block import extract_diag_block


class IDDeepGP_VFF_LMC(DeepGP):
    def __init__(
            self,
            input_dims, output_dims, hidden_dims, num_layers,
            a, b, M, nu,
            whitened='none', mean_z_type='with-x',  # may be forced to 'zero' if `whitened` != `none`
            hidden_normalization_info: Optional[dict] = None,  # {'type', 'preprocess_a', 'preprocess_b'}
            kernel_sharing_across_dims: bool = False,
            has_kernel_noise=True, boundary_learnable=False,
            with_concat=False, jitter_val: Optional[float] = None,
    ):
        input_dims_list, output_dims_list = create_dim_list(input_dims, output_dims, hidden_dims,
                                                            num_layers, with_concat)
        self.hidden_normalization_info = hidden_normalization_info

        layers = ModuleList([])
        for i in range(num_layers):
            # Intermediate Layer Normalization
            if i == 0 or hidden_normalization_info is None:  # no need for normalization before the 1st layer
                layer_normalization = None
            else:
                layer_normalization = create_normalization_from_dict(
                    hidden_normalization_info, num_features=input_dims_list[i]
                )

            mean_x_type = 'linear-fix-identity'
            if i == num_layers - 1:  # The last layer has different settings
                mean_x_type = 'zero'   # zero mean
                mean_z_type = 'zero'
                has_kernel_noise = False  # no kernel noise
            if whitened != 'none':
                mean_z_type = 'zero'

            idgp_lmc_layer = IDGP_LMC_Layer(
                input_dims_list[i], output_dims_list[i],
                a, b, M, nu,
                mean_x_type=mean_x_type, mean_z_type=mean_z_type,
                whitened=whitened,
                normalization=layer_normalization,
                kernel_sharing_across_dims=kernel_sharing_across_dims,
                has_kernel_noise=has_kernel_noise,
                boundary_learnable=boundary_learnable,
                jitter_val=jitter_val
            )
            layers.append(idgp_lmc_layer)

        super().__init__()
        self.layers = layers
        self.likelihood = MultitaskGaussianLikelihood(output_dims, has_global_noise=(output_dims != 1))

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.with_concat = with_concat if num_layers != 1 else False

    @property
    def device(self) -> torch.device:
        return self.layers[0].device

    def forward(self, x: Tensor, full_cov=False, S=1, prior=False) -> MultitaskMultivariateNormal:
        inputs = x.clone() if self.with_concat else None
        with num_likelihood_samples(S):
            for i, layer in enumerate(self.layers):
                if self.num_layers == 1:
                    return layer(x, get_samples=False, full_cov=full_cov, are_first_layer_inputs=True, prior=prior)
                elif i == 0:  # first layer
                    x = layer(x, get_samples=True, full_cov=full_cov, are_first_layer_inputs=True, prior=prior)
                elif i == self.num_layers - 1:  # final layer produce a distribution rather than samples
                    pre_x = (x, inputs) if self.with_concat else (x,)
                    x = layer(*pre_x, get_samples=False, full_cov=full_cov, are_first_layer_inputs=False, prior=prior)
                else:
                    pre_x = (x, inputs) if self.with_concat else (x,)
                    x = layer(*pre_x, get_samples=True, full_cov=full_cov, are_first_layer_inputs=False, prior=prior)
        return x

    @torch.no_grad()
    def _get_sampled_mean_and_covar(self, x_batch, y_mean=None, y_std=None, full_cov=False, S=10):
        """
        :return: recovered component mean [N, S, t] and covar [N, S, t, t] of the Gaussian mixtures
        Note that full_cov will affect the intermediate-layer sampling,
        thus eventually affecting the output distributions.
        We have to remove the correlation along dim `N` in this method, while Dims `t` are Dependent due to LMC.
        """
        y_mean, y_std = unsqueeze_y_stat(y_mean, y_std, self.output_dims, self.device)
        # todo other likelihoods
        preds: MultitaskMultivariateNormal = self.likelihood(self(x_batch.to(self.device), full_cov=full_cov, S=S))  # noqa

        component_mus = preds.mean * y_std + y_mean
        component_mus = component_mus.transpose(-2, -3)  # [N, S, t]
        assert component_mus.shape == (x_batch.size(0), S, self.output_dims)

        # for ID-DGP's covar (interleaved):
        # BlockDiagLOP [S, Nt, Nt] (actually becomes AddedDiag = KronProdDiagLOP + BlockDiagLOP after likelihood noise)
        # or DenseLOP if full_cov
        assert preds._interleaved
        covs_batch = preds.lazy_covariance_matrix  # [S, Nt, Nt]
        base_blocks = extract_diag_block(covs_batch, num_task=self.output_dims, interleaved=True)  # [S, N, t, t]
        recover_component_covars = base_blocks * (y_std.mT @ y_std)  # [S, N, t, t]
        component_covars = recover_component_covars.transpose(-3, -4)  # [N, S, t, t]
        return component_mus, component_covars

    @torch.no_grad()
    def predict_mean_and_var(self, x_batch, y_mean=None, y_std=None, full_cov=False, S=10):
        """
        :return: the mean and var of r.v. from Gaussian mixtures  mean: [N, t]; covar: [N, t, t]
        """
        component_mus, component_covars = self._get_sampled_mean_and_covar(
            x_batch, y_mean, y_std, full_cov=full_cov, S=S
        )
        return predict_mixture_mean_and_covar(component_mus, component_covars)

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_loader, y_mean=None, y_std=None, loader_has_y=True, full_cov=False, S=10
    ):
        return DSVI_DeepGP.predict_mean_and_var_loader(
            self, test_loader, y_mean=y_mean, y_std=y_std, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )

    @torch.no_grad()
    def predict_measure(self, test_loader, y_mean=None, y_std=None, S=10):
        y_mean, y_std = unsqueeze_y_stat(y_mean, y_std, self.output_dims, self.device)
        num_samples = torch.as_tensor(S, dtype=torch.get_default_dtype(), device=self.device)

        square_errors = []
        log_likelihoods = []
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            y_recover = y_batch * y_std + y_mean

            component_mus, component_covars = self._get_sampled_mean_and_covar(
                x_batch, y_mean, y_std, S=S, full_cov=False  # remove correlation along dim `N`
            )
            pred_dist = MultivariateNormal(component_mus, component_covars)  # [N, S, D_out] with batch_shape: [N, S]

            mean_average = component_mus.mean(-2)  # [N, D_out]
            square_errors.append(torch.square(y_recover - mean_average).sum(-1))

            log_q_y = pred_dist.log_prob(y_recover.unsqueeze(-2))  # [N, S]
            assert log_q_y.shape == (y_recover.shape[0], S)
            log_q_y = torch.logsumexp(log_q_y - torch.log(num_samples), dim=-1)  # [N], \log \sum 1/S q_i(y)
            log_likelihoods.append(log_q_y)

        square_errors = torch.cat(square_errors, dim=-1)
        log_likelihoods = torch.cat(log_likelihoods, dim=-1)
        return square_errors.mean(dim=-1).sqrt(), log_likelihoods.mean(dim=-1)

    def propagate(self, x, full_cov=False, S=1, prior=False):
        """
        :return: [(normalized_input_samples, output_dist, output_samples), ...]
        """
        inputs = x.clone() if self.with_concat else False
        res_all_layers = []
        with (num_likelihood_samples(S)):
            for i, layer in enumerate(self.layers):
                if self.num_layers == 1:
                    res_all_layers.append(layer.propagate(x, full_cov=full_cov,
                                                          are_first_layer_inputs=True, prior=prior))
                    return res_all_layers
                elif i == 0:  # first layer
                    normal_in_samples, output_dist, x = layer.propagate(x, full_cov=full_cov,
                                                                        are_first_layer_inputs=True, prior=prior)
                else:
                    pre_x = (x, inputs) if self.with_concat else (x,)
                    normal_in_samples, output_dist, x = layer.propagate(*pre_x, full_cov=full_cov,
                                                                        are_first_layer_inputs=False, prior=prior)
                res_all_layers.append((normal_in_samples, output_dist, x))
        return res_all_layers

    @staticmethod
    def _compute_mixture_mean_covar(output_dist: MultitaskMultivariateNormal):
        component_means = output_dist.mean.transpose(-2, -3)  # [N, S, t]
        base_blocks = extract_diag_block(
            output_dist.lazy_covariance_matrix, num_task=component_means.size(-1), interleaved=True
        )
        component_covs = base_blocks.transpose(-3, -4)
        mixture_means, mixture_covars = predict_mixture_mean_and_covar(component_means, component_covs)
        return mixture_means, mixture_covars

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        """
        re-organize the predictions at all layers and compute the mixture mean and covariance
        :return: (normalized inputs Fs: [S, N, t]; Fmeans: [N, t]; Fvars: [N, t, t]; samples: [S, N, t])
        """
        res_all_layers = self.propagate(x, full_cov=full_cov, S=S, **kwargs)
        Finputs, Fmeans, Fvars, Foutputs = [], [], [], []
        for input_samples, output_dist, output_samples in res_all_layers:
            Finputs.append(input_samples)   # [S, N, t]

            mixture_means, mixture_covars = self._compute_mixture_mean_covar(output_dist)
            Fmeans.append(mixture_means)  # [N, t]
            Fvars.append(mixture_covars)  # [N, t, t]

            Foutputs.append(output_samples)

        return Finputs, Fmeans, Fvars, Foutputs

