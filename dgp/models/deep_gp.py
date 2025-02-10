from typing import Optional

import numpy as np
import torch
from torch.nn import ModuleList
from torch.distributions import Normal

from gpytorch.models.deep_gps import DeepGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, CosineKernel, PeriodicKernel
from gpytorch.means import LinearMean, ZeroMean
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.settings import num_likelihood_samples
from gpytorch.distributions import MultivariateNormal

from linear_operator.operators import DiagLinearOperator, BlockDiagLinearOperator

from dgp.models.layer_gp import DSVI_GPLayer
from dgp.models.model_utils import create_dim_list, unsqueeze_y_stat, predict_mixture_mean_and_covar
from intdom_dgp.linear_op_utils.extract_diag_block import extract_diag_block


def _return_scaled_kernel(kernel_type, batch_shape=torch.Size([]), ard_num_dims=1):
    if kernel_type == "matern12":
        return ScaleKernel(MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=ard_num_dims))
    elif kernel_type == "matern32":
        return ScaleKernel(MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=ard_num_dims))
    elif kernel_type == "matern52":
        return ScaleKernel(MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=ard_num_dims))
    elif kernel_type == "rbf":
        return ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=ard_num_dims))
    elif kernel_type == "cosine":
        k = CosineKernel(batch_shape=batch_shape, ard_num_dims=ard_num_dims)
        k.period_length = 0.01
        return ScaleKernel(k)
    elif kernel_type == "periodic":
        return ScaleKernel(PeriodicKernel(batch_shape=batch_shape, ard_num_dims=ard_num_dims))
    else:
        raise NotImplementedError("Unknown kernel type.")


def _rearrange_multitask_mvn_covar(lazy_cov_matrix, t):
    if isinstance(lazy_cov_matrix, DiagLinearOperator):  # DiagLOP [S, tN, tN], when full_cov=False
        diag = lazy_cov_matrix.diagonal(dim1=-2, dim2=-1).view(*lazy_cov_matrix.shape[:-2], t, -1)  # [S, t, N]
        diag = diag.transpose(-1, -2)  # [S, N, t]
    elif isinstance(lazy_cov_matrix, BlockDiagLinearOperator):  # BlockDiagLOP: [S, tN, tN] when full_cov=True
        base_blocks = lazy_cov_matrix.base_linear_op  # LOP: [S, t, N, N]
        diag = base_blocks.diagonal(dim1=-1, dim2=-2).transpose(-1, -2)  # [S, t, N] -> [S, N, t]
    else:
        raise NotImplementedError
    return diag  # [S, N, t]


class DSVI_DeepGP(DeepGP):
    def __init__(
            self,
            input_dims, output_dims, hidden_dims, num_layers,
            X_running: Optional[np.ndarray],  # used for initialization
            Z_running: Optional[np.ndarray],  # initialized Z at the input layer
            kernel_type='matern32',
            ard=True, kernel_sharing_across_dims=True,
            whitened: str = 'cholesky',
            has_kernel_noise=True, with_concat=False, **kwargs
    ):
        input_dims_list, output_dims_list = create_dim_list(input_dims, output_dims, hidden_dims,
                                                            num_layers, with_concat)
        layers = ModuleList([])
        for i in range(num_layers):
            dim_in = input_dims_list[i]
            dim_out = output_dims_list[i]
            # set mean function for each layer
            if i != num_layers - 1:
                mean_module = LinearMean(input_size=dim_in, batch_shape=torch.Size([dim_out]), bias=False)
                if dim_in == dim_out:  # identity mean
                    mean_module.weights.requires_grad = False  # [batch_size, input_size, 1]
                    mean_module.initialize(weights=torch.eye(dim_in).unsqueeze(-1))  # ⚠️
                else:
                    if dim_in > dim_out:
                        _, _, V = np.linalg.svd(X_running, full_matrices=False)  # V: [..., D_in], orthonormal rows
                        W = V[:dim_out, :].T
                    else:  # dim_in < dim_out
                        W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, dim_out - dim_in))], 1)

                    W_tensor = torch.as_tensor(W, dtype=mean_module.weights.dtype,
                                               device=mean_module.weights.device)  # [dim_in, dim_out]
                    mean_module.initialize(weights=W_tensor.transpose(-1, -2).unsqueeze(-1))  # [dim_out, dim_in,1]
                    mean_module.weights.requires_grad = False
            else:  # final layer
                mean_module = ZeroMean(batch_shape=torch.Size([dim_out]))
                has_kernel_noise = False
            # kernel sharing
            kernel_batch = torch.Size([]) if kernel_sharing_across_dims else torch.Size([dim_out])
            ard_num_dims = dim_in if ard else 1
            kernel = _return_scaled_kernel(kernel_type, batch_shape=kernel_batch, ard_num_dims=ard_num_dims)

            gplayer = DSVI_GPLayer(
                dim_in, dim_out, mean_module, kernel,
                Z=torch.as_tensor(Z_running, dtype=torch.get_default_dtype()),
                whitened=whitened,
                has_kernel_noise=has_kernel_noise,
                learning_inducing_locations=True, **kwargs
            )
            if dim_in != dim_out and i != num_layers - 1:
                Z_running = Z_running.dot(W)  # ndarray
                X_running = X_running.dot(W)
            layers.append(gplayer)

        super().__init__()
        self.layers = layers
        self.likelihood = MultitaskGaussianLikelihood(output_dims, has_global_noise=(output_dims != 1))

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.with_concat = with_concat

    @property
    def device(self) -> torch.device:  # may change after __init__
        return self.layers[0].device

    def forward(self, x, full_cov=False, S=1, **kwargs):  # kwargs can be `prior=True`
        inputs = x.clone() if self.with_concat else None
        with num_likelihood_samples(S):
            for i, layer in enumerate(self.layers):
                if self.num_layers == 1:
                    return layer(x, full_cov=full_cov, **kwargs)
                elif i == 0:  # first layer
                    x = layer(x, full_cov=full_cov, **kwargs)
                else:
                    pre_x = (x, inputs) if self.with_concat else (x,)
                    x = layer(*pre_x, full_cov=full_cov, **kwargs)
        return x

    @torch.no_grad()
    def _get_sampled_mean_and_covar(self, x_batch, y_mean=None, y_std=None, full_cov=False, S=10):
        """
        :return: Mixture component mean [N, S, t] and covar [N, S, t, t] after likelihood
        Note that full_cov will influence the intermediate-layer sampling,
        thus eventually affecting the output distribution.
        We have to remove the correlation along dim `N`  for output in this method.
        Dims `t` are independent by model definition.
        """
        preds = self.likelihood(self(x_batch.to(self.device), full_cov=full_cov, S=S))
        assert preds.mean.ndim == 3 and preds.mean.size(-1) == self.output_dims

        y_mean, y_std = unsqueeze_y_stat(y_mean, y_std, self.output_dims, self.device)
        component_mus = preds.mean * y_std + y_mean
        component_mus = component_mus.transpose(-2, -3)  # [N, S, t]

        assert not preds._interleaved
        # for DGP covar: remove correlation along dim `N`, and dim `t` essentially independent
        variance = preds.variance  # [S, N, t],
        recover_diag = variance * (y_std ** 2)
        block_diag = DiagLinearOperator(recover_diag)  # [S, N, t, t]
        component_covars = block_diag.transpose(-3, -4)  # [S, N, t, t] -> [N, S, t, t]
        return component_mus, component_covars

    @torch.no_grad()
    def predict_mean_and_var(self, x_batch, y_mean=None, y_std=None, full_cov=False, S=10):
        """
        :return: mean [N, t] and covar [N, t, t] of r.v. from Gaussian mixtures
        """
        component_mus, component_covars = self._get_sampled_mean_and_covar(
            x_batch, y_mean, y_std, full_cov=full_cov, S=S
        )
        return predict_mixture_mean_and_covar(component_mus, component_covars)

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_loader, y_mean=None, y_std=None, loader_has_y=True, full_cov=False, S=10
    ):
        mixture_means = []
        mixture_covs = []
        for data_batch in test_loader:
            x_batch = data_batch[0] if loader_has_y else data_batch
            mixture_mus_batch, mixture_covs_batch = self.predict_mean_and_var(
                x_batch, y_mean, y_std, full_cov=full_cov, S=S
            )
            mixture_means.append(mixture_mus_batch)  # [N, t]
            mixture_covs.append(mixture_covs_batch)  # [N, t, t]
        mixture_means = torch.cat(mixture_means, dim=-2)
        mixture_covs = torch.cat(mixture_covs, dim=-3)
        return mixture_means, mixture_covs  # [N, t], [N, t, t]

    @torch.no_grad()
    def predict_measure(self, test_loader, y_mean=None, y_std=None, S=10):
        """
        :return: RMSE and Mean Marginal log-likelihood using `full_cov=False`
        """
        y_mean, y_std = unsqueeze_y_stat(y_mean, y_std, self.output_dims, self.device)
        num_samples = torch.as_tensor(S, dtype=torch.get_default_dtype())

        square_errors = []
        log_likelihoods = []
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            q_f = self(x_batch, full_cov=False, S=S)
            preds = self.likelihood(q_f)  # Multitask  MVN [S, N, D_out]
            assert preds.mean.ndim == 3 and preds.mean.size(-1) == self.output_dims

            mean_average = preds.mean.mean(0)  # [N, D_out]
            square_errors.append(
                torch.square(y_batch * y_std + y_mean - mean_average * y_std - y_mean).sum(-1)
            )  # [N]

            ll = self.likelihood.log_marginal(  # sum log over dim `D_out`, see `gaussian_likelihood.log_marginal`
                y_batch, q_f
            ) - torch.log(y_std).sum(-1)  # [S, N]
            assert S == ll.size(0)
            mean_ll = torch.logsumexp(ll - torch.log(num_samples), 0)  # [N], \log \sum 1/S q_i(y)
            log_likelihoods.append(mean_ll)

        square_errors = torch.cat(square_errors, dim=-1)
        log_likelihoods = torch.cat(log_likelihoods, dim=-1)
        return square_errors.mean(dim=-1).sqrt(), log_likelihoods.mean(dim=-1)

    @torch.no_grad()
    def predict_measure_check(self, test_loader, y_mean=None, y_std=None, S=10):
        """
        :return: check RMSE and Mean MLL when `full_cov=False` using MVN.log_prob instead of likelihood.log_marginal
        """
        y_mean, y_std = unsqueeze_y_stat(y_mean, y_std, self.output_dims, self.device)
        num_samples = torch.as_tensor(S, dtype=torch.get_default_dtype())

        new_se = []
        new_ll = []
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            preds = self.likelihood(self(x_batch, full_cov=False, S=S))  # Multitask  MVN [S, N, D_out]
            assert preds.mean.ndim == 3 and preds.mean.size(-1) == self.output_dims
            # can also use _get_sampled_mean_and_covar(...)
            y_recover = y_batch * y_std + y_mean
            mean_recover = preds.mean * y_std + y_mean
            cov_recover = preds.lazy_covariance_matrix
            # Note than DGP's output distribution Multitask MVN is not interleaved
            cov_recover = extract_diag_block(cov_recover, num_task=self.output_dims, interleaved=False)  # [S, t, N, N]
            cov_recover = DiagLinearOperator(cov_recover.diagonal(dim2=-1, dim1=-2).mT)  # Diag([S, N, t])
            cov_recover = cov_recover * (y_std.mT @ y_std)
            new_preds = MultivariateNormal(mean_recover.transpose(-2, -3), cov_recover.transpose(-3, -4))  # [N, S, t]

            new_mean_recover_average = new_preds.mean.mean(-2)
            new_se.append(torch.square(y_recover - new_mean_recover_average).sum(-1))

            log_q_y = new_preds.log_prob(y_recover.unsqueeze(-2))
            log_q_y = torch.logsumexp(log_q_y - torch.log(num_samples), dim=-1)
            new_ll.append(log_q_y)

        new_se = torch.cat(new_se, dim=-1)
        new_ll = torch.cat(new_ll, dim=-1)
        return new_se.mean(dim=-1).sqrt(), new_ll.mean(dim=-1)

    @torch.no_grad()
    def predict_measure_charis(self, test_loader, y_mean=None, y_std=None, S=10):
        """
        :return: For charis, give metrics per output dim: [N, D_out]
        RMSE and Mean Marginal log-likelihood using `full_cov=False`
        """
        y_mean, y_std = unsqueeze_y_stat(y_mean, y_std, self.output_dims, self.device)
        num_samples = torch.as_tensor(S, dtype=torch.get_default_dtype())

        square_errors = []
        log_likelihoods = []
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            q_f = self(x_batch, full_cov=False, S=S)
            preds = self.likelihood(q_f)  # Multitask  MVN [S, N, D_out]
            assert preds.mean.ndim == 3 and preds.mean.size(-1) == self.output_dims

            mean_average = preds.mean.mean(0)  # [N, D_out]
            square_errors.append(
                torch.square(y_batch * y_std + y_mean - mean_average * y_std - y_mean)
            )  # [N, D_out]

            mariginal = self.likelihood.marginal(q_f)
            indep_dist = Normal(mariginal.mean, mariginal.variance.clamp_min(1e-8).sqrt())
            res = indep_dist.log_prob(y_batch) - torch.log(y_std)   # [S, N, D_out]
            assert S == res.size(0)

            mean_ll = torch.logsumexp(res - torch.log(num_samples), 0)  # [N, D_out], \log \sum 1/S q_i(y)
            log_likelihoods.append(mean_ll)  # [N, D_out]

        square_errors = torch.cat(square_errors, dim=-2)
        log_likelihoods = torch.cat(log_likelihoods, dim=-2)
        return square_errors.mean(dim=-2).sqrt(), log_likelihoods.mean(dim=-2)

    def propagate(self, x, full_cov=False, S=1, **kwargs):
        """
        :return: [(input_samples, output_dist), (input_samples, output_dist), ...] with shape [S, N, t]
        """
        inputs = x.clone() if self.with_concat else None
        res_all_layers = []
        with num_likelihood_samples(S):
            for i, layer in enumerate(self.layers):
                if self.num_layers == 1:
                    res_all_layers.append(layer.propagate(x, full_cov=full_cov, **kwargs))
                    return res_all_layers
                elif i == 0:  # first layer
                    input_samples, x = layer.propagate(x, full_cov=full_cov, **kwargs)
                else:
                    pre_x = (x, inputs) if self.with_concat else (x,)
                    input_samples, x = layer.propagate(*pre_x, full_cov=full_cov, **kwargs)

                res_all_layers.append((input_samples, x))
        return res_all_layers

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        """
        re-organize the predictions at all layers
        :return: (Fmeans: [N, t]; Fvars: [N, t, t]; Fs samples: [S, N, t]; )
        """
        res_all_layers = self.propagate(x, full_cov=full_cov, S=S, **kwargs)
        Fmeans, Fvars, Fs = [], [], []
        for input_samples, output_dist in res_all_layers:
            Fs.append(input_samples)  # [S, N, t]

            component_means = output_dist.mean.transpose(-2, -3)  # [N, S, t]
            component_covs = output_dist.variance.transpose(-2, -3)  # [N, S, t]
            component_covs = DiagLinearOperator(component_covs)  # [N, S, t, t]
            mixture_means, mixture_covars = predict_mixture_mean_and_covar(component_means, component_covs)

            Fmeans.append(mixture_means)  # [N, t]
            Fvars.append(mixture_covars)  # [N, t, t]
        if full_cov:
            samples_final_layer = output_dist.rsample()  # get the samples from the final layer
        else:
            samples_final_layer = Normal(loc=output_dist.mean, scale=output_dist.variance.sqrt()).rsample()
        del Fs[0]
        Fs.append(samples_final_layer)
        return Fmeans, Fvars, Fs



