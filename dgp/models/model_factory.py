import torch

from dgp.models.deep_gp import DSVI_DeepGP


class DGP_1layers_1d(DSVI_DeepGP):
    def __init__(
            self,
            Z_init, whitened='cholesky',
            kernel_type='matern32', has_kernel_noise=True
    ):
        X_running = None
        super().__init__(
            1, 1, 1, 1,
            X_running, Z_init,
            kernel_type, whitened=whitened,
            has_kernel_noise=has_kernel_noise
        )

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=False, full_cov=True, S=10
    ):
        mixture_means, mixture_covars = super().predict_mean_and_var_loader(
            test_x_loader, y_mean=None, y_std=None, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )
        return mixture_means.squeeze(-1), mixture_covars.squeeze(-1).squeeze(-1)  # [N]

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=True, S=10, **kwargs):
        return DGP_2layer_1d.predict_all_layers(self, x, full_cov=full_cov, S=S, **kwargs)


class DGP_2layer_1d(DSVI_DeepGP):  # can be extended to more layers
    def __init__(
            self,
            Z_running, whitened='cholesky',
            kernel_type='matern32', kernel_sharing_across_dims=True,
            has_kernel_noise=True
    ):
        X_running = None  # we don't actually use X_running when dim_in=dim_out
        super().__init__(
            1, 1, 1, 2, X_running, Z_running,
            kernel_type, kernel_sharing_across_dims=kernel_sharing_across_dims, whitened=whitened,
            has_kernel_noise=has_kernel_noise
        )

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=False, full_cov=True, S=10
    ):
        mixture_means, mixture_covars = super().predict_mean_and_var_loader(
            test_x_loader, y_mean=None, y_std=None, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )
        return mixture_means.squeeze(-1), mixture_covars.squeeze(-1).squeeze(-1)  # [N]

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=True, S=10, **kwargs):
        Fmeans, Fvars, Fs = super(self.__class__, self).predict_all_layers(x, full_cov=full_cov, S=S, **kwargs)
        for i, (sample, mean, var) in enumerate(zip(Fs, Fmeans, Fvars)):
            Fs[i] = sample.squeeze(-1)
            Fmeans[i] = mean.squeeze(-1)  # [N]
            Fvars[i] = var.squeeze(-1).squeeze(-1)  # [N]

        return Fmeans, Fvars, Fs






