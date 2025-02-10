import torch

from dlfm_vff.models.deep_lfm_vff_lmc import DeepLFM_VFF_LMC
from dlfm_vff.models.deep_lfm_rff import DeepLFM_RFFLMC
from intdom_dgp_lmc.models.deep_iddgp_lmc import IDDeepGP_VFF_LMC


def fix_lmc_coeff(layers):
    for idgp_lmc_layer in layers:
        lmc_coefficients = idgp_lmc_layer.lmc_layer.lmc_coefficients
        lmc_coefficients.detach().fill_(1.)
        lmc_coefficients.requires_grad = False


def squeeze_res(input_sample=None, mean=None, var=None, output_sample=None):
    if input_sample is not None:
        input_sample = input_sample.squeeze(-1)
    if mean is not None:
        mean = mean.squeeze(-1)
    if var is not None:
        var = var.squeeze(-1).squeeze(-1)
    if output_sample is not None:
        output_sample = output_sample.squeeze(-1)
    return input_sample, mean, var, output_sample


class DeepLFM_VFF_2layer_1d(DeepLFM_VFF_LMC):
    def __init__(
            self,
            a, b, M, nu,
            whitened='none', mean_z_type='with-x',
            fix_lmc=True,
            hidden_normalization_info=None,
            has_kernel_noise=True, boundary_learnable=False
    ):
        super(DeepLFM_VFF_2layer_1d, self).__init__(
            1, 1, 1, 2,
            a, b, M, nu,
            whitened=whitened, mean_z_type=mean_z_type, ode_type='ode1',
            hidden_normalization_info=hidden_normalization_info,
            kernel_sharing_across_dims=False, has_kernel_noise=has_kernel_noise,
            boundary_learnable=boundary_learnable
        )
        if fix_lmc:
            fix_lmc_coeff(self.layers)

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=False, full_cov=True, S=10
    ):
        mixture_means, mixture_covars = IDDeepGP_VFF_LMC.predict_mean_and_var_loader(
            self, test_x_loader, y_mean=None, y_std=None, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )
        return mixture_means.squeeze(-1), mixture_covars.squeeze(-1).squeeze(-1)  # [N]

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        Finputs, Fmeans, Fvars, Foutputs = IDDeepGP_VFF_LMC.predict_all_layers(
            self, x, full_cov=full_cov, S=S, **kwargs
        )
        for i, (input_sample, mean, var, output_sample) in enumerate(zip(Finputs, Fmeans, Fvars, Foutputs)):
            Foutputs[i], Fmeans[i], Fvars[i], Foutputs[i] = squeeze_res(input_sample, mean, var, output_sample)
        return Finputs, Fmeans, Fvars, Foutputs


class DeepLFM_VFF_1Layer_1d(DeepLFM_VFF_2layer_1d):
    def __init__(
            self, a, b, M, nu,
            whitened='none', fix_lmc=True,
            boundary_learnable=False
    ):
        super(DeepLFM_VFF_2layer_1d, self).__init__(
            1, 1, 1, 1,
            a, b, M, nu,
            whitened=whitened, mean_z_type='with-x', ode_type='ode1',
            hidden_normalization_info=None, has_kernel_noise=False,
            boundary_learnable=boundary_learnable
        )
        if fix_lmc:
            fix_lmc_coeff(self.layers)


############################################
class DeepLFM_RFF_2layer_1d(DeepLFM_RFFLMC):
    def __init__(
            self, M, nu, Z_init=None, num_samples=100,
            whitened='none', fix_lmc=True,
            has_kernel_noise=True, integral_lower_bound=None
    ):
        super().__init__(
            1, 1, 1, 2,
            M, nu, Z_init=Z_init, num_samples=num_samples,
            whitened=whitened, has_kernel_noise=has_kernel_noise, integral_lower_bound=integral_lower_bound
        )
        if fix_lmc:
            fix_lmc_coeff(self.layers)

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=False, full_cov=True, S=10
    ):
        return DeepLFM_VFF_2layer_1d.predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )  # squeezed

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        Fmeans, Fvars, Foutputs = super().predict_all_layers(
            x, full_cov=full_cov, S=S, **kwargs
        )
        for i, (mean, var, output_sample) in enumerate(zip(Fmeans, Fvars, Foutputs)):
            _, Fmeans[i], Fvars[i], Foutputs[i] = squeeze_res(input_sample=None, mean=mean, var=var,
                                                              output_sample=output_sample)
        return Fmeans, Fvars, Foutputs


class DeepLFM_RFF_1layer_1d(DeepLFM_RFF_2layer_1d):
    def __init__(
            self, M, nu, Z_init=None, num_samples=100,
            whitened='none', fix_lmc=True, integral_lower_bound=None
    ):
        super(DeepLFM_RFF_2layer_1d, self).__init__(
            1, 1, 1, 1,
            M, nu, Z_init=Z_init, num_samples=num_samples,
            whitened=whitened, has_kernel_noise=False, integral_lower_bound=integral_lower_bound
        )
        if fix_lmc:
            fix_lmc_coeff(self.layers)


