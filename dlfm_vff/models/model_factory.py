import torch

from dlfm_vff.models.deep_lfm_vff_lmc import DeepLFM_VFF_LMC
from dlfm_vff.models.deep_lfm_analytic import DeepLFM_AnalyticLMC
from dlfm_vff.models.deep_lfm_rff import DeepLFM_RFFLMC
from intdom_dgp_lmc.models.model_factory import IDDGP_2layers_1d, fix_lmc_coeff, squeeze_res


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
        return IDDGP_2layers_1d.predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )  # squeezed

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        return IDDGP_2layers_1d.predict_all_layers(self, x, full_cov=full_cov, S=S, **kwargs)  # squeezed


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


#########################################################
class DeepLFM_analytic_2layer_1d(DeepLFM_AnalyticLMC):
    def __init__(
            self, M, nu, Z_init=None,
            whitened='none', fix_lmc=True,
            has_kernel_noise=True
    ):
        super().__init__(
            1, 1, 1, 2,
            M, nu, Z_init=Z_init, whitened=whitened, has_kernel_noise=has_kernel_noise
        )
        if fix_lmc:
            fix_lmc_coeff(self.layers)

    @torch.no_grad()
    def predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=False, full_cov=True, S=10
    ):
        return IDDGP_2layers_1d.predict_mean_and_var_loader(
            self, test_x_loader, loader_has_y=loader_has_y, full_cov=full_cov, S=S
        )  # squeezed

    @torch.no_grad()
    def predict_all_layers(self, x, full_cov=False, S=10, **kwargs):
        Fmeans, Fvars, Foutputs = DeepLFM_AnalyticLMC.predict_all_layers(
            self, x, full_cov=full_cov, S=S, **kwargs
        )
        for i, (mean, var, output_sample) in enumerate(zip(Fmeans, Fvars, Foutputs)):
            _, Fmeans[i], Fvars[i], Foutputs[i] = squeeze_res(input_sample=None, mean=mean, var=var,
                                                              output_sample=output_sample)
        return Fmeans, Fvars, Foutputs


class DeepLFM_analytic_1layer_1d(DeepLFM_analytic_2layer_1d):
    def __init__(
            self, M, nu, Z_init=None,
            whitened='none', fix_lmc=True
    ):
        super(DeepLFM_analytic_2layer_1d, self).__init__(
            1, 1, 1, 1,
            M, nu, Z_init=Z_init,
            whitened=whitened, has_kernel_noise=False
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
        return IDDGP_2layers_1d.predict_mean_and_var_loader(
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


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    model = DeepLFM_VFF_2layer_1d(a=-1., b=2., M=3, nu=0.5, whitened='cholesky')

    x_test = torch.randn(25, 1)

    f_inputs, f_means, f_vars, f_outputs = model.predict_all_layers(x_test, full_cov=False, S=10)
    for i, (nor_input, mean, var, out_sample) in enumerate(zip(f_inputs, f_means, f_vars, f_outputs)):
        print(f"===== the layer number :{i} =====")
        print(f"normalized input shape: {nor_input.shape}")
        print(f"mean shape: {mean.shape}")
        print(f"var shape: {var.shape}")
        print(f"output sample shape: {out_sample.shape} \n")
