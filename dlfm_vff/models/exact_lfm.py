from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d


class ExactLFM1d(ExactGP):
    def __init__(
            self, train_x, train_y, gaussian_likelihood,
            mean_type='zero', kernel_type='matern12',
            ode_type='ode1'
    ):
        if train_x.ndim == 1:
            train_x = train_x.unsqueeze(-1)
        assert train_x.size(-1) == 1

        super(ExactLFM1d, self).__init__(train_x, train_y, gaussian_likelihood)

        self.mean_type = mean_type
        self.kernel_type = kernel_type
        self.ode_type = ode_type

        self.mean_module = self.build_mean_module()
        self.scaled_matern_kernel, self.lfm_kernel = self.build_kernels()

    def build_mean_module(self):
        if self.mean_type == 'zero':
            return ZeroMean()
        else:
            raise NotImplementedError(f'Unknown mean type {self.mean_type}')

    def build_kernels(self):
        assert self.kernel_type == 'matern12' or self.kernel_type == 'matern32' or self.kernel_type == 'matern52'
        nu = 0.5 if self.kernel_type == 'matern12' else 1.5 if self.kernel_type == 'matern32' else 2.5
        scaled_matern_kernel = ScaleKernel(MaternKernel(nu=nu))
        lfm_kernel = LFM_MaternKernel1d(scaled_matern_kernel, ode_type=self.ode_type)
        return scaled_matern_kernel, lfm_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.lfm_kernel(x)
        return MultivariateNormal(mean_x, covar_x)

    def latent_force(self, x):    # get p(u)
        mean_u = self.mean_module(x)
        covar_u = self.scaled_matern_kernel(x)
        return MultivariateNormal(mean_u, covar_u)

