from gpytorch.kernels import ScaleKernel

from dlfm_vff.models.exact_lfm import ExactLFM1d
from dlfm_vff.kernels.rff_matern_kernels import Matern_RFF_Kernel
from dlfm_vff.kernels.rff_lfm_kernels import LFMMatern_RFF_Kernel1d


class ExactLFM1d_RFF(ExactLFM1d):
    def __init__(
            self, train_x, train_y, gaussian_likelihood,
            num_rff: int = 20,
            mean_type='zero', kernel_type='matern12',
            ode_type='ode1'
    ):
        self.num_rff = num_rff
        super().__init__(train_x, train_y, gaussian_likelihood, mean_type, kernel_type, ode_type)
        # alias
        self.scaled_matern_rff_kernel = self.scaled_matern_kernel
        self.lfm_rff_kernel = self.lfm_kernel

    def build_kernels(self):  # override
        assert self.kernel_type == 'matern12' or self.kernel_type == 'matern32' or self.kernel_type == 'matern52'
        nu = 0.5 if self.kernel_type == 'matern12' else 1.5 if self.kernel_type == 'matern32' else 2.5

        scaled_matern_rff_kernel = ScaleKernel(Matern_RFF_Kernel(nu=nu, num_samples=self.num_rff))
        lfm_rff_kernel = LFMMatern_RFF_Kernel1d(scaled_matern_rff_kernel, ode_type=self.ode_type,
                                                integral_lower_bound=None)
        return scaled_matern_rff_kernel, lfm_rff_kernel
