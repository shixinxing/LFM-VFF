from typing import Optional
import warnings
import torch
from torch import Tensor

from gpytorch.models import ApproximateGP
from gpytorch.kernels import Kernel
from gpytorch.utils.memoize import clear_cache_hook, cached
from gpytorch.distributions import MultivariateNormal

from dgp.models.model_utils import _build_kernel_noise, _add_kernel_noise


class _Intdom_LatentGP(ApproximateGP):
    """
    Abstract base class for inter-domain latent GPs.
    Only accept one-dimensional inputs.
    Given three blocks of covariance matrix (Kxx, Kzz, Kzx), compute posterior q(f)
    """
    def __init__(
            self,
            input_dims: int,  # num of latent GPs, one GP per input dim
            kxx: Kernel, kzz: Kernel, kzx: Kernel,
            inducing_points: Tensor,  # [(l), 2M+1, 1], differ from DGP class
            mean_x_type='zero', mean_z_type='zero',  # or `with-x`
            whitened: str = 'none',   # control which whitened Variational Strategy to use,
            kernel_sharing_across_dims=False,
            has_kernel_noise=False, jitter_val: Optional[float] = None
    ):
        inducing_points = inducing_points.clone()

        self.kernel_batch_shape = kxx.batch_shape
        self.num_latents = input_dims  # num of latent GPs, i,e, output_dims before LMC

        self.mean_x_type = mean_x_type
        self.mean_z_type = mean_z_type
        self.whitened = whitened
        if whitened != 'none' and self.mean_z_type != 'zero':
            warnings.warn(f"Using whitening '{whitened}', mean z type shouldn't be {self.mean_z_type}, "
                          f"we force it to zero mean function", UserWarning)
            self.mean_z_type = 'zero'  # whitened, prior mean(Z) = 0

        self.has_kernel_noise = has_kernel_noise
        self.kernel_sharing_across_dims = kernel_sharing_across_dims

        variational_strategy = self.create_variational_strategy(inducing_points, jitter_val)

        super().__init__(variational_strategy)

        self.kxx, self.kzz, self.kzx = kxx, kzz, kzx

        self.build_mean_module()
        self.build_kernel_noise()

    def create_variational_strategy(self, inducing_points, jitter_val):  # Z: [(D=l), 2M+1, 1]
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.kxx.device

    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    def build_mean_module(self):
        raise NotImplementedError

    def build_kernels(self, *args, **kwargs):
        raise NotImplementedError

    def build_kernel_noise(self):
        return _build_kernel_noise(self)

    def add_kernel_noise(self, covar_x, full_cov):
        return _add_kernel_noise(self, covar_x, full_cov)

    def __call__(self, inputs: Tensor, prior=False, full_cov=True, **kwargs):  # input: [(S), N, D=l]
        inputs_multitask = inputs.transpose(-1, -2).unsqueeze(-1)  # [(S), D=l, N, 1]
        if self.training:  # Delete previously cached items (e.g. m(Z), Kzz, Kxx)
            self._clear_cache()
        return super().__call__(
            inputs_multitask, full_cov=full_cov, prior=prior, **kwargs
        )  # go to Variational Strategy

    def mf_x(self, x):
        return self.mean_module(x)

    @cached(name='mean_z_memo')
    def mf_z(self, z):
        return self.mean_z_module(z)

    @cached(name='covar_x_memo', ignore_args=False)
    def covar_x_with_noise(self, x, diag, **kwargs):
        covar_x = self.kxx(x, diag=diag, **kwargs)
        covar_x = self.add_kernel_noise(covar_x, full_cov=not diag)
        return covar_x

    def covar_z(self, z, *args, **kwargs):  # [(l), 2M+1, 2M+1]
        raise NotImplementedError

    def covar_zx(self, z, x, *args, **kwargs):  # [l, 2M+1, N]
        raise NotImplementedError

    def forward(self, x, output_form='Full', full_cov=True):
        if x is None:
            assert output_form == 'VFF'  # used to get prior p(u)
        else:
            assert x.size(-1) == 1 and x.size(-3) == self.num_latents

        z = self.variational_strategy.inducing_points  # [(D=l), 2M+1, 1]

        if output_form == 'VFF':   # used in Variational Strategy and  q(u) initialization
            mean_z = self.mf_z(z)  # [D=l, 2M+1]
            covar_z = self.covar_z(z)  # cached to accelerate prediction
            # Kzz may not have batch dims, while mean(Z) must have batch dims due to latent dims.
            covar_z = covar_z.expand(*mean_z.shape[:-1], *covar_z.shape[-2:])
            return MultivariateNormal(mean_z, covar_z)

        elif output_form == 'Full':
            mean_x = self.mf_x(x)  # [(S), D=l, N]
            mean_z = self.mf_z(z)  # cached [D=l, 2M+1]

            covar_x = self.covar_x_with_noise(x, diag=not full_cov)  # all cached
            covar_zz = self.covar_z(z)  # may not have batch shape if shared [(D=l), 2M+1, 2M+1]
            covar_zx = self.covar_zx(z, x)
            return mean_x, mean_z, covar_x, covar_zz, covar_zx
        elif output_form == 'Prior':
            mean_x = self.mf_x(x)
            covar_x = self.covar_x_with_noise(x, diag=not full_cov)
            return MultivariateNormal(mean_x, covar_x)
        else:
            raise ValueError(f"Invalid output_form: {output_form}.")



