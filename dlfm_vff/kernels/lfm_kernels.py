import math

import torch
from torch import Tensor

from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.constraints import Positive
from linear_operator.operators import to_linear_operator

from dlfm_vff.kernels.kernel_functions.func_GKG import (
    lfm_matern12_kernel_func, lfm_matern32_kernel_func, lfm_matern52_kernel_func
)


class LFM_MaternKernel1d(Kernel):
    """
    LFM kernels induced by Matérn kernels in closed form with lower integral bound to -infinity
    Note that for each dimension of the input, we only consider a single latent force
    """
    def __init__(
            self, scaled_matern_kernel,
            ode_type='ode1',
            **kwargs
    ):
        super(LFM_MaternKernel1d, self).__init__(**kwargs)
        self.scaled_matern_kernel = scaled_matern_kernel
        # whether to share `gama` across inputs' dimensions
        self.ode_batch_shape = self.scaled_matern_kernel.base_kernel.batch_shape
        if ode_type == 'ode1':
            self.register_parameter('raw_alpha', torch.nn.Parameter(torch.ones(self.ode_batch_shape)))
            self.register_constraint('raw_alpha', Positive())
            self.register_parameter('raw_beta', torch.nn.Parameter(torch.ones(self.ode_batch_shape)))
            self.register_constraint('raw_beta', Positive())
        else:
            # TODO with other ODE
            raise NotImplementedError("TODO with other ODEs")

        self.ode_type = ode_type
        self.nu = self.scaled_matern_kernel.base_kernel.nu

    @property   # override Kernel's batch_shape, this kernel's batch shape is determined by base matern kernel
    def batch_shape(self) -> torch.Size:
        return self.scaled_matern_kernel.base_kernel.batch_shape

    @batch_shape.setter
    def batch_shape(self, batch_shape: torch.Size):
        self.scaled_matern_kernel.base_kernel.batch_shape = batch_shape

    @property
    def outputscale(self):
        return self.scaled_matern_kernel.outputscale  # [b,]

    @outputscale.setter
    def outputscale(self, value):
        self.scaled_matern_kernel.outputscale = value

    @property
    def lengthscale(self):
        return self.scaled_matern_kernel.base_kernel.lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        self.scaled_matern_kernel.base_kernel.lengthscale = value

    @property
    def lamda(self):
        return math.sqrt(self.nu * 2) / self.lengthscale   # [b,1,d=1]

    @lamda.setter
    def lamda(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.lengthscale = math.sqrt(self.nu * 2) / value

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    @property
    def gama(self):
        return self.alpha / self.beta

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        """
        x1, x2: [...,(S), b, N, 1]
        """
        # assert x1.size(-1) == 1 and x2.size(-1) == 1
        distance = self.covar_dist(x1, x2, diag=diag, square_dist=False, **params)  # [(S, b), N, M] or [(S, b), N]
        if diag:
            distance = distance.unsqueeze(-1)
        lamda = self.lamda   # [b,1,d=1]
        gama = self.gama.unsqueeze(-1).unsqueeze(-1).expand_as(lamda)  # broadcast if sharing gama
        beta = self.beta.unsqueeze(-1).unsqueeze(-1).expand_as(lamda)
        outputscale = self.outputscale.unsqueeze(-1).unsqueeze(-1)

        if len(distance.shape) < len(gama.shape):  # just for test, this will never happen in practice
            brdcst_shape = torch.broadcast_shapes(gama.shape, distance.shape)
            distance = distance.expand(*brdcst_shape)

        if self.nu == 0.5:
            output = lfm_matern12_kernel_func(distance, gama, beta, outputscale, lamda)
        elif self.nu == 1.5:
            output = lfm_matern32_kernel_func(distance, gama, beta, outputscale, lamda)
        elif self.nu == 2.5:
            output = lfm_matern52_kernel_func(distance, gama, beta, outputscale, lamda)
        else:
            raise NotImplementedError

        if diag:
            return output.squeeze(-1)  # won't return LOP but tensor
        return to_linear_operator(output)

