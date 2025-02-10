from typing import Optional

import torch
from torch.distributions.studentT import StudentT

from gpytorch.kernels import RFFKernel, MaternKernel
from torch import Tensor


class Matern_RFF_Kernel(RFFKernel):
    """
    Approximate Matern kernel by Random Fourier Features from a student-T distribution,
    adapted from gpytorch's RFFKernel
    """
    def __init__(
            self, nu: float, num_samples: int,
            batch_shape=torch.Size([]), **kwargs
    ):
        self.nu = nu
        assert nu == 0.5 or nu == 1.5 or nu == 2.5
        self.dist = StudentT(2 * nu, loc=0, scale=1)
        super().__init__(num_samples, num_dims=1, batch_shape=batch_shape, **kwargs)  # only accept 1-dim inputs

    def _init_weights(   # override
            self, num_dims=1,
            num_samples: Optional[int] = None,
            randn_weights: Optional[Tensor] = None
    ):
        if randn_weights is None:
            randn_shape = torch.Size([*self._batch_shape, 1, num_samples])
            randn_weights = self.dist.sample(randn_shape).to(self.raw_lengthscale)
        self.register_buffer("randn_weights", randn_weights)

    def forward(self, *args, **kwargs) -> Tensor:
        return super().forward(*args, **kwargs)

    def _featurize(self, x: Tensor, normalize: bool = False) -> Tensor:
        return super()._featurize(x, normalize=normalize)


