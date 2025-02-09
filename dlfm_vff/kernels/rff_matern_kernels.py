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


if __name__ == "__main__":
    def test_kernel(nu=0.5, num_samples=1000, batch_shape=torch.Size([])):
        matern_rff_kernel = Matern_RFF_Kernel(nu, num_samples, batch_shape)
        matern_rff_kernel.lengthscale = torch.rand(*batch_shape, 1, 1)
        matern_kernel = MaternKernel(nu=nu, batch_shape=batch_shape)
        matern_kernel.lengthscale = matern_rff_kernel.lengthscale

        x1 = torch.randn(*batch_shape, 4, 1) * 2
        x2 = torch.randn(*batch_shape, 5, 1) * 3
        K_truth = matern_kernel(x1, x2).to_dense()
        K_rff = matern_rff_kernel(x1, x2).to_dense()

        print(f"===== nu:{nu}, batch_shape:{batch_shape} =====")
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}\n")
        print(f"K_truth (shape: {K_truth.shape}): \n{K_truth}\n")
        print(f"K_rff (shape: {K_rff.shape}): \n{K_rff}\n")
        delta = K_truth - K_rff
        assert torch.linalg.norm(K_truth) < 1e-5
        relative_error = torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(K_truth)
        print(f"relative error: {relative_error}, max delta: {torch.max(torch.abs(delta))}\n")

    torch.set_default_dtype(torch.float64)
    # test_kernel(nu=0.5, num_samples=1000000, batch_shape=torch.Size([2]))
    # test_kernel(nu=1.5, num_samples=100000, batch_shape=torch.Size([2]))
    test_kernel(nu=2.5, num_samples=100000, batch_shape=torch.Size([2]))


