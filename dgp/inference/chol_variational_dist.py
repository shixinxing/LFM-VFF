import torch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal


class CholeskyVariationalDist_ZeroInit(CholeskyVariationalDistribution):
    """
    A Cholesky variational distribution q(u)=N(m, S) overriding `initialize_variational_distribution`
    Previous variational strategy initializes q(u) to p(u),
    here the initialization is changed: m can be initialized to zero no matter the mean of p(u).
    """
    def __init__(
            self,
            num_inducing_points: int,
            batch_shape: torch.Size = torch.Size([]),
            mean_init_std: float = 1e-3,
            force_mean_to_zero: bool = False,
            **kwargs,
    ):
        super().__init__(num_inducing_points, batch_shape, mean_init_std, **kwargs)
        self.force_mean_to_zero = force_mean_to_zero

    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        if self.force_mean_to_zero:  # force m=0 in q(u) = N(m, S)
            self.variational_mean.data = torch.zeros_like(self.variational_mean)
            self.variational_mean.data.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
            self.chol_variational_covar.data.copy_(prior_dist.lazy_covariance_matrix.cholesky().to_dense())
        else:
            return super().initialize_variational_distribution(prior_dist)


