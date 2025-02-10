import torch
from torch import Tensor

from gpytorch.distributions import MultivariateNormal


def blockdiag_kl_divergence(
        q: MultivariateNormal,
        p: MultivariateNormal
) -> Tensor:   # _VariationalStrategy.kl_divergence() doesn't utilize the Kvv's structure
    """
    p, q: MVN mean: [..., t, N], covar: [..., t, N, N]
    KL[q||p] * 2 = (mean_q - mean_p)^T @ Kzz^{-1} @ (mean_q - mean_p) + tr{Kzz^{-1} @ S} - N + log|Kzz|/|S|
    """
    q_mu, S = q.mean.unsqueeze(-1), q.lazy_covariance_matrix  # CholLinearOP
    p_mu, Kvv = p.mean.unsqueeze(-1), p.lazy_covariance_matrix  # D + beta beta^T class/BlockDiag Matrix

    delta_mu = q_mu - p_mu
    KInv_mu = torch.linalg.solve(Kvv, delta_mu)
    Mahalanobis = (delta_mu.transpose(-1, -2) @ KInv_mu).squeeze(-1).squeeze(-1)  # [t]
    # solve with TriangularLOP `S.root` is not implemented; LOP's sum is restricted to along one dim
    # trace_term = 0.5 * (S.root * torch.linalg.solve(Kvv, S.root.to_dense())).sum(-1).sum(-1)
    trace_term = Kvv.trace_KiX(S)  # utilize tailored tr{K^{-1}S}
    constant = torch.as_tensor(q_mu.size(-2), dtype=torch.get_default_dtype())
    log_det_Kvv = Kvv.logdet()  # efficient computation for the left two dets.
    log_det_S = S.logdet()
    re = (Mahalanobis + trace_term - constant + log_det_Kvv - log_det_S) * 0.5
    return re


def blockdiag_kl_divergence_inv_quad_logdet(
        q: MultivariateNormal,
        p: MultivariateNormal
) -> Tensor:  # adapt from the above but utilize efficient `inv_quad_logdet`
    q_mu, S = q.mean.unsqueeze(-1), q.lazy_covariance_matrix  # CholLinearOP
    p_mu, Kvv = p.mean.unsqueeze(-1), p.lazy_covariance_matrix  # D + beta beta^T, BlockDiag Kuu

    delta_mu = q_mu - p_mu
    Mahalanobis, log_det_Kvv = Kvv.inv_quad_logdet(inv_quad_rhs=delta_mu, logdet=True, reduce_inv_quad=True)  # [t]
    trace_term = Kvv.trace_KiX(S)  # tr{K^{-1}S}
    constant = torch.as_tensor(q_mu.size(-2), dtype=torch.get_default_dtype())
    log_det_S = S.logdet()
    re = (Mahalanobis + trace_term - constant + log_det_Kvv - log_det_S) * 0.5
    return re


def kl_divergence_whitened(
        q: MultivariateNormal
) -> Tensor:
    """
    when p(u) = N(0, I), Kzz = I
    KL[q||p] * 2 = mean_q^T @ mean_q + tr{S} - N - log|S|
    """
    q_mu, S = q.mean, q.lazy_covariance_matrix  # CholLinearOP
    Mahalanobis = q_mu.square().sum(-1)
    trace_term = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    constant = torch.as_tensor(q_mu.size(-1), dtype=torch.get_default_dtype(), device=q_mu.device)
    log_det_S = S.logdet()
    res = (Mahalanobis + trace_term - constant - log_det_S) * 0.5
    return res


