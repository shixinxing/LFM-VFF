import torch
from torch import Tensor

from dlfm_vff.kernels.kernel_functions.utils import (
    get_eps, prepare_elements_mkg, prepare_masked_elements,
    cov_cos_a_b, cov_sin_a_b,
    buid_mask_gama_lam
)

EPS = 0
# EPS = get_eps(torch.get_default_dtype())


def cov_cos_lt_a(exp_lam_r_a, gama, lamda) -> Tensor:
    return exp_lam_r_a / (gama + lamda)


def reuse_coeff_cos(gama, lamda, z2_plus_gama2):
    return 1 / (gama + lamda) - gama / z2_plus_gama2


def cov_cos_gt_b(
        coeff_reuse_cos,
        r_b, exp_gama_r_a, exp_gama_r_b, exp_lam_r_b,
        gama, lamda, z2_plus_gama2,
        mask_g_lam
) -> Tensor:
    Kvf = torch.zeros(torch.broadcast_shapes(exp_gama_r_a.shape, z2_plus_gama2.shape),
                      device=r_b.device, dtype=r_b.dtype)
    if torch.any(mask_g_lam):
        (gama_masked, lamda_masked,
         exp_gama_r_a_masked, exp_gama_r_b_masked, exp_lam_r_b_masked) = prepare_masked_elements(
            mask_g_lam, gama, lamda, exp_gama_r_a, exp_gama_r_b, exp_lam_r_b
        )

        term_1 = coeff_reuse_cos[mask_g_lam] * exp_gama_r_a_masked
        term_2 = - reuse_coeff_cos(gama_masked, - lamda_masked, z2_plus_gama2[mask_g_lam]) * exp_gama_r_b_masked
        term_3 = 1 / (gama_masked - lamda_masked) * exp_lam_r_b_masked
        Kvf[..., mask_g_lam, :, :] = term_1 + term_2 + term_3

    mask_g_lam = ~ mask_g_lam
    if torch.any(mask_g_lam):
        (gama_masked, lamda_masked, exp_gama_r_a_masked, exp_lam_r_b_masked) = prepare_masked_elements(
            mask_g_lam, gama, lamda, exp_gama_r_a, exp_gama_r_b=None, exp_lam_r_b=exp_lam_r_b
        )

        term_1 = coeff_reuse_cos[mask_g_lam] * exp_gama_r_a_masked
        term_2 = (lamda_masked / z2_plus_gama2[mask_g_lam] + r_b[..., mask_g_lam, :, :]) * exp_lam_r_b_masked
        Kvf[..., mask_g_lam, :, :] = term_1 + term_2
    return Kvf


def cov_sin_gt_b(coeff_reuse_sin, exp_gama_r_a, exp_gama_r_b) -> Tensor:
    return coeff_reuse_sin * (exp_gama_r_a - exp_gama_r_b)


def lfm_matern12_cross_cov(z_cos, t_T, a, b, gama, beta, lamda) -> Tensor:
    """
    t_T:[(S, d,) 1, N]; z:[(d,) M, 1]; gama, beta, lamda: [d, 1, 1]; Kvf: [(S, d,), M, N]
    :return: MKG: [(S, d,), M, N]
    """
    r_a, r_b, exp_gama_r_a, exp_gama_r_b, exp_lam_r_a, exp_lam_r_b, z2_plus_gama2 = prepare_elements_mkg(
        z_cos, t_T, a, b, gama, beta, lamda
    )
    # cosine part
    coeff_reuse_cos = reuse_coeff_cos(gama, lamda, z2_plus_gama2)
    mkg_cos_part = torch.where(
        torch.as_tensor(t_T < a),
        cov_cos_lt_a(exp_lam_r_a, gama, lamda),
        cov_cos_a_b(z_cos, r_a, gama, z2_plus_gama2, coeff_reuse_cos, exp_gama_r_a)
    )

    mask_index = buid_mask_gama_lam(gama, lamda)
    mkg_cos_part = torch.where(
        torch.as_tensor(t_T > b),
        cov_cos_gt_b(
            coeff_reuse_cos,
            r_b, exp_gama_r_a, exp_gama_r_b, exp_lam_r_b,
            gama, lamda, z2_plus_gama2, mask_index
        ),
        mkg_cos_part
    )
    # sine part
    z_sin = z_cos[..., 1:, :]
    coeff_reuse_sin = z_sin / z2_plus_gama2[..., 1:, :]
    mkg_sin_part = torch.where(
        torch.as_tensor(t_T < a),
        torch.as_tensor(0, device=t_T.device, dtype=t_T.dtype),
        cov_sin_a_b(z_sin, r_a, gama, z2_plus_gama2[..., 1:, :], coeff_reuse_sin, exp_gama_r_a)
    )

    mkg_sin_part = torch.where(
        torch.as_tensor(t_T > b),
        cov_sin_gt_b(coeff_reuse_sin, exp_gama_r_a, exp_gama_r_b),
        mkg_sin_part
    )

    return torch.cat([mkg_cos_part, mkg_sin_part], dim=-2) / (beta + EPS)


