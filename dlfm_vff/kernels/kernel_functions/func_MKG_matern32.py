import torch
from torch import Tensor

from dlfm_vff.kernels.kernel_functions.utils import (
    get_eps, prepare_elements_mkg, prepare_masked_elements,
    cov_cos_a_b, cov_sin_a_b,
    buid_mask_gama_lam
)

# EPS = get_eps(torch.get_default_dtype())


def cov_cos_lt_a(exp_lam_r_a, r_a, gama, lamda):
    res = (lamda * r_a + 1) / (gama + lamda) + lamda / (gama + lamda).square()
    return res * exp_lam_r_a


def reuse_coeff_cos(gama, lamda, z2_plus_gama2):
    return (gama + 2 * lamda) / (gama + lamda).square() - gama / z2_plus_gama2


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
        term_3 = cov_cos_lt_a(
            exp_lam_r_b_masked, - r_b[..., mask_g_lam, :, :], gama_masked, - lamda_masked
        )  # reuse
        Kvf[..., mask_g_lam, :, :] = term_1 + term_2 + term_3

    mask_g_lam = ~ mask_g_lam
    if torch.any(mask_g_lam):
        (gama_masked, lamda_masked, exp_gama_r_a_masked, exp_lam_r_b_masked) = prepare_masked_elements(
            mask_g_lam, gama, lamda, exp_gama_r_a, exp_gama_r_b=None, exp_lam_r_b=exp_lam_r_b
        )
        r_b_masked = r_b[..., mask_g_lam, :, :]

        term_1 = coeff_reuse_cos[mask_g_lam] * exp_gama_r_a_masked
        term_2 = (
                lamda_masked / z2_plus_gama2[mask_g_lam] + (lamda_masked * r_b_masked + 2) * r_b_masked / 2
        ) * exp_lam_r_b_masked
        Kvf[..., mask_g_lam, :, :] = term_1 + term_2
    return Kvf


def cov_sin_lt_a(z_sin, r_a, exp_lam_r_a, gama, lamda):
    g_plus_lam = gama + lamda
    res = - z_sin * (r_a / g_plus_lam + 1 / g_plus_lam.square())
    return res * exp_lam_r_a


def reuse_coeff_sin(z_sin, gama, lamda, z2_plus_gama2):
    return - z_sin / (gama + lamda).square() + z_sin / z2_plus_gama2


def cov_sin_gt_b(
        coeff_reuse_sin,
        z_sin, r_b, exp_gama_r_a, exp_gama_r_b, exp_lam_r_b,
        gama, lamda, z2_plus_gama2_sin,
        mask_g_lam
):
    Kvf = torch.zeros(torch.broadcast_shapes(exp_gama_r_a.shape, coeff_reuse_sin.shape),
                      device=r_b.device, dtype=r_b.dtype)
    if torch.any(mask_g_lam):
        (gama_masked, lamda_masked,
         exp_gama_r_a_masked, exp_gama_r_b_masked, exp_lam_r_b_masked) = prepare_masked_elements(
            mask_g_lam, gama, lamda, exp_gama_r_a, exp_gama_r_b, exp_lam_r_b
        )
        r_b_masked = r_b[..., mask_g_lam, :, :]

        term_1 = coeff_reuse_sin[mask_g_lam] * exp_gama_r_a_masked
        term_2 = - reuse_coeff_sin(
            z_sin[mask_g_lam], gama_masked, - lamda_masked, z2_plus_gama2_sin[mask_g_lam]
        ) * exp_gama_r_b_masked
        term_3 = cov_sin_lt_a(
            z_sin[mask_g_lam], - r_b_masked, exp_lam_r_b_masked, gama_masked, - lamda_masked
        )
        Kvf[..., mask_g_lam, :, :] = term_1 + term_2 + term_3

    mask_g_lam = ~ mask_g_lam
    if torch.any(mask_g_lam):
        (gama_masked, lamda_masked, exp_gama_r_a_masked, exp_lam_r_b_masked) = prepare_masked_elements(
            mask_g_lam, gama, lamda, exp_gama_r_a, exp_gama_r_b=None, exp_lam_r_b=exp_lam_r_b
        )
        r_b_masked = r_b[..., mask_g_lam, :, :]

        term_1 = coeff_reuse_sin[mask_g_lam] * exp_gama_r_a_masked
        term_2 = (
            - z_sin[mask_g_lam] / z2_plus_gama2_sin[mask_g_lam] + z_sin[mask_g_lam] * r_b_masked.square() / 2
        ) * exp_lam_r_b_masked
        Kvf[..., mask_g_lam, :, :] = term_1 + term_2
    return Kvf


def lfm_matern32_cross_cov(z_cos, t_T, a, b, gama, beta, lamda) -> Tensor:
    r_a, r_b, exp_gama_r_a, exp_gama_r_b, exp_lam_r_a, exp_lam_r_b, z2_plus_gama2 = prepare_elements_mkg(
        z_cos, t_T, a, b, gama, beta, lamda
    )

    coeff_reuse_cos = reuse_coeff_cos(gama, lamda, z2_plus_gama2)
    # cosine part
    mkg_cos_part = torch.where(
        torch.as_tensor(t_T < a),
        cov_cos_lt_a(exp_lam_r_a, r_a, gama, lamda),
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

    z_sin = z_cos[..., 1:, :]
    z2_plus_gama2_sin = z2_plus_gama2[..., 1:, :]
    coeff_reuse_sin = reuse_coeff_sin(z_sin, gama, lamda, z2_plus_gama2_sin)

    mkg_sin_part = torch.where(
        torch.as_tensor(t_T < a),
        cov_sin_lt_a(z_sin, r_a, exp_lam_r_a, gama, lamda),
        cov_sin_a_b(z_sin, r_a, gama, z2_plus_gama2_sin, coeff_reuse_sin, exp_gama_r_a)
    )
    mkg_sin_part = torch.where(
        torch.as_tensor(t_T > b),
        cov_sin_gt_b(
            coeff_reuse_sin,
            z_sin, r_b, exp_gama_r_a, exp_gama_r_b, exp_lam_r_b,
            gama, lamda, z2_plus_gama2_sin, mask_index
        ),
        mkg_sin_part
    )

    return torch.cat([mkg_cos_part, mkg_sin_part], dim=-2) / beta









