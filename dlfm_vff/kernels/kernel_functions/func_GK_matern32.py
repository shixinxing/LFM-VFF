import torch

from dlfm_vff.kernels.kernel_functions.utils import (
    construct_mask_gama_lam, prepare_masked_elements, prepare_elements_green_k
)


def reuse_coeff(lam_d, gama, lamda) -> torch.Tensor:
    g_plus_lam = gama + lamda
    coeff = (lam_d + 1) / g_plus_lam + lamda / g_plus_lam.square()
    return coeff


def t1_lt_t2(exp_lam_d, lam_d, gama, lamda):
    coeff = reuse_coeff(lam_d, gama, lamda)
    return coeff * exp_lam_d


def t1_gt_t2(exp_gama_d, exp_lam_d, lam_d, gama, lamda):
    GK, mask_index = construct_mask_gama_lam(exp_gama_d.shape, gama - lamda)
    if torch.any(mask_index):
        gama_masked, lamda_masked, exp_gama_d_masked, exp_lam_d_masked, lam_d_masked = prepare_masked_elements(
            mask_index, gama, lamda, exp_gama_d, exp_lam_d, lam_d
        )

        term_1 = 4 * lamda_masked.pow(3) * exp_gama_d_masked / (gama_masked.square() - lamda_masked.square()).square()
        term_2 = reuse_coeff(lam_d_masked, gama_masked, - lamda_masked) * exp_lam_d_masked
        GK[..., mask_index, :, :] = term_1 + term_2

    mask_index = ~ mask_index
    if torch.any(mask_index):
        _, lamda_masked, exp_lam_d_masked, lam_d_masked = prepare_masked_elements(
            mask_index, gama, lamda, exp_lam_d, lam_d
        )
        res = (2 * lam_d_masked.square() + 4 * lam_d_masked + 3) * exp_lam_d_masked / (4 * lamda_masked)
        GK[..., mask_index, :, :] = res
    return GK


def lfm_matern32_green_k(t1, t2, distance, gama, beta, outputscale, lamda):
    t2_T = t2.mT
    lam_d, sig2_beta, exp_lam_d, exp_gama_d = prepare_elements_green_k(
        distance, gama, beta, outputscale, lamda
    )
    green_k = torch.where(
        torch.as_tensor(t1 <= t2_T),
        t1_lt_t2(exp_lam_d, lam_d, gama, lamda),
        t1_gt_t2(exp_gama_d, exp_lam_d, lam_d, gama, lamda)
    )
    return outputscale * green_k / beta


if __name__ == '__main__':
    from dlfm_vff.kernels.kernel_functions.utils import test_GK

    torch.set_default_dtype(torch.float64)
    test_GK(nu=0.5, func=lfm_matern32_green_k)


