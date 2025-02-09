import torch

from dlfm_vff.kernels.kernel_functions.utils import (
    construct_mask_gama_lam, prepare_masked_elements, prepare_elements_green_k
)


def t1_lt_t2(exp_lam_d, gama, lamda):
    return exp_lam_d / (gama + lamda)


def t1_gt_t2(exp_gama_d, exp_lam_d, lam_d, gama, lamda):
    GK, mask_index = construct_mask_gama_lam(exp_gama_d.shape, gama - lamda)
    if torch.any(mask_index):
        gama_masked, lamda_masked, exp_gama_d_masked, exp_lam_d_masked, lam_d_masked = prepare_masked_elements(
            mask_index, gama, lamda, exp_gama_d, exp_lam_d, lam_d
        )

        term_1 = - 2 * lamda_masked * exp_gama_d_masked / (gama_masked.square() - lamda_masked.square())
        term_2 = exp_lam_d_masked / (gama_masked - lamda_masked)
        GK[..., mask_index, :, :] = term_1 + term_2

    mask_index = ~ mask_index
    if torch.any(mask_index):
        _, lamda_masked, exp_lam_d_masked, lam_d_masked = prepare_masked_elements(
            mask_index, gama, lamda, exp_lam_d, lam_d
        )
        GK[..., mask_index, :, :] = (1 + 2 * lam_d_masked) * exp_lam_d_masked / (2 * lamda_masked)
    return GK


def lfm_matern12_green_k(t1, t2, distance, gama, beta, outputscale, lamda):
    """
    t1, t2, distance: |t1 - t2|, [(S, D), N, M], t2 as inducing points
    gama, beta, lamda: [D, 1, 1]
    :return: GK: [(S, D), N, M]
    """
    t2_T = t2.mT
    lam_d, sig2_beta, exp_lam_d, exp_gama_d = prepare_elements_green_k(
        distance, gama, beta, outputscale, lamda
    )
    green_k = torch.where(
        torch.as_tensor(t1 <= t2_T),
        t1_lt_t2(exp_lam_d, gama, lamda),
        t1_gt_t2(exp_gama_d, exp_lam_d, lam_d, gama, lamda)
    )
    return outputscale * green_k / beta


if __name__ == '__main__':
    from dlfm_vff.kernels.kernel_functions.utils import test_GK

    torch.set_default_dtype(torch.float64)
    test_GK(nu=0.5, func=lfm_matern12_green_k)


