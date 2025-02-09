import torch

from dlfm_vff.kernels.kernel_functions.utils import get_eps, prepare_elements_gkg, construct_mask_gama_lam

EPS = get_eps(torch.get_default_dtype())


def lfm_matern12_kernel_func(distance, gama, beta, outputscale, lamda):
    # reuse terms as much as possible
    (
        lam_d, gama_d, sig2_div_beta, exp_lam_d, exp_gama_d, gama_sub_lam
    ) = prepare_elements_gkg(distance, gama, beta, outputscale, lamda)

    output, mask_index = construct_mask_gama_lam(exp_lam_d.shape, gama_sub_lam, bound=None)
    if torch.any(mask_index):   # to avoid all False mask: lam[mask]'s shape = [0, 1, 1]
        gama_masked, lamda_masked = gama[mask_index], lamda[mask_index]  # [d', 1, 1]

        numer = (
            gama_masked * exp_lam_d[..., mask_index, :, :] - lamda_masked * exp_gama_d[..., mask_index, :, :]
        )
        denomi = gama_masked * gama_sub_lam[mask_index] * (gama_masked + lamda_masked)
        res = sig2_div_beta[mask_index] * numer / denomi
        output[..., mask_index, :, :] = res  # still work w/o batch dim

    mask_index = ~ mask_index
    if torch.any(mask_index):
        lamda_masked = lamda[mask_index]

        numer = (1. + lam_d[..., mask_index, :, :]) * exp_lam_d[..., mask_index, :, :]
        res = sig2_div_beta[mask_index] * numer / (2. * lamda_masked.square() + EPS)
        output[..., mask_index, :, :] = res

    return output


def lfm_matern32_kernel_func(distance, gama, beta, outputscale, lamda):
    (
        lam_d, gama_d, sig2_div_beta, exp_lam_d, exp_gama_d, gama_sub_lam
    ) = prepare_elements_gkg(distance, gama, beta, outputscale, lamda)

    output, mask_index = construct_mask_gama_lam(exp_lam_d.shape, gama_sub_lam, bound=None)
    if torch.any(mask_index):
        gama_masked, lamda_masked = gama[mask_index], lamda[mask_index]
        gama2_lam2 = gama_masked.square() - lamda_masked.square()

        term_1 = (lam_d[..., mask_index, :, :] + 1) / gama2_lam2 - 2 * lamda_masked.square() / gama2_lam2.square()
        term_1 = term_1 * exp_lam_d[..., mask_index, :, :]
        term_2 = 2 * lamda_masked.pow(3) / (gama_masked * gama2_lam2.square())
        term_2 = term_2 * exp_gama_d[..., mask_index, :, :]
        res = sig2_div_beta[mask_index] * (term_1 + term_2)
        output[..., mask_index, :, :] = res

    mask_index = ~ mask_index
    if torch.any(mask_index):
        lamda_masked = lamda[mask_index]
        lam_d_masked = lam_d[..., mask_index, :, :]

        term = (lam_d_masked.square() + 3 * lam_d_masked + 3) * exp_lam_d[..., mask_index, :, :]
        res = sig2_div_beta[mask_index] * term / (4 * lamda_masked.square() + EPS)
        output[..., mask_index, :, :] = res

    return output


def lfm_matern52_kernel_func(distance, gama, beta, outputscale, lamda):
    (
        lam_d, gama_d, sig2_div_beta, exp_lam_d, exp_gama_d, gama_sub_lam
    ) = prepare_elements_gkg(distance, gama, beta, outputscale, lamda)

    output, mask_index = construct_mask_gama_lam(exp_lam_d.shape, gama_sub_lam, bound=None)
    if torch.any(mask_index):
        gama_masked, lamda_masked = gama[mask_index], lamda[mask_index]
        gama2_lam2 = gama_masked.square() - lamda_masked.square()
        lam_d_masked = lam_d[..., mask_index, :, :]

        term_1 = (lam_d_masked.square() + 3) / gama2_lam2
        term_2 = lam_d_masked * (3 * gama_masked.square() - 7 * lamda_masked.square()) / gama2_lam2.square()
        term_3 = 4 * lamda_masked.square() * (- gama_masked.square() + 3 * lamda_masked.square()) / gama2_lam2.pow(3)
        res_1 = (term_1 + term_2 + term_3) * exp_lam_d[..., mask_index, :, :]

        res_2 = 8 * lamda_masked.pow(5) * exp_gama_d[..., mask_index, :, :] / (gama_masked * gama2_lam2.pow(3))
        res = sig2_div_beta[mask_index] * (res_1 - res_2) / 3
        output[..., mask_index, :, :] = res

    mask_index = ~ mask_index
    if torch.any(mask_index):
        lamda_masked = lamda[mask_index]
        lam_d_masked = lam_d[..., mask_index, :, :]

        term = lam_d_masked.pow(3) + 6 * lam_d_masked.pow(2) + 15 * lam_d_masked + 15
        term = term * exp_lam_d[..., mask_index, :, :] / (18 * lamda_masked.square() + EPS)
        res = sig2_div_beta[mask_index] * term
        output[..., mask_index, :, :] = res

    return output

