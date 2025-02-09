import math
import torch
from torch import Tensor
from gpytorch.kernels.kernel import dist


def get_eps(dtype):
    return 1e-10 if dtype == torch.float64 else 1e-4


def get_bound(dtype):
    return 1e-10 if dtype == torch.float64 else 1e-4


# GKG Computation
def construct_mask_gama_lam(output_shape, g_lam, bound=None):
    if bound is None:
        bound = get_bound(torch.get_default_dtype())
    output = torch.zeros(output_shape, dtype=torch.get_default_dtype(), device=g_lam.device)
    mask_index = torch.abs(g_lam) > bound
    mask_index = mask_index.squeeze(-1).squeeze(-1)  # get the mask along batch dim: [d'] or []
    return output, mask_index


def prepare_elements_gkg(distance, gama, beta, outputscale, lamda):
    lam_d, gama_d = lamda * distance, gama * distance
    sig2_beta2 = outputscale / beta.square()
    exp_lam_d, exp_gama_d = torch.exp(- lam_d), torch.exp(- gama_d)
    gama_sub_lam = gama - lamda
    return (
        lam_d, gama_d, sig2_beta2, exp_lam_d, exp_gama_d, gama_sub_lam
    )


# MKG computation
def prepare_elements_mkg(z_cos, t_T, a, b, gama, beta, lamda):
    r_a, r_b = torch.abs(t_T - a), torch.abs(t_T - b)
    exp_gama_r_a, exp_gama_r_b = torch.exp(- gama * r_a), torch.exp(- gama * r_b)
    exp_lam_r_a, exp_lam_r_b = torch.exp(- lamda * r_a), torch.exp(- lamda * r_b)
    z2_plus_gama2 = z_cos.square() + gama.square()
    return (
        r_a, r_b,
        exp_gama_r_a, exp_gama_r_b, exp_lam_r_a, exp_lam_r_b,
        z2_plus_gama2
    )


def cov_cos_a_b(z_cos, r_a, gama, z2_plus_gama2, coeff_reuse_cos, exp_gama_r_a) -> Tensor:
    theta = - torch.atan(z_cos / gama)
    cos = torch.cos(z_cos * r_a + theta) / torch.sqrt(z2_plus_gama2)
    term_2 = coeff_reuse_cos * exp_gama_r_a
    return cos + term_2


def cov_sin_a_b(z_sin, r_a, gama, z2_plus_gama2_sin, coeff_reuse_sin, exp_gama_r_a) -> Tensor:
    theta = - torch.atan(z_sin / gama)
    sin = torch.sin(z_sin * r_a + theta) / torch.sqrt(z2_plus_gama2_sin)
    term_2 = coeff_reuse_sin * exp_gama_r_a
    return sin + term_2


def buid_mask_gama_lam(gama, lamda):
    mask_index = torch.abs(gama - lamda) > get_bound(torch.get_default_dtype())
    mask_index = mask_index.squeeze(-1).squeeze(-1)
    return mask_index


def prepare_masked_elements(
        mask_g_lam,
        gama, lamda,
        exp_gama_r_a, exp_gama_r_b=None, exp_lam_r_b=None  # sometimes don't need exp(-gama * r_b)
):
    gama_masked, lamda_masked = gama[mask_g_lam], lamda[mask_g_lam]
    exp_gama_r_a_masked = exp_gama_r_a[..., mask_g_lam, :, :]
    res = [gama_masked, lamda_masked, exp_gama_r_a_masked]
    if exp_gama_r_b is not None:
        exp_gama_r_b_masked = exp_gama_r_b[..., mask_g_lam, :, :]
        res.append(exp_gama_r_b_masked)
    if exp_lam_r_b is not None:
        exp_lam_r_b_masked = exp_lam_r_b[..., mask_g_lam, :, :]
        res.append(exp_lam_r_b_masked)
    return tuple(res)


# GK Computation
def prepare_elements_green_k(distance, gama, beta, outputscale, lamda):
    lam_d, gama_d = lamda * distance, gama * distance
    sig2_beta = outputscale / beta
    exp_lam_d, exp_gama_d = torch.exp(- lam_d), torch.exp(- gama_d)
    return (
        lam_d, sig2_beta, exp_lam_d, exp_gama_d
    )


# MKG testing
def test_lfm_frag_mkg(func):
    a, b = -2, 3
    t_lt_a = torch.linspace(a - 6, a - 1, 6).view(2, 1, -1)  # [2, 1, 3]
    z_cos_block = torch.linspace(0, 3, steps=4).unsqueeze(-1).expand(2, -1, 1)  # [2, 4, 1]
    alpha = torch.tensor([0.5, 3.]).unsqueeze(-1).unsqueeze(-1)
    beta = torch.ones_like(alpha)
    gama = alpha / beta
    lam = torch.tensor([0.15, 3.]).unsqueeze(-1).unsqueeze(-1)
    res_lt_a = func(z_cos_block, t_lt_a, a, b, gama, beta, lam)
    print(f"t < a: {a} \nt (shape:{t_lt_a.shape}) = {t_lt_a}, \nz (shape:{z_cos_block.shape}) = {z_cos_block}")
    print(f"Kzx (shape:{res_lt_a.shape}): {res_lt_a}\n")

    b = 8
    t_ab = torch.linspace(1, 6, steps=6).view(2, 1, -1)
    res_ab = func(z_cos_block, t_ab, a, b, gama, beta, lam)
    print(f"a < t < b: \nt (shape:{t_ab.shape}) = {t_ab}, \nz (shape:{z_cos_block.shape}) = {z_cos_block}")
    print(f"Kzx (shape:{res_ab.shape}): {res_ab}\n")

    b = 3
    t_gt_b = torch.linspace(4, 9, steps=6).view(2, 1, -1)
    res_gt_b = func(z_cos_block, t_gt_b, a, b, gama, beta, lam)
    print(f"t > b {b}: \nt (shape:{t_gt_b.shape}) = {t_gt_b}, \nz (shape:{z_cos_block.shape}) = {z_cos_block}")
    print(f"Kzx (shape:{res_gt_b.shape}): {res_gt_b}\n")


# GK testing
def test_GK(nu, func):
    t1 = torch.linspace(-10, -4, 5).view(1, -1, 1).repeat(2, 1, 1)
    t2 = torch.linspace(-10, -4, 5).view(1, -1, 1).repeat(2, 1, 1)
    print(f"t1 {t1.shape}: {t1}")
    print(f"t2 {t2.shape}: {t2}")

    distance = dist(t1, t2)
    gama = torch.as_tensor([0.5, 0.15]).view(-1, 1, 1)
    beta = torch.as_tensor([0.3, 3]).view(-1, 1, 1)
    outputscale = torch.as_tensor([2., 3.]).view(-1, 1, 1)
    lengthscale = torch.as_tensor([0.2, math.sqrt(2. * nu) / gama[-1, 0, 0]]).view(-1, 1, 1)
    lam = math.sqrt(2. * nu) / lengthscale

    if nu == 0.5:
        res = func(t1, t2, distance, gama, beta, outputscale, lam)
        print(f"GK ({res.shape}):\n {res}\n")
    else:
        raise ValueError(f"nu :{nu}")

