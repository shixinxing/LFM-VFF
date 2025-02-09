import torch
from torch import Tensor

from gpytorch.kernels import Kernel
from linear_operator import to_linear_operator

from intdom_dgp.kernels.intdom_gp_kernels import VFFKernel
from dlfm_vff.kernels.lfm_kernels import LFM_MaternKernel1d
from dlfm_vff.kernels.kernel_functions.func_MKG_matern12 import lfm_matern12_cross_cov
from dlfm_vff.kernels.kernel_functions.func_MKG_matern32 import lfm_matern32_cross_cov
from dlfm_vff.kernels.kernel_functions.func_MKG_matern52 import lfm_matern52_cross_cov
from dlfm_vff.kernels.kernel_functions.func_GK_matern12 import lfm_matern12_green_k
from dlfm_vff.kernels.kernel_functions.func_GK_matern32 import lfm_matern32_green_k
from dlfm_vff.kernels.kernel_functions.func_GK_matern52 import lfm_matern52_green_k


class LFM_VFF_InterDomainKernel(Kernel):
    """
    K_{vf} induced by VFFs and LFMs
    :return: [(S, l), 2M+1, 2M(+1)]
    """
    def __init__(self,
                 lfm_kernel: LFM_MaternKernel1d,
                 vff_kernel: VFFKernel,
                 **kwargs):
        super().__init__(**kwargs)   # This will set the batch shape equal to that of base Matern kernel
        self.lfm_kernel = lfm_kernel
        self.vff_kernel = vff_kernel
        self.nu = self.vff_kernel.nu

    @property
    def batch_shape(self) -> torch.Size:
        return self.lfm_kernel.batch_shape

    def __call__(self, x: Tensor, **params):
        """
        :param: x: [(S), l, N, d=1], with z:[l, 2M+1, d=1]
        :return: linear operator with [(S), l, 2M+1, N]
        """
        assert x.size(-1) == 1
        a, b = self.vff_kernel.a, self.vff_kernel.b
        lamb = self.lfm_kernel.lamda
        beta = self.lfm_kernel.beta.unsqueeze(-1).unsqueeze(-1)
        gama = self.lfm_kernel.gama.unsqueeze(-1).unsqueeze(-1)

        x_T = x.transpose(-1, -2)  # [(S, l) d=1, N]
        z_cos_block = self.vff_kernel.vff_cos_expand_batch.unsqueeze(-1)  # [(l), M+1, 1]
        if self.nu == 0.5:
            Kvf = lfm_matern12_cross_cov(z_cos_block, x_T, a, b, gama, beta, lamb)
        elif self.nu == 1.5:
            Kvf = lfm_matern32_cross_cov(z_cos_block, x_T, a, b, gama, beta, lamb)
        elif self.nu == 2.5:
            Kvf = lfm_matern52_cross_cov(z_cos_block, x_T, a, b, gama, beta, lamb)
        else:
            raise ValueError(f"Unknown nu: {self.nu}.")
        return to_linear_operator(Kvf)  # [...,t, 2M+1, N]

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class LFM_InterDomainKernels(Kernel):
    """
    Kuf by conventional latent force model: k (z, x) * G
    """
    def __init__(self, lfm_kernel: LFM_MaternKernel1d, **kwargs):
        super(LFM_InterDomainKernels, self).__init__(**kwargs)
        self.lfm_kernel = lfm_kernel
        self.nu = lfm_kernel.nu
        self.ode_type = lfm_kernel.ode_type

    @property
    def batch_shape(self) -> torch.Size:
        return self.lfm_kernel.batch_shape

    def __call__(self, z: Tensor, x: Tensor, **params):  # noqa
        """
        :param z, x: [(S), D_in, N, 1]
        :return: k (z, x) * G
        """
        # first compute G * k (x (t1), z (t2))
        assert z.size(-1) == 1 and z.size(-1) == 1
        lamb = self.lfm_kernel.lamda
        gama = self.lfm_kernel.gama.unsqueeze(-1).unsqueeze(-1)
        beta = self.lfm_kernel.beta.unsqueeze(-1).unsqueeze(-1)
        outputscale = self.lfm_kernel.outputscale.unsqueeze(-1).unsqueeze(-1)

        distance = self.covar_dist(x, z, diag=False, square_dist=False, **params)  # |t1 - t2|
        if len(distance.shape) < len(gama.shape):  # just for test, this will never happen in practice
            brdcst_shape = torch.broadcast_shapes(gama.shape, distance.shape)
            distance = distance.expand(*brdcst_shape)

        if self.nu == 0.5:
            output = lfm_matern12_green_k(x, z, distance, gama, beta, outputscale, lamb)
        elif self.nu == 1.5:
            output = lfm_matern32_green_k(x, z, distance, gama, beta, outputscale, lamb)
        elif self.nu == 2.5:
            output = lfm_matern52_green_k(x, z, distance, gama, beta, outputscale, lamb)
        else:
            raise ValueError(f"Wrong nu: {self.nu}.")

        return to_linear_operator(output.mT)

    def forward(*args, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    from gpytorch.kernels import MaternKernel, ScaleKernel

    def test_batched_mkg_kernel(nu=0.5):
        scaled_matern_kernel = ScaleKernel(MaternKernel(nu=nu, batch_shape=torch.Size([2])))
        lfm_kernel = LFM_MaternKernel1d(scaled_matern_kernel)
        a, b, M = -2, 3, 4
        vff_kernel = VFFKernel(a, b, M, scaled_matern_kernel)
        cross_kernel = LFM_VFF_InterDomainKernel(lfm_kernel, vff_kernel)

        alpha = torch.tensor([0.5, 3.]).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tensor([0.01, 1.]).unsqueeze(-1).unsqueeze(-1)
        lam = torch.tensor([0.15, 3.]).unsqueeze(-1).unsqueeze(-1)  # lam = gama = alpha / beta
        lfm_kernel.alpha, lfm_kernel.beta, lfm_kernel.lamda = alpha, beta, lam

        t_lt_a = torch.linspace(a - 6, a - 0.5, steps=4).unsqueeze(-1).repeat(2, 1, 1)  # [2, 4, 1]
        res_lt_a = cross_kernel(t_lt_a)
        print(f"===== nu: {nu}, M: {M}, batch shape: {torch.Size([2])} =====")
        print(f"t < {a}")
        print(f"t (shape {t_lt_a.shape}) = \n{t_lt_a}")
        print(f"Kzx (shape:{res_lt_a.shape}): \n{res_lt_a.to_dense()}\n")

        t_ab = torch.linspace(a, b, steps=4).unsqueeze(-1).repeat(2, 1, 1)
        res_ab = cross_kernel(t_ab)
        print(f"{a} < t < {b}")
        print(f"t (shape:{t_ab.shape}) = \n{t_ab}")
        print(f"Kzx (shape:{res_ab.shape}): \n{res_ab.to_dense()}\n")

        t_gt_b = torch.linspace(b + 0.3, b + 6.3, steps=4).unsqueeze(-1).repeat(2, 1, 1)
        res_gt_b = cross_kernel(t_gt_b)
        print(f"t > {b}")
        print(f"t (shape:{t_gt_b.shape}) : \n{t_gt_b}")
        print(f"Kzx (shape:{res_gt_b.shape}): \n{res_gt_b.to_dense()}\n")

        t_all = torch.cat([t_lt_a[:, :2, :], t_ab[:, :2, :], t_gt_b[:, :2, :]], dim=-2)
        res = cross_kernel(t_all)
        print(f"all t: ")
        print(f"t (shape:{t_all.shape}): \n{t_all}")
        print(f"Kzx (shape:{res.shape}): \n{res.to_dense()}\n")

        x = torch.randn(3, 2, 5, 1)
        res = cross_kernel(x)
        print(f"check shape: x (shape {x.shape}), Kzx shape {res.shape}")


    def test_batched_green_kernel(nu=0.5):
        scaled_matern_kernel = ScaleKernel(MaternKernel(nu=nu, batch_shape=torch.Size([2])))
        lfm_kernel = LFM_MaternKernel1d(scaled_matern_kernel)
        cross_kernel = LFM_InterDomainKernels(lfm_kernel)

        outputscale = torch.tensor([0.2, 6])
        alpha = torch.tensor([0.5, 3.]).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tensor([0.02, 1.]).unsqueeze(-1).unsqueeze(-1)
        lam = torch.tensor([0.15, 3.+1e-4]).unsqueeze(-1).unsqueeze(-1)  # lam = gama = alpha / beta
        lfm_kernel.alpha, lfm_kernel.beta, lfm_kernel.lamda, lfm_kernel.outputscale = alpha, beta, lam, outputscale

        print(f"===== nu: {nu}, batch shape {torch.Size([2])} =====")

        x = torch.linspace(-5, 4.14, 4).unsqueeze(-1).repeat(2, 1, 1)
        z = torch.linspace(4.14, 10.26, 5).unsqueeze(-1).repeat(2, 1, 1)
        print(f"===== x <= z =====")
        res_x_lt_z = cross_kernel(z, x)
        print(f"x (shape {x.shape}): \n {x}")
        print(f"z (shape {z.shape}): \n {z}")
        print(f"Kzx (shape {res_x_lt_z.shape}: \n{res_x_lt_z.to_dense()}\n")

        x = z
        z = torch.linspace(-5, 4.14, 4).unsqueeze(-1).repeat(2, 1, 1)
        print(f"===== x >= z =====")
        res_x_gt_z = cross_kernel(z, x)
        print(f"x (shape {x.shape}): \n {x}")
        print(f"z (shape {z.shape}): \n {z}")
        print(f"Kzx (shape {res_x_gt_z.shape}: \n{res_x_gt_z.to_dense()}\n")

        zz = torch.linspace(-5, 4.14, 4).unsqueeze(-1).repeat(2, 1, 1)
        z[:, :2, :] = x[:, :2, :]
        x[:, :2, :] = zz[:, :2, :]
        res = cross_kernel(z, x)
        print(f"===== x mix z =====")
        print(f"x (shape {x.shape}): \n{x}")
        print(f"z (shape {z.shape}): \n{z}")
        print(f"Kzx (shape {res.shape}): \n{res.to_dense()}")


    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=5, linewidth=120)
    torch.manual_seed(0)

    # test_batched_mkg_kernel(nu=0.5)
    # test_batched_mkg_kernel(nu=1.5)
    test_batched_mkg_kernel(nu=2.5)

    # test_batched_green_kernel(nu=0.5)
    # test_batched_green_kernel(nu=1.5)
    # test_batched_green_kernel(nu=2.5)


