import math
import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.utils.memoize import cached, clear_cache_hook
from linear_operator.operators import LowRankRootLinearOperator, DiagLinearOperator

from intdom_dgp.linear_op_utils.block_diag_Kuu import LowRankRootAddedDiagKuu, BlockDiagKuu


class VFFKernel(Kernel):
    """
    Kernel for K_uu, the batch shape of this kernel follows the base matern kernel Kxx
    """
    def __init__(
            self,
            a, b, M,
            scaled_matern_kernel,
            boundary_learnable=False, **kwargs
    ):
        super(VFFKernel, self).__init__(ard_num_dims=1, **kwargs)
        # for param transferring, saving, and reloading
        self.register_buffer('boundary_learnable', torch.tensor(boundary_learnable))
        self.register_buffer('M', torch.as_tensor(M))
        self.num_features = 2 * self.M + 1

        if not boundary_learnable:
            self.register_buffer('a', torch.as_tensor(a))
            self.register_buffer('b', torch.as_tensor(b))
        else:
            self.register_parameter('a', torch.nn.Parameter(torch.as_tensor(a)))
            self.register_parameter('b', torch.nn.Parameter(torch.as_tensor(b)))

        self.scaled_matern_kernel = scaled_matern_kernel
        self.register_buffer('nu', torch.as_tensor(self.scaled_matern_kernel.base_kernel.nu))

    @property   # batch shape indicates output dims and whether the base kernel shares across D_out
    def batch_shape(self) -> torch.Size:  # override Kernel.batch_shape
        return self.scaled_matern_kernel.base_kernel.batch_shape  # can be [D_out=l] or [] if base matern is shared

    @property
    def lengthscale(self) -> Tensor:  # override Kernel.lengthscale
        return self.scaled_matern_kernel.base_kernel.lengthscale

    @property
    def lamda(self) -> Tensor:
        return torch.sqrt(2. * self.nu) / self.lengthscale

    @property
    def outputscale(self) -> Tensor:
        return self.scaled_matern_kernel.outputscale

    @property
    def vff_cos(self):  # [M+1]
        index = torch.arange(start=0, end=self.M + 1,
                             dtype=torch.get_default_dtype(), device=self.device)  # Kernel has `device` property
        vff_non_expand = 2. * math.pi * index / (self.b - self.a)
        return vff_non_expand

    @property
    def vff_sin(self):
        return self.vff_cos[1:]

    @property
    @cached(name='z_cos_memo', ignore_args=True)  # save computation if a/b aren't learnable
    def vff_cos_expand_batch(self):  # [(D_out=l), M+1]
        return self.vff_cos.expand(self.batch_shape + torch.Size([-1]))

    @property
    @cached(name='z_sin_memo', ignore_args=True)
    def vff_sin_expand_batch(self):
        return self.vff_sin.expand(self.batch_shape + torch.Size([-1]))

    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    def __call__(self, **params) -> BlockDiagKuu:
        """
        :return: linear operator with shape [(D_out), 2M+1, 2M+1]
        """
        if self.boundary_learnable and self.training:
            self._clear_cache()  # clear `z` stored in `self._memoize_cache` when training

        lamb = self.lamda
        sig2 = self.outputscale.view(self.outputscale.shape + torch.Size([1, 1]))  # [(l), 1, 1]

        z_cos_block = self.vff_cos_expand_batch.unsqueeze(-1)  # [(l), M(+1), 1]; search in cache
        z_sin_block = self.vff_sin_expand_batch.unsqueeze(-1)

        def construct_low_rank_beta_1():
            beta_vector = torch.ones_like(z_cos_block) / sig2.sqrt()  # [(l), M+1, 1]
            return LowRankRootLinearOperator(beta_vector)

        def construct_low_rank_beta_2(z_sin):
            beta_vector = z_sin / (lamb * sig2.sqrt())
            return LowRankRootLinearOperator(beta_vector)

        def construct_low_rank_beta_12(z_cos):
            beta_vector_1 = torch.ones_like(z_cos_block) / sig2.sqrt()  # [(l), M+1, 1]
            coeff = 1. / torch.sqrt(8. * sig2)
            beta_vector_2 = coeff * (3. * torch.square(z_cos / lamb) - 1.)
            beta_vector = torch.cat([beta_vector_1, beta_vector_2], dim=-1)  # [(l), M+1, 2]
            return LowRankRootLinearOperator(beta_vector)

        if self.nu == 0.5:
            two_or_four = torch.where(z_cos_block == 0., 2., 4.)
            d_cos = (self.b - self.a) * (lamb.square() + z_cos_block.square()) / (two_or_four * sig2 * lamb)
            d_cos = DiagLinearOperator(d_cos.squeeze(-1))
            mat_beta_1 = construct_low_rank_beta_1()
            # A = mat_beta_1 + d_cos     # pay attention to the addition order
            A = LowRankRootAddedDiagKuu(mat_beta_1, d_cos)

            d_sin = (self.b - self.a) * (lamb.square() + z_sin_block.square()) / (4. * sig2 * lamb)
            d_sin = DiagLinearOperator(d_sin.squeeze(-1))  # diag([(l), M(+1)])
            return BlockDiagKuu(A, d_sin)

        elif self.nu == 1.5:
            four_or_eight = torch.where(z_cos_block == 0., 4., 8.)
            d_cos = (self.b - self.a) * torch.square(lamb.square() + z_cos_block.square()) / (
                    four_or_eight * sig2 * lamb.pow(3))
            d_cos = DiagLinearOperator(d_cos.squeeze(-1))
            mat_beta_1 = construct_low_rank_beta_1()
            A = LowRankRootAddedDiagKuu(mat_beta_1, d_cos)

            d_sin = (self.b - self.a) * torch.square(lamb.square() + z_sin_block.square()) / (
                    8. * sig2 * lamb.pow(3))
            d_sin = DiagLinearOperator(d_sin.squeeze(-1))
            mat_beta_2 = construct_low_rank_beta_2(z_sin_block)
            B = LowRankRootAddedDiagKuu(mat_beta_2, d_sin)
            return BlockDiagKuu(A, B)

        elif self.nu == 2.5:
            or_16_32 = torch.where(z_cos_block == 0., 16., 32.)
            d_cos = 3. * (self.b - self.a) * (lamb.square() + z_cos_block.square()).pow(3) / (
                    or_16_32 * sig2 * lamb.pow(5))
            d_cos = DiagLinearOperator(d_cos.squeeze(-1))
            mat_beta_1 = construct_low_rank_beta_12(z_cos_block)
            A = LowRankRootAddedDiagKuu(mat_beta_1, d_cos)

            d_sin = 3. * (self.b - self.a) * (lamb.square() + z_sin_block.square()).pow(3) / (
                    32. * sig2 * lamb.pow(5))
            d_sin = DiagLinearOperator(d_sin.squeeze(-1))
            mat_beta_2 = construct_low_rank_beta_2(torch.sqrt(torch.as_tensor(3.)) * z_sin_block)
            B = LowRankRootAddedDiagKuu(mat_beta_2, d_sin)
            return BlockDiagKuu(A, B)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _build_mask_Kvf(vffkernel, z_cos_block, x_T, a, b):
    # Masks for different conditions
    mask_lt_a = x_T < a
    mask_gt_b = x_T > b
    mask_a_b = (~ mask_lt_a) & (~ mask_gt_b)
    # Initialize Kvf
    tmp_shape = torch.Size([vffkernel.num_features, 1])
    z_shape = torch.Size(z_cos_block.shape[:-2] + tmp_shape)
    K_shape = torch.broadcast_shapes(z_shape, x_T.shape)
    Kvf = torch.zeros(K_shape, device=z_cos_block.device)  # default `cpu`
    return Kvf, mask_a_b, mask_lt_a, mask_gt_b



