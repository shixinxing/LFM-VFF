from typing import Union, Optional
import torch
from linear_operator.operators import (
    LinearOperator, DiagLinearOperator, LowRankRootAddedDiagLinearOperator,
    CholLinearOperator, RootLinearOperator
)
from torch import Tensor


class LowRankRootAddedDiagKuu(LowRankRootAddedDiagLinearOperator):
    def trace_KiX(self, X: Union[Tensor, LinearOperator]):
        """
        K^{-1} = A^{-1} - A^{-1}W [I + W^TA^{-1}W]^{-1} W^{T}A^{-1}
        compute tr(K^{-1} X) = tr(A^{-1}X) - tr(A^{-1}W [I + W^TA^{-1}W]^{-1} W^{T}A^{-1}X)
        where I + W^TA^{-1}W = LL^T
        """
        A_inv = self._diag_tensor.inverse()  # Diag LOP
        X_diag = torch.diagonal(X, dim1=-2, dim2=-1)  # X can be CholLOP/RootLOP
        trace_AiX = torch.sum(A_inv.diagonal(dim1=-2, dim2=-1) * X_diag, dim=-1)

        W = self._linear_op.root
        AiW = A_inv.matmul(W)  # utilize DiagLOP; got DenseLOP
        LiWTAi = torch.linalg.solve_triangular(self.chol_cap_mat, AiW.mT.to_dense(), upper=False)  # (Tensor ,Tensor)
        trace_2 = torch.sum(LiWTAi * (LiWTAi @ X), dim=(-2, -1))

        return trace_AiX - trace_2

    def add_jitter(self, jitter_val: float = 1e-3) -> 'LowRankRootAddedDiagKuu':  # forward reference
        return super().add_jitter(jitter_val)  # noqa, return the same class

    def add_diagonal(self, diag) -> 'LowRankRootAddedDiagKuu':
        return super().add_diagonal(diag)  # noqa, return the same class

    def matmul_sqrt(self, lhs: Union[Tensor, LinearOperator]):
        """
        L = [D^{1/2}, W], we have LL^T = Kuu
        :return: lhs.matmul(L) = [lhsD^{1/2}, lhsW]
        """
        diag = self._diag_tensor.sqrt().diagonal(dim1=-2, dim2=-1).unsqueeze(-2)
        left = lhs * diag
        right = lhs.matmul(self._linear_op.root)
        return torch.cat([left, right], dim=-1)

    def sqrt(self):
        """
        :return: L = [D^{1/2}, beta], LL^T = Kuu
        """
        diag_mat = self._diag_tensor.sqrt().to_dense()
        beta = self._linear_op.root.to_dense()
        return torch.cat([diag_mat, beta], dim=-1)

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # adapt from the parent class; the parent class doesn't allow `inv_quad_rhs` to broadcast
        inv_quad_term, logdet_term = None, None

        if inv_quad_rhs is not None:
            self_inv_rhs = self._solve(inv_quad_rhs)
            inv_quad_term = (inv_quad_rhs * self_inv_rhs).sum(dim=-2)
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(dim=-1)

        if logdet:
            logdet_term = self._logdet()

        return inv_quad_term, logdet_term


class BlockDiagKuu(LinearOperator):
    """
    LOP = [Block_A, 0,
           0,  Block_B], with Block_A: [batch_shape, D1, D1], Block_B: [batch_shape, D2, D2], Diag or LowRankAddedDiag
    """
    def __init__(
            self,
            A: Union[LowRankRootAddedDiagKuu, LowRankRootAddedDiagLinearOperator],
            B: Union[LowRankRootAddedDiagKuu, LowRankRootAddedDiagLinearOperator, DiagLinearOperator]
    ):
        self.A = A
        self.B = B
        self.D1 = A.size(-1)
        self.D2 = B.size(-1)
        super(BlockDiagKuu, self).__init__(A, B)  # will call _check_args()

    def _check_args(self, *args, **kwargs) -> Union[str, None]:
        try:
            if (self.D1 != self.A.size(-2)) or (self.D2 != self.B.size(-2)):
                raise RuntimeError("The blocks A/B must be square.")
            elif self.A.shape[:-2] != self.B.shape[:-2]:
                raise RuntimeError("The blocks A/B have incompatible batch shapes. Only accept the same batch shape.")
            return None
        except RuntimeError as error:
            return str(error)

    @property
    def batch_shape(self) -> torch.Size:
        return self.A.shape[:-2]

    def _size(self) -> torch.Size:
        batch_shape = self.batch_shape
        last_shape = self.D1 + self.D2
        return torch.Size((*batch_shape, last_shape, last_shape))

    def _get_rhs_slices(self, rhs):
        # slice by rows
        rhs_A = rhs[..., :self.D1, :]
        rhs_B = rhs[..., self.D1:, :]
        return rhs_A, rhs_B

    def _get_lhs_slices(self, lhs):
        # slice by columns
        lhs_A = lhs[..., :self.D1]
        lhs_B = lhs[..., self.D1:]
        return lhs_A, lhs_B

    def _matmul(self, rhs):
        rhs_A, rhs_B = self._get_rhs_slices(rhs)
        res_A = self.A @ rhs_A
        res_B = self.B @ rhs_B
        return torch.cat([res_A, res_B], dim=-2)

    def _mul_constant(self, other):  # used in VI: (S - Kuu)
        return LinearOperator._mul_constant(self, other)

    def _transpose_nonbatch(self):
        return self.__class__(self.A.mT, self.B.mT)

    def _solve(self, rhs, *kwargs):
        if rhs.ndim == 1:
            rhs = rhs.unsqueeze(-1)
        rhs_A, rhs_B = self._get_rhs_slices(rhs)
        solve_A = self.A._solve(rhs_A, *kwargs)
        solve_B = self.B._solve(rhs_B, *kwargs)
        return torch.cat([solve_A, solve_B], dim=-2)

    def inverse(self):
        raise NotImplementedError

    def _diagonal(self):  # = LOP.diagonal(dim1=-2, dim2=-1)
        diag_A = self.A.diagonal()  # Tensor
        diag_B = self.B.diagonal()
        return torch.cat([diag_A, diag_B], dim=-1)  # return Tensor

    def solve(self, right_tensor, left_tensor=None):
        # compute `K_uu^{-1} R` or `L K_uu{-1} R`
        right_tensor_A, right_tensor_B = self._get_rhs_slices(right_tensor)
        solve_A = self.A.solve(right_tensor_A.to_dense())  # A^{-1}R_a and allow batch shape broadcast
        solve_B = self.B.solve(right_tensor_B.to_dense())
        inv_R = torch.cat([solve_A, solve_B], dim=-2)

        if left_tensor is not None:
            return left_tensor @ inv_R
        return inv_R

    def trace_KiX(self, X: Union[LinearOperator, Tensor]):  # used in KL
        # slice
        if isinstance(X, CholLinearOperator):  # CholLOP[index, index] is not implemented
            tril_tensor = X.root.to_dense()
            tl = CholLinearOperator(tril_tensor[..., :self.D1, :self.D1])
            br = RootLinearOperator(tril_tensor[..., self.D1:, :])
        else:
            tl = X[..., :self.D1, :self.D1]
            br = X[..., self.D1:, self.D1:]

        top = self.A.trace_KiX(tl)
        if isinstance(self.B, DiagLinearOperator):
            B_inv = self.B.inverse()
            br_diag = torch.diagonal(br, dim1=-2, dim2=-1)
            bottom = torch.sum(B_inv.diagonal(dim1=-1, dim2=-2) * br_diag, dim=-1)
        else:
            bottom = self.B.trace_KiX(br)
        return top + bottom

    def inv_quad_logdet(   # LOP.logdet() will also call this method.
            self,
            inv_quad_rhs=None,
            logdet: Optional[bool] = False,
            reduce_inv_quad: Optional[bool] = True,  # if True, return trace
    ):  # compute tr{R^T K^{-1} R} or diag{R^T K^{-1} R} and log|K|
        inv_quad_term, logdet_term = None, None
        if inv_quad_rhs is not None:
            inv_quad_rhs_A, inv_quad_rhs_B = self._get_rhs_slices(inv_quad_rhs)

        else:
            inv_quad_rhs_A, inv_quad_rhs_B = None, None

        inv_quad_A, logdet_A = self.A.inv_quad_logdet(inv_quad_rhs_A, logdet=logdet, reduce_inv_quad=reduce_inv_quad)
        inv_quad_B, logdet_B = self.B.inv_quad_logdet(inv_quad_rhs_B, logdet=logdet, reduce_inv_quad=reduce_inv_quad)

        if inv_quad_rhs is not None:
            inv_quad_term = inv_quad_A + inv_quad_B  # trace or diagonal elements

        if logdet:
            logdet_term = logdet_A + logdet_B

        return inv_quad_term, logdet_term

    def add_jitter(self, jitter_val: float = 1e-3):
        A_jitter = self.A.add_jitter(jitter_val)
        B_jitter = self.B.add_jitter(jitter_val)
        return self.__class__(A_jitter, B_jitter)

    def add_diagonal(self, diag: Tensor):  # diag: [b, D1 + D2 or 1]
        if diag.ndim == 0:
            return self.add_jitter(diag)
        if diag.shape[-1] == 1:
            add_diag_A = diag
            add_diag_B = diag
        else:
            add_diag_A = diag[..., :self.D1]
            add_diag_B = diag[..., self.D1:]
        A_new = self.A.add_diagonal(add_diag_A)
        B_new = self.B.add_diagonal(add_diag_B)
        return self.__class__(A_new, B_new)

    def matmul_sqrt(self, lhs: Union[Tensor, LinearOperator]):
        """
        :return: lhs.matmul([L_a, 0, \\ 0, L_b])
        """
        lhs_A, lhs_B = self._get_lhs_slices(lhs)
        left = self.A.matmul_sqrt(lhs_A)
        right = self.B.matmul_sqrt(lhs_B)
        return torch.cat([left, right], dim=-1)

    def sqrt(self):
        A_sqrt = self.A.sqrt()
        B_sqrt = self.B.sqrt()
        top = torch.cat([A_sqrt, torch.zeros(*self.batch_shape, self.D1, B_sqrt.size(-1))], dim=-1)
        bottom = torch.cat([torch.zeros(*self.batch_shape, self.D2, A_sqrt.size(-1)), B_sqrt], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def to_dense(self) -> Tensor:
        tl = self.A.to_dense()
        br = self.B.to_dense()
        top = torch.cat([tl, torch.zeros(*self.batch_shape, self.D1, self.D2, device=tl.device)], dim=-1)
        bottom = torch.cat([torch.zeros(*self.batch_shape, self.D2, self.D1, device=br.device), br], dim=-1)
        return torch.cat([top, bottom], dim=-2)

