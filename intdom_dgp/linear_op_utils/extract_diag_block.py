from typing import Union
import torch
from torch import Tensor
from linear_operator.operators import (
    LinearOperator, BlockDiagLinearOperator, AddedDiagLinearOperator,
    DiagLinearOperator, to_linear_operator
)


def _extract_lop_block_dig(
        matrix_nt: BlockDiagLinearOperator,
        num_task: int,
        interleaved: bool = True
):
    """
    Particularly extract the BlockDiagLinearOperator: [Nt, Nt] -> [t, N, N] or [N, t, t]
    """
    task_covar = matrix_nt.base_linear_op  # [S, N, t, t] or [S, t, N, N]
    if interleaved:
        assert task_covar.size(-1) == num_task
    else:
        assert task_covar.size(-3) == num_task
    return task_covar


# adapt from gpytorch.MultitaskMultivariateNormal.to_data_independent_dist()
def extract_diag_block(
        matrix_nt: Union[Tensor, LinearOperator],
        num_task: int,
        interleaved: bool = True
):
    """
    Extracts t x t diagonal blocks from the input tensor.
    :param matrix_nt: The input tensor with shape [S, N*t, N*t].
    :param num_task: The size of the square blocks to be extracted.
    :param interleaved: task first.
    :returns: matrix containing the extracted blocks,
            with shape [S, N, t, t] if interleaved, otherwise [S, t, N, N].
    """
    if isinstance(matrix_nt, BlockDiagLinearOperator):
        return _extract_lop_block_dig(matrix_nt, num_task, interleaved)

    # BlockDiagLOP(DiagLOP [(S,) t*N, t*N]) will result a DiagLOP instead of BlockDiag
    elif isinstance(matrix_nt, DiagLinearOperator):
        diag = matrix_nt.diagonal(dim1=-2, dim2=-1)  # [S, t*N]
        if interleaved:  # [S, N*t]
            res = diag.view(*diag.shape[:-1], -1, num_task)  # [S, N, t]
        else:
            res = diag.view(*diag.shape[:-1], num_task, -1)  # [S, t, N]
        return DiagLinearOperator(res)

    elif isinstance(matrix_nt, AddedDiagLinearOperator) and isinstance(matrix_nt._linear_op, BlockDiagLinearOperator):  # noqa
        # used for cov after likelihood noise
        block_diag_lop = matrix_nt._linear_op
        diag_lop = matrix_nt._diag_tensor  # can be specifically KroneckerProdDiagLOP inherited from DiagLOP

        base_diag_lop = _extract_lop_block_dig(block_diag_lop, num_task, interleaved)  # [N, t, t] or [t, N, N]
        noise_diag_lop = diag_lop.diagonal(dim1=-1, dim2=-2)  # [..., N*t]
        if interleaved:
            noise_diag_lop = noise_diag_lop.view(*noise_diag_lop.shape[:-1], -1, num_task)  # [N, t]
        else:
            noise_diag_lop = noise_diag_lop.view(*noise_diag_lop.shape[:-1], num_task, -1)  # [t, N]

        base_diag_lop = to_linear_operator(base_diag_lop).add_diagonal(noise_diag_lop)
        return base_diag_lop

    else:
        num_data = matrix_nt.shape[-1] // num_task
        if interleaved:
            data_indices = torch.arange(0, num_data * num_task, num_task, device=matrix_nt.device).view(-1, 1, 1)
            task_indices = torch.arange(0, num_task, device=matrix_nt.device)
        else:
            data_indices = torch.arange(0, num_data, device=matrix_nt.device).view(-1, 1, 1)
            task_indices = torch.arange(0, num_data * num_task, num_data, device=matrix_nt.device)
        task_covars = matrix_nt.to_dense()[
            ..., data_indices + task_indices.unsqueeze(-1), data_indices + task_indices.unsqueeze(-2)
        ]
        return task_covars




