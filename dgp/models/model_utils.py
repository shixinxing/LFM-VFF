import torch

from gpytorch.models import ApproximateGP
from gpytorch.constraints import Positive
from gpytorch.distributions import MultitaskMultivariateNormal

from linear_operator.operators import DiagLinearOperator


def _build_kernel_noise(model: ApproximateGP, noise_sharing_across_dims: bool = True):
    noise_batch = torch.Size([]) if noise_sharing_across_dims else torch.Size([model.num_latents])
    if model.has_kernel_noise:
        # 1e-5: -11.51; 1e-4: -9.21; 1e-2: -4.6
        model.register_parameter('raw_kernel_noise',
                                 torch.nn.Parameter(torch.ones(noise_batch) * (- 11.51)))  # [(l)]
        model.register_constraint('raw_kernel_noise', Positive())


def _add_kernel_noise(model, covar_x_or_z, full_cov):
    # can be: (1) Diag + Diag = DiagLOP; (2) BlockDiagLOP + Diag_noise = AddedDiagLOP
    if not full_cov:
        covar_x_or_z = DiagLinearOperator(covar_x_or_z)  # transform to LOP
    if model.has_kernel_noise:
        kernel_noise = model.raw_kernel_noise_constraint.transform(model.raw_kernel_noise)
        covar_x_or_z = covar_x_or_z.add_diagonal(kernel_noise.unsqueeze(-1))  # jitter shape[(l), 1]
    return covar_x_or_z


def create_dim_list(input_dims, output_dims, hidden_dims, num_layers, with_concat):
    input_dims_list = [input_dims]
    for i in range(num_layers - 1):
        if with_concat:
            input_dims_list.append(hidden_dims + input_dims)
        else:
            input_dims_list.append(hidden_dims)
    output_dims_list = [hidden_dims for _ in range(num_layers - 1)]
    output_dims_list.append(output_dims)
    return input_dims_list, output_dims_list


def unsqueeze_y_stat(y_mean, y_std, out_dims: int, device):
    if y_mean is None:
        y_mean = torch.zeros(1, out_dims, device=device)
    if y_std is None:
        y_std = torch.ones(1, out_dims, device=device)

    y_mean = torch.as_tensor(y_mean, device=device)
    y_std = torch.as_tensor(y_std, device=device)
    y_mean = y_mean.unsqueeze(-2) if y_mean.ndim == 1 else y_mean
    y_std = y_std.unsqueeze(-2) if y_std.ndim == 1 else y_std
    assert y_mean.shape == (1, out_dims) and y_std.shape == (1, out_dims), "y_mean and y_std must have output dims."

    return y_mean, y_std


def predict_mixture_mean_and_covar(component_means, component_covars):
    """
    :param component_means: [N, S, t]
    :param component_covars: [N, S, t, t], for each N, we have S components from preds
    :return: mixture mean [N, t], mixture covariance [N, t, t]
    """
    mixture_mean = component_means.mean(dim=-2)  # [N, t]

    E_X2 = (component_means.unsqueeze(-1) @ component_means.unsqueeze(-1).mT
            + component_covars.to_dense())  # [N, S, t, t]
    E_X2 = E_X2.mean(dim=-3)  # [N, t, t]
    EX_square = mixture_mean.unsqueeze(-1) @ mixture_mean.unsqueeze(-1).mT
    mixture_covar = E_X2 - EX_square
    return mixture_mean, mixture_covar


def mixture_mean_and_var_1d(dist: MultitaskMultivariateNormal):
    mean = dist.mean.squeeze(-1)  # [S, N]
    var = dist.variance.squeeze(-1)  # [S, N]

    mixture_mean = mean.mean(0)  # [N]
    E_X2 = mean.square() + var
    E_X2 = E_X2.mean(0)
    EX_square = mixture_mean.square()
    mixture_var = E_X2 - EX_square  # [N]
    return mixture_mean, mixture_var


