from typing import Optional
import torch
from gpytorch.models import ApproximateGP
from gpytorch.means import ZeroMean, LinearMean

from intdom_dgp_lmc.means.mean_z_identity_x import MeanZ_IdentityX
from intdom_dgp.models.layer_normalization import (
    Affine_SameAsInput_Normalization, Tanh_Normalization,
    Batch_MaxMin_Normalization, MaxMin_MovingAverage_LayerNormalizationMaxMin,
    BatchNorm1d_LayerNormalization
)
from dlfm_vff.means.lfm_mean_z_identity_x import LFM_MeanZ_IdentityX_ODE1


def _build_mean_module(
        model: ApproximateGP,
        mean_x_type: str = 'zero',  # or 'linear-fix-identity'
        mean_z_type: str = 'zero',  # or 'with-x'
        model_type='LFM'
):   # model.num_latents = num of GPs before LMC
    if mean_x_type == 'zero':
        model.mean_module = ZeroMean(batch_shape=torch.Size([model.num_latents]))
    elif mean_x_type == 'linear-fix-identity':
        model.mean_module = LinearMean(input_size=1, batch_shape=torch.Size([model.num_latents]), bias=False)
        w = model.mean_module.weights  # [num_latents, input_size=1, 1]
        model.mean_module.initialize(weights=torch.ones_like(w))
        w.requires_grad = False
    else:
        raise NotImplementedError(f"invalid mean_type for x: {mean_x_type}.")

    if mean_z_type == 'zero':
        model.mean_z_module = ZeroMean(batch_shape=torch.Size([model.num_latents]))

    elif mean_z_type == 'with-x':
        if mean_x_type == 'zero':
            model.mean_z_module = ZeroMean(batch_shape=torch.Size([model.num_latents]))
        elif mean_x_type == 'linear-fix-identity':
            if model_type == 'IDGP':
                model.mean_z_module = MeanZ_IdentityX(model.kzz, batch_shape=torch.Size([model.num_latents]))
            elif model_type == 'LFM_ode1':
                model.mean_z_module = LFM_MeanZ_IdentityX_ODE1(
                    model.scaled_matern_or_lfm_kernel, model.vff_kernel, batch_shape=torch.Size([model.num_latents])
                )
            else:
                raise NotImplementedError(f"Wrong model type: {model_type} for prior mean x {mean_x_type}!")
        else:
            raise NotImplementedError(f"prior mean of z for mean_x_type {mean_x_type} is not implemented.")

    else:
        raise NotImplementedError(f"invalid mean_type for z: {mean_z_type}.")


def create_normalization_from_dict(
        hidden_normalization_info: Optional[dict],
        num_features: Optional[int] = None
) -> Optional[torch.nn.Module]:
    if hidden_normalization_info is None:
        return None

    if hidden_normalization_info['type'] == 'same-as-input':
        layer_normalization = Affine_SameAsInput_Normalization(
            hidden_normalization_info['preprocess_a'], hidden_normalization_info['preprocess_b'],
            hidden_normalization_info['min_tensor'], hidden_normalization_info['max_tensor']
        )
    elif hidden_normalization_info['type'] == 'tanh':
        layer_normalization = Tanh_Normalization(
            hidden_normalization_info['preprocess_a'], hidden_normalization_info['preprocess_b'],
            hidden_normalization_info['min_tensor'], hidden_normalization_info['max_tensor'],
            h_trainable=hidden_normalization_info['h_trainable']
        )
    elif hidden_normalization_info['type'] == 'minibatch-maxmin':
        layer_normalization = Batch_MaxMin_Normalization(
            hidden_normalization_info['preprocess_a'], hidden_normalization_info['preprocess_b'],
            track_gradients=hidden_normalization_info['track_gradients']
        )
    elif hidden_normalization_info['type'] == 'moving-average':
        layer_normalization = MaxMin_MovingAverage_LayerNormalizationMaxMin(
            hidden_normalization_info['preprocess_a'], hidden_normalization_info['preprocess_b'],
            num_features=num_features,
            track_gradients=hidden_normalization_info['track_gradients']
        )
    elif hidden_normalization_info['type'] == 'batch-norm':
        layer_normalization = BatchNorm1d_LayerNormalization(
            num_features=num_features
        )
    else:
        raise NotImplementedError

    return layer_normalization


