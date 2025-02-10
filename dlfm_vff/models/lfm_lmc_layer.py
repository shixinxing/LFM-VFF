from typing import Union, Optional
import torch

from intdom_dgp_lmc.models.layer_idgp_lmc import IDGP_LMC_Layer, LMCLayer
from intdom_dgp_lmc.models._idgp_lmc_layer import _IDGP_LMC_Layer
from intdom_dgp.models.layer_normalization import (
    Affine_SameAsInput_Normalization, Batch_MaxMin_Normalization, MaxMin_MovingAverage_LayerNormalizationMaxMin,
    BatchNorm1d_LayerNormalization, Tanh_Normalization
)

from dlfm_vff.models.lfm_vff_latent_gp import LFM_VFF_LatentGP
from dlfm_vff.models.lfm_rff_latent_gp import LFM_RFF_LatentGP


class LFM_VFF_LMC_Layer(IDGP_LMC_Layer):
    def __init__(
            self,
            input_dims: int, output_dims: int,
            a, b, M, nu,
            mean_x_type='linear', mean_z_type='zero', ode_type='ode1',
            whitened: str = 'none',
            normalization: Union[
                None,
                Affine_SameAsInput_Normalization,
                Batch_MaxMin_Normalization,
                MaxMin_MovingAverage_LayerNormalizationMaxMin,
                BatchNorm1d_LayerNormalization,
                Tanh_Normalization
            ] = None,  # before the layer
            kernel_sharing_across_dims: bool = False,
            has_kernel_noise=True, boundary_learnable=False,
            jitter_val=None,
    ):
        latent_idgp = LFM_VFF_LatentGP(
            input_dims, a, b, M, nu, mean_x_type, mean_z_type, ode_type,
            whitened, kernel_sharing_across_dims, has_kernel_noise, boundary_learnable, jitter_val
        )
        lmc_layer = LMCLayer(input_dims, output_dims)
        super(IDGP_LMC_Layer, self).__init__(latent_idgp, lmc_layer, normalization=normalization)


class LFM_RFF_LMC_Layer(_IDGP_LMC_Layer):
    """
    RFF LFM -> LMC
    """
    def __init__(
            self,
            input_dims: int, output_dims: int,
            nu: float,
            Z: torch.Tensor,
            num_samples=100,
            mean_x_type='linear-fix-identity',
            whitened: str = 'none',
            kernel_sharing_across_dims: bool = False,
            integral_lower_bound: Optional[float] = None,
            has_kernel_noise=True, learning_inducing_locations=True,
            jitter_val=None
    ):
        latent_lfm = LFM_RFF_LatentGP(
            input_dims, nu, Z, num_samples, whitened, mean_x_type,
            learning_inducing_locations=learning_inducing_locations,
            kernel_sharing_across_dims=kernel_sharing_across_dims,
            integral_lower_bound=integral_lower_bound,
            has_kernel_noise=has_kernel_noise,
            jitter_val=jitter_val
        )
        lmc_layer = LMCLayer(input_dims, output_dims)
        super().__init__(latent_lfm, lmc_layer, normalization=None)
        # alias
        self.latent_lfm = self.latent_idgp


