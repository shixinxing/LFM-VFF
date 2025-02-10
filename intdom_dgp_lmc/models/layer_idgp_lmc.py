from typing import Optional, Union
from torch import Tensor

from intdom_dgp_lmc.models.layer_vff_latent_gp import VFF_LatentGP
from intdom_dgp_lmc.models.layer_lmc import LMCLayer
from intdom_dgp_lmc.models._idgp_lmc_layer import _IDGP_LMC_Layer
from intdom_dgp.models.layer_normalization import (
    Affine_SameAsInput_Normalization, Batch_MaxMin_Normalization, MaxMin_MovingAverage_LayerNormalizationMaxMin,
    BatchNorm1d_LayerNormalization, Tanh_Normalization
)


class IDGP_LMC_Layer(_IDGP_LMC_Layer):
    """
    normalization -> Latent GPs -> LMC
    """
    def __init__(
            self,
            input_dims: int, output_dims: int,
            a, b, M, nu,
            mean_x_type='linear-fix-identity', mean_z_type='zero',
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
            jitter_val: Optional[float] = None,
    ):
        latent_idgp = VFF_LatentGP(
            input_dims, a, b, M, nu,
            mean_x_type, mean_z_type, whitened,
            kernel_sharing_across_dims, has_kernel_noise, boundary_learnable, jitter_val
        )
        lmc_layer = LMCLayer(input_dims, output_dims)
        super().__init__(latent_idgp, lmc_layer, normalization=normalization)

    def layer_normalize(self, f: Tensor) -> Tensor:  # override
        if self.normalization is None:
            return f
        if isinstance(
                self.normalization,
                (MaxMin_MovingAverage_LayerNormalizationMaxMin, Batch_MaxMin_Normalization)
        ):
            f_nor = self.normalization(f, training=self.training)
        elif isinstance(
                self.normalization,
                (Affine_SameAsInput_Normalization, BatchNorm1d_LayerNormalization, Tanh_Normalization)
        ):
            f_nor = self.normalization(f)
        else:
            raise NotImplementedError
        return f_nor
