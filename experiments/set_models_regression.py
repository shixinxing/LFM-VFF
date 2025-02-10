from dlfm_vff.models.deep_lfm_vff_full import DeepLFM_VFF_Full
from dlfm_vff.models.deep_lfm_vff_lmc import DeepLFM_VFF_LMC
from dlfm_vff.models.deep_lfm_rff import DeepLFM_RFFLMC

from set_models_synthetic_1d import get_nu, get_info_from_model_info


def set_params(
        model,
        lengthscale=None, sig2=None, likelihood_noise=None,
        alpha=None, beta=None
):
    num_layers = model.num_layers

    # Helper function to ensure the parameter is iterable and matches the number of layers
    def ensure_iterable(param):
        if param is not None:
            if hasattr(param, '__iter__') and not isinstance(param, str):
                if len(param) != num_layers:
                    raise ValueError(f"Length of parameter must match the number of model layers ({num_layers}).")
                return param
            else:
                return [param] * num_layers
        else:
            return [None] * num_layers

    lengthscales = ensure_iterable(lengthscale)
    sig2s = ensure_iterable(sig2)
    alphas = ensure_iterable(alpha)
    betas = ensure_iterable(beta)

    for i in range(num_layers):
        if isinstance(model, (DeepLFM_VFF_LMC, DeepLFM_RFFLMC)):
            latent_idgp = model.layers[i].latent_idgp
            layer_kernel = latent_idgp.kxx if hasattr(latent_idgp, 'kxx') else latent_idgp.scaled_kernel
        else:
            raise NotImplementedError

        # Set lengthscale if provided
        if lengthscales[i] is not None:
            if hasattr(layer_kernel, 'base_kernel'):
                layer_kernel.base_kernel.lengthscale = lengthscales[i]
            else:
                layer_kernel.lengthscale = lengthscales[i]

        # Set sig2 if provided
        if sig2s[i] is not None:
            layer_kernel.outputscale = sig2s[i]

        # Set alpha and beta if provided (specific to certain model types)
        if isinstance(model, (DeepLFM_VFF_Full, DeepLFM_VFF_LMC, DeepLFM_RFFLMC)) and alphas[i] is not None and betas[i] is not None:
            layer_kernel.alpha = alphas[i]
            layer_kernel.beta = betas[i]

    if likelihood_noise is not None:
        model.likelihood.task_noises = likelihood_noise
        if hasattr(model.likelihood, "noise"):  # global noise
            model.likelihood.noise = likelihood_noise
        # model.likelihood.raw_task_noises.requires_grad = False


def get_info_from_model_info_regression(model_info: dict):
    model_type, kernel_type, whitened, M = get_info_from_model_info(model_info)
    input_dims, output_dims, hidden_dims, num_layers = (model_info['input_dims'], model_info['output_dims'],
                                                        model_info['hidden_dims'], model_info['num_layers'])
    return (model_type, kernel_type, whitened, M,
            input_dims, output_dims, hidden_dims, num_layers)


def set_models_dlfm_full_or_blockdiag_regression(model_info: dict, X_running=None):
    (model_type, kernel_type, whitened, M,
     input_dims, output_dims, hidden_dims, num_layers) = get_info_from_model_info_regression(model_info)
    nu = get_nu(kernel_type)
    print(f"model: {model_type}, kernel: {kernel_type}, M: {M}")

    if model_type == 'dlfm-full':
        model = DeepLFM_VFF_Full(
            input_dims, output_dims, hidden_dims, num_layers,
            model_info['a'], model_info['b'], M, nu,
            X_running=X_running, whitened=whitened, mean_z_type='zero', ode_type='ode1',
            hidden_normalization_info=model_info['normalization'], kernel_sharing_across_output_dims=True,
            has_kernel_noise=True
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}.")


def set_models_dlfm_lmc_regression(model_info):
    (model_type, kernel_type, whitened, M,
     input_dims, output_dims, hidden_dims, num_layers) = get_info_from_model_info_regression(model_info)
    nu = get_nu(kernel_type)
    assert model_type == 'dlfm-lmc', 'Only DLFMs with LMC are supported.'
    print(f"model: {model_type}, kernel: {kernel_type}, M: {M}")

    model = DeepLFM_VFF_LMC(
        input_dims, output_dims, hidden_dims, num_layers,
        model_info['a'], model_info['b'], M, nu, whitened=whitened, mean_z_type=model_info['mean_z_type'],
        hidden_normalization_info=model_info['normalization'], kernel_sharing_across_dims=False, has_kernel_noise=True,
        boundary_learnable=False
    )
    return model

