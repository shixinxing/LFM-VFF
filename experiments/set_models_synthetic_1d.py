from dlfm_vff.models.model_factory import DeepLFM_VFF_2layer_1d, DeepLFM_VFF_1Layer_1d


def get_nu(kernel_type: str) -> float:
    if kernel_type[:6] == 'matern':
        nu = 0.5 if kernel_type == 'matern12' else 1.5 if kernel_type == 'matern32' else 2.5
    else:
        raise NotImplementedError
    return nu


def get_info_from_model_info(model_info: dict):
    model_type = model_info['model_type']
    kernel_type = model_info['kernel_type']
    whitened = model_info['whitened']
    M = model_info['M']
    return model_type, kernel_type, whitened, M


def set_models_dlfm_vff(model_info):
    model_type, kernel_type, whitened, M = get_info_from_model_info(model_info)
    assert model_type[:8] == 'dlfm-vff', 'Only DLFM with VFFs are supported.'
    nu = get_nu(kernel_type)
    print(f"nu = {nu}")

    if model_type == 'dlfm-vff-1':
        model = DeepLFM_VFF_1Layer_1d(
            model_info['a'], model_info['b'], M, nu, whitened=whitened, fix_lmc=True
        )
    elif model_type == 'dlfm-vff-2':
        model = DeepLFM_VFF_2layer_1d(
            model_info['a'], model_info['b'], M, nu, whitened=whitened,
            mean_z_type=model_info['mean_z_type'], fix_lmc=True, has_kernel_noise=True,
            hidden_normalization_info=model_info['normalization']
        )
    else:
        raise NotImplementedError
    return model





