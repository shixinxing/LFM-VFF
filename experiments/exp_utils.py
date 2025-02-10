import logging
import random

import torch
import numpy as np

from intdom_dgp.models.layer_normalization import (
    Affine_SameAsInput_Normalization, Batch_MaxMin_Normalization, MaxMin_MovingAverage_LayerNormalizationMaxMin,
    BatchNorm1d_LayerNormalization, Tanh_Normalization
)

from dlfm_vff.models.deep_lfm_vff_lmc import DeepLFM_VFF_LMC


def set_dtype_seed_device(args):
    # seed and device setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dtype setting
    if args.float == '64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    #     torch.mps.manual_seed(args.seed)  # LOP does not support MPS
    else:
        device = torch.device('cpu')

    print(f"Device: {device}, Seed: {args.seed}, Float: {torch.get_default_dtype()}")
    return device


def print_max_gradient(named_param_iter, print_func):
    max_grad = None
    max_grad_name = None

    for name, param in named_param_iter:
        if param.grad is not None:
            max_grad_param = torch.max(torch.abs(param.grad))
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
                max_grad_name = name

    print_func(f"Max Grad: {max_grad} (Param: {max_grad_name})")


def print_all_params(model, print_func=print, threshold=100, show_details=False):
    anomaly_flag = False
    for name, param in model.named_parameters():
        max_val = torch.max(torch.abs(param.data))
        has_anomaly = max_val > threshold
        anomaly_flag = has_anomaly if has_anomaly else anomaly_flag
        print_func(f"Param: {name} | Size: {param.size()} | Max abs value: {max_val:.2f} "
                   f"| Anomaly detected: {has_anomaly}")
    print_func(f"has anomalies?: {anomaly_flag}\n")
    if show_details:
        for name, param in model.named_parameters():
            print_func(f"Param: {name} | Value: {param}")
    print_func(" ")


@torch.no_grad()
def display_training(model, print_func=print, print_max_grad=False):

    def display_normalization_info(layer):
        if layer.normalization is not None:
            if isinstance(
                    layer.normalization,
                    (Affine_SameAsInput_Normalization, Batch_MaxMin_Normalization,
                     MaxMin_MovingAverage_LayerNormalizationMaxMin, BatchNorm1d_LayerNormalization,
                     Tanh_Normalization)
            ):
                print_func(layer.normalization.get_dict_info())
            else:
                raise NotImplementedError

    def display_kernel_noise(layer):
        if layer.has_kernel_noise:
            print_func(f"    kernel noise: "
                       f"{layer.raw_kernel_noise_constraint.transform(layer.raw_kernel_noise).cpu()}")

    def display_likelihood_noise():
        print_func(f"    task_noises: {model.likelihood.task_noises.cpu()}")
        if hasattr(model.likelihood, "noise"):  # global noise
            print_func(f"    global_noise: {model.likelihood.noise.cpu()}")

    if isinstance(model, DeepLFM_VFF_LMC):
        for i, layer in enumerate(model.layers):
            latent_idgp, lmc_layer = layer.latent_idgp, layer.lmc_layer
            kxx = latent_idgp.kxx
            print_func(f"---- Layer {i} ----")
            display_normalization_info(layer)
            print_func(f"    outputscale: {kxx.outputscale.cpu()}")
            length = kxx.base_kernel.lengthscale.cpu() if hasattr(kxx, 'base_kernel') else kxx.lengthscale.cpu()
            print_func(f"    l ({length.shape}): {length.squeeze(-1).squeeze(-1)}")
            if hasattr(kxx, 'alpha'):
                print_func(f"    alpha: {kxx.alpha}")
                print_func(f"    beta: {kxx.beta}")
                print_func(f"    gamma: {kxx.gama}")
                print_func(f"    lambda: {kxx.lamda}")
            display_kernel_noise(latent_idgp)
            print_func(f"    lmc: {lmc_layer.lmc_coefficients.cpu()}")
            if hasattr(latent_idgp, 'kzz') and hasattr(latent_idgp.kzz, 'a'):
                print_func(f"    a: {latent_idgp.kzz.a}, b: {latent_idgp.kzz.b}")
        display_likelihood_noise()

    print_max_gradient(model.named_parameters(), print_func) if print_max_grad else None
    print_func("  ")


def train(
        model, num_samples, full_cov,
        train_loader,
        optimizer, mll,
        num_epochs, display_epochs,
        device, write_log=False,
        print_max_grad=False,
        record_loss=False,
        **kwargs
):
    model.train()
    loss_history = []
    for i in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, full_cov=full_cov, S=num_samples, **kwargs)
            loss = - mll(output, y_batch)
            loss.backward()
            optimizer.step()

            if record_loss:
                loss_history.append(loss.detach().cpu().item())

        if (i + 1) % display_epochs == 0:
            print(f"##### Epoch {i + 1}/{num_epochs} - Loss {loss.detach().cpu()} ##### ")
            display_training(model, print_func=print, print_max_grad=print_max_grad)
            if write_log:
                logging.info(f"##### Epoch {i + 1}/{num_epochs} - Loss {loss.detach().cpu()} ##### ")
                display_training(model, print_func=logging.info, print_max_grad=print_max_grad)
    if record_loss:
        return loss_history


def train_record(
        model, num_samples, full_cov,
        train_loader,
        test_loader, train_info,  # need additional information
        optimizer, mll,
        num_epochs, display_epochs,
        device, write_log=False,
        print_max_grad=False, record=True,
        **kwargs
):
    model.train()
    loss_history, metric_rmse, metric_mll = [], [], []
    for i in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, full_cov=full_cov, S=num_samples, **kwargs)
            loss = - mll(output, y_batch)
            loss.backward()
            optimizer.step()

            if record:
                loss_history.append(loss.detach().cpu().item())
                model.eval()
                rmse, mean_log_likelihood = model.predict_measure(
                    test_loader, y_mean=None, y_std=None, S=train_info['num_samples_test']  # return standardize metrics
                )
                metric_rmse.append(rmse.detach().cpu().item())
                metric_mll.append(mean_log_likelihood.detach().cpu().item())
                model.train()

        if (i + 1) % display_epochs == 0:
            print(f"##### Epoch {i + 1}/{num_epochs} - Loss {loss.detach().cpu()} ##### ")
            display_training(model, print_func=print, print_max_grad=print_max_grad)
            if record:
                print(f"MLL: {metric_mll[-1]},   RMSE: {metric_rmse[-1]}\n")
            if write_log:
                logging.info(f"##### Epoch {i + 1}/{num_epochs} - Loss {loss.detach().cpu()} ##### ")
                display_training(model, print_func=logging.info, print_max_grad=print_max_grad)
                if record:
                    logging.info(f"MLL: {metric_mll[-1]},   RMSE: {metric_rmse[-1]}\n")
    if record:
        return loss_history, metric_rmse, metric_mll

