import argparse
import datetime
import json
import os
import logging
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

import matplotlib
from matplotlib import pyplot as plt

from data_generation import multi_step_shift
from set_models_synthetic_1d import set_models_dlfm_vff
from set_models_regression import set_params

from exp_utils import set_dtype_seed_device, train, print_all_params
from plot import plot_regression_1d, visualize_all_layers


warning_triggered = False


def run_synthetic(args):
    print(f"model: {args.model_type}")
    device = set_dtype_seed_device(args)

    time_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    save_dir = os.path.join(args.save_dir, args.dataset_name, args.model_type, time_str)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_path = os.path.join(str(save_dir), args.model_type + '-' + time_str + '.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')

    # load datapoints
    if args.dataset_name == 'multi-step-shift':
        x_train, y_train, x_test, y_test = multi_step_shift()
    else:
        raise NotImplementedError
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=x_train.size(0), shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=x_test.size(0), shuffle=False)
    data_info = {
        'name': args.dataset_name,
        'num_training_points': x_train.size(0),
        'seed': args.seed,
        'dtype': str(torch.get_default_dtype())

    }

    # set models
    model_info = {
        'model_type': args.model_type,
        'kernel_type': args.kernel_type,
        'whitened': args.whitened,
        'M': args.M,

        'mean_z_type': 'zero' if args.whitened != 'none' else 'with-x',
        'a': args.a,
        'b': args.b,

        'num_rff_samples': 50,
        'rff_lower_bound': None,

        'lengthscale': 0.1,
        'outputscale': 0.1,
        'alpha': 3,
        'beta': 0.1,
        'likelihood_variance': 0.01,

        'normalization': {
            'type': 'same-as-input',
            'preprocess_a': 0.,
            'preprocess_b': 1.,
            'min_tensor': 0.,
            'max_tensor': 1.
        }
    }

    if args.model_type[:8] == 'dlfm-vff':
        model = set_models_dlfm_vff(model_info)
    else:
        raise NotImplementedError

    set_params(
        model, model_info['lengthscale'], model_info['outputscale'], model_info['likelihood_variance'],
        alpha=model_info['alpha'], beta=model_info['beta']
    )

    for i in range(model.num_layers - 1):  # not for the outer layer
        vs = model.variational_strategy.sub_variational_strategies[i]
        vs._variational_distribution.chol_variational_covar.detach().mul_(1e-5)

    model.to(device=device, dtype=torch.get_default_dtype())
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=data_info['num_training_points']))
    if args.model_type[:4] == 'dlfm':
        optimizer = torch.optim.Adam(
           model.parameters(), lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_info = {
        'device': str(device),
        'dtype': str(torch.get_default_dtype()),
        'seed': args.seed,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'minibatch_size': data_info['num_training_points'],  # all points at once

        'num_samples_training': 10,
        'num_samples_test': 100,
        'full_cov': True if args.full_cov else False
    }

    # training
    print(f"Starting training {args.model_type}...\n")
    train(
        model,
        train_info['num_samples_training'], train_info['full_cov'],
        train_loader, optimizer, mll,
        train_info['num_epochs'], args.display_epochs,
        device, write_log=True, print_max_grad=False
    )
    print(f"Finished training {args.model_type}.\n")
    print("===== check all params ===== ")
    print_all_params(model, print_func=print)

    # posterior prediction
    model.eval()
    num_plot_points = 400
    x = torch.linspace(-0.2, 1.2, num_plot_points).unsqueeze(-1)
    plot_loader = DataLoader(x, batch_size=num_plot_points, shuffle=False)

    # plot the regression results
    if device == torch.device('cpu'):
        matplotlib.use('MacOSX')
    full_cov_plot = True
    mixture_mean, mixture_variance = model.predict_mean_and_var_loader(
        plot_loader, full_cov=full_cov_plot, S=train_info['num_samples_test']
    )
    fig_title = (args.model_type.upper() + '-' + args.whitened
                 + ('-full_cov' if full_cov_plot else '-diag_cov'))
    plot_regression_1d(
        x_train, y_train, x_test, y_test,
        x, mixture_mean, mixture_variance,
        fig_title, save_path_name=os.path.join(str(save_dir), args.model_type + '-' + 'y_x-' + time_str)
    )

    # plot all layers and give metrics
    if args.model_type[:8] == 'dlfm-vff':
        Finputs, Fmeans, Fvars, Foutputs = model.predict_all_layers(
            x, full_cov=full_cov_plot, S=train_info['num_samples_test']
        )
        visualize_all_layers(
            x_train, y_train, x_test, y_test,
            x, Foutputs, Fmeans, Fvars, Fs_nor=Finputs,
            title=fig_title,
            save_path_name=os.path.join(str(save_dir), args.model_type + '-f_f_x-' + time_str),
            num_print_samples=8
        )
        sqrt_err, log_ll = model.predict_measure(test_loader, S=train_info['num_samples_test'])
        print(f"Test RMSE: {sqrt_err}, Mean log likelihood: {log_ll}\n")
    else:
        raise NotImplementedError

    summary_args = {
        'data_info': data_info,
        'model_info': model_info,
        'train_info': train_info,
        'rmse': sqrt_err.cpu().item(),
        'mean_log_likelihood': log_ll.cpu().item()
    }
    with open(os.path.join(str(save_dir), 'summary.json'), 'w') as f:
        json.dump(summary_args, f, indent=4)

    plt.show()

    # save ndarray for plotting
    np.save(os.path.join(str(save_dir), f'x_train_{args.model_type}.npy'), x_train.squeeze(-1).numpy())
    np.save(os.path.join(str(save_dir), f'y_train_{args.model_type}.npy'), y_train.squeeze(-1).numpy())
    np.save(os.path.join(str(save_dir), f'x_test_{args.model_type}.npy'), x_test.squeeze(-1).numpy())
    np.save(os.path.join(str(save_dir), f'y_test_{args.model_type}.npy'), y_test.squeeze(-1).numpy())
    np.save(os.path.join(str(save_dir), f'x_{args.model_type}.npy'), x.squeeze(-1).numpy())
    np.save(os.path.join(str(save_dir), f'mix_mean_{args.model_type}.npy'), mixture_mean.numpy())
    np.save(os.path.join(str(save_dir), f'mix_var_{args.model_type}.npy'), mixture_variance.numpy())
    np.save(os.path.join(str(save_dir), f'Fs_{args.model_type}.npy'), Foutputs[-1][:3, :].numpy())  # [S, N]

    # save the intermediate results for plotting
    torch.save(Fmeans, os.path.join(str(save_dir), f"Fmeans_{args.model_type}.pth"))
    torch.save(Fvars, os.path.join(str(save_dir), f'Fvars_{args.model_type}.pth'))
    torch.save(Foutputs, os.path.join(str(save_dir), f"Fs_{args.model_type}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="=" * 5 + "Synthetic 1-d Data" + "=" * 5)

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--float', type=str, default="64", help='float type: 32 or 64')

    parser.add_argument('--dataset_name', type=str, default="multi-step-shift")

    parser.add_argument('--model_type', type=str, default="dlfm-vff-2", help='model type: dlfm-vff-1, dlfm-vff-2')
    parser.add_argument('--kernel_type', type=str, default='matern32')
    parser.add_argument('--M', type=int, default=20, help='number of inducing frequencies w/o offset')
    parser.add_argument('--whitened', type=str, default='cholesky', help='none, cholesky')
    parser.add_argument('--full_cov', type=int, default=0)

    parser.add_argument("--a", type=float, default=-0.5, help="integral lower bound")
    parser.add_argument("--b", type=float, default=1.5, help="integral upper bound")

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--display_epochs', type=int, default=1000, help='number of displaying epochs')

    parser.add_argument('--save_dir', type=str, default="tmp", help='save directory')

    args = parser.parse_args()

    run_synthetic(args)
