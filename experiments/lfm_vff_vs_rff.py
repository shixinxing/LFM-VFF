import os
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO, ExactMarginalLogLikelihood

from dlfm_vff.models.exact_lfm_rff import ExactLFM1d_RFF
from dlfm_vff.models.exact_lfm import ExactLFM1d
from dlfm_vff.models.model_factory import DeepLFM_VFF_1Layer_1d
from experiments.data_generation import rff_compare_vff

subdir = 'vff_vs_rff'
time_str = datetime.datetime.now().strftime("%d_%H-%M-%S")
if not os.path.exists(subdir):
    os.makedirs(subdir)

matplotlib.use('TkAgg')
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# load data
x_train, y_train, _, _ = rff_compare_vff()
assert x_train.ndim > 1, "The input is one-dimensional, reshape it."
dataset_train = TensorDataset(x_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=x_train.size(0))

# model setting
nu, kernel_type = 0.5, 'matern12'

model_exact = ExactLFM1d(x_train, y_train.squeeze(-1), GaussianLikelihood(), kernel_type=kernel_type)
# model_exact.lfm_kernel.lengthscale = 0.01
# model_exact.likelihood.noise = 0.2

a, b = - 1, 2
M_1 = 20
model_vff_1 = DeepLFM_VFF_1Layer_1d(a, b, M_1, nu, whitened='cholesky', fix_lmc=True)
M_2 = 80
model_vff_2 = DeepLFM_VFF_1Layer_1d(a, b, M_2, nu, whitened='cholesky', fix_lmc=True)

num_rff_1 = M_1
model_rff_1 = ExactLFM1d_RFF(x_train, y_train.squeeze(-1), GaussianLikelihood(), num_rff=num_rff_1,
                             kernel_type=kernel_type)
num_rff_2 = M_2
model_rff_2 = ExactLFM1d_RFF(x_train, y_train.squeeze(-1), GaussianLikelihood(), num_rff=num_rff_2,
                             kernel_type=kernel_type)
num_rff_3 = 500
model_rff_3 = ExactLFM1d_RFF(x_train, y_train.squeeze(-1), GaussianLikelihood(), num_rff=num_rff_3,
                             kernel_type=kernel_type)


# train exact model
mll = ExactMarginalLogLikelihood(model_exact.likelihood, model_exact)
optimizer = torch.optim.Adam(model_exact.parameters(), lr=0.01)
model_exact.train()
for i in range(1500):
    optimizer.zero_grad()
    output = model_exact(x_train)
    loss = - mll(output, y_train.squeeze(-1))
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        print(f'Epoch - {i} Loss: {loss.item():.4f}, ')
        print(f'    sig2: {model_exact.lfm_kernel.outputscale}, l: {model_exact.lfm_kernel.lengthscale} ')
        print(f'    alpha: {model_exact.lfm_kernel.alpha}, beta: {model_exact.lfm_kernel.beta}, '
              f'gamma: {model_exact.lfm_kernel.gama}')
        print(f"    noise: {model_exact.likelihood.noise}\n")

# fix parameters
l, sig2 = model_exact.lfm_kernel.lengthscale, model_exact.lfm_kernel.outputscale
alpha, beta, noise = model_exact.lfm_kernel.alpha, model_exact.lfm_kernel.beta, model_exact.likelihood.noise

for model_vff in [model_vff_1, model_vff_2]:
    lfm_kernel = model_vff.layers[0].latent_idgp.kxx
    lfm_kernel.outputscale = sig2
    lfm_kernel.lengthscale = l
    lfm_kernel.alpha = alpha
    lfm_kernel.beta = beta
    model_vff.likelihood.task_noises = noise
    for param in lfm_kernel.parameters():
        param.requires_grad = False
    for param in model_vff.likelihood.parameters():
        param.requires_grad = False


for model_rff in [model_rff_1, model_rff_2, model_rff_3]:
    lfm_rff_kernel = model_rff.lfm_rff_kernel
    matern_rff_kernel = model_rff.scaled_matern_rff_kernel
    matern_rff_kernel.outputscale = sig2
    matern_rff_kernel.base_kernel.lengthscale = l
    lfm_rff_kernel.alpha = alpha
    lfm_rff_kernel.beta = beta
    model_rff.likelihood.noise = noise
    for param in lfm_rff_kernel.parameters():
        param.requires_grad = False
    for param in model_rff.likelihood.parameters():
        param.requires_grad = False

# train vff model
for model_vff in [model_vff_1, model_vff_2]:
    mll = DeepApproximateMLL(VariationalELBO(model_vff.likelihood, model_vff, num_data=x_train.size(0)))
    optimizer = torch.optim.Adam(model_vff.parameters(), lr=0.01)

    model_vff.train()
    for i in range(5000):
        optimizer.zero_grad()
        output = model_vff(x_train, full_cov=True, S=100)
        loss = - mll(output, y_train)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch - {i}: loss = {loss.item():.4f}')
    print(" ")


# Function to plot data
def plot(x_train, y_train, x, mean, var, ax, color='blue'):
    lowers_1, uppers_1 = mean - var.sqrt(), mean + var.sqrt()
    lowers_2, uppers_2 = mean - 2 * var.sqrt(), mean + 2 * var.sqrt()
    x_train_np, y_train_np, x_np = x_train.squeeze(-1).numpy(), y_train.squeeze(-1).numpy(), x.squeeze(-1).numpy()

    ax.plot(x_train_np, y_train_np, 'r.', markersize=1, label='Training Points')
    ax.plot(x_np, mean.numpy(), '-', color=color, linewidth=0.5, label='Mean Prediction')
    ax.fill_between(x_np, lowers_1.numpy(), uppers_1.numpy(), color=color, alpha=0.2, label='1 Standard Deviation')
    ax.fill_between(x_np, lowers_2.numpy(), uppers_2.numpy(), color=color, alpha=0.1, label='2 Standard Deviations')

    ax.tick_params(axis='both', which='major', labelsize=2)


# Enable LaTeX rendering and set font to match LaTeX documents
rc('text', usetex=True)
rc('font', family='serif', size=12)
# Define a better color palette, e.g., using Tableau's color-blind friendly palette
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#E69F00']
labels = ['Blue', 'Orange', 'Green', 'Yellow', 'Pink', 'Light Blue', 'Gold']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharey=True, dpi=300)
x = torch.linspace(-0.1, 1.1, 300).unsqueeze(-1)

model_exact.eval()
with torch.no_grad():
    pred_dist = model_exact.likelihood(model_exact(x))
    mean, var = pred_dist.mean, pred_dist.variance
    plot(x_train, y_train, x, mean, var, ax=axs[0, 0], color=colors[0])

np.save('x.npy', x.squeeze(-1).numpy())
np.save('x_train.npy', x_train.squeeze(-1).numpy())
np.save('y_train.npy', y_train.squeeze(-1).numpy())
np.save('mean.npy', mean.numpy())
np.save('var.npy', var.numpy())


for i, model_vff in enumerate([model_vff_1, model_vff_2]):
    model_vff.eval()
    mix_mean, mix_var = model_vff.predict_mean_and_var(x, full_cov=True, S=200)
    mix_mean, mix_var = mix_mean.squeeze(-1), mix_var.squeeze(-1).squeeze(-1)
    plot(x_train, y_train, x, mix_mean, mix_var, ax=axs[0, i+1], color=colors[0])

    np.save(f'vff_mean_{i+1}.npy', mix_mean.numpy())
    np.save(f'vff_var_{i+1}.npy', mix_var.numpy())

# test rff model
for i, model_rff in enumerate([model_rff_1, model_rff_2, model_rff_3]):
    model_rff.eval()
    model_rff.likelihood.eval()
    prediction_dist = model_rff.likelihood(model_rff(x))
    mean, var = prediction_dist.mean, prediction_dist.variance
    plot(x_train, y_train, x, mean, var, ax=axs[1, i], color=colors[0])

    np.save(f'rff_mean_{i+1}.npy', mean.numpy())
    np.save(f'rff_var_{i+1}.npy', var.numpy())

axs[1, 0].set_xlabel('$x$', fontsize=5)
axs[0, 0].set_ylabel('$y$', fontsize=5)


ax = axs[1, 2]
# Shrink current axis by 20% to fit the legend outside
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=2)  # smaller legend
plt.tight_layout()
plt.savefig(os.path.join(subdir, time_str + '.pdf'), format='pdf', bbox_inches='tight')
plt.show()




