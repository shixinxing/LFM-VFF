from matplotlib import pyplot as plt
from typing import Optional


def plot_regression_1d(
        x_train, y_train, x_test, y_test,
        x, mixture_mean, mixture_var,
        title, save_path_name, plot_y_lim: Optional[tuple] = None
):
    lowers_1, uppers_1 = mixture_mean - mixture_var.sqrt(), mixture_mean + mixture_var.sqrt()
    lowers_2, uppers_2 = mixture_mean - 2 * mixture_var.sqrt(), mixture_mean + 2 * mixture_var.sqrt()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    x_train_np, y_train_np = x_train.squeeze(-1).numpy(), y_train.squeeze(-1).numpy()
    x_test_np, y_test_np = x_test.squeeze(-1).numpy(), y_test.squeeze(-1).numpy()

    ax.plot(x_train_np, y_train_np, 'r.', markersize=3, label='Training Points')
    ax.plot(x_test_np, y_test_np, 'g.', markersize=3, label='Test Points')
    ax.plot(x.squeeze(-1).numpy(), mixture_mean.numpy(), 'b-', linewidth=1.5, label='Mean Prediction')
    ax.fill_between(x.squeeze(-1).numpy(), lowers_1.numpy(), uppers_1.numpy(),
                    alpha=0.20, color='blue')
    ax.fill_between(x.squeeze(-1).numpy(), lowers_2.numpy(), uppers_2.numpy(),
                    alpha=0.10, color='blue', label=r'2$\sigma$')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_ylim(-2, 2) if plot_y_lim is None else ax.set_ylim(*plot_y_lim)
    ax.legend()

    ax.grid(True)
    ax.set_title(title)
    plt.savefig(save_path_name, bbox_inches='tight')


def visualize_all_layers(
        x_train, y_train, x_test, y_test,
        x, Fs, Fmeans, Fvars,
        title, save_path_name,
        num_print_samples=0, Fs_nor=None
):
    """
    Visualizes the predictions for all layers.
    - Fs: Samples from each layer. Expected shape for each element in list: [S, N]
    - Fmeans: Mean predictions for each layer. Expected shape for each element in list: [N]
    - Fvars: Variance predictions for each layer. Expected shape for each element in list: [N]
    """
    x = x.squeeze(-1)
    num_layers = len(Fmeans)

    fig, axs = plt.subplots(num_layers, 1, figsize=(8, num_layers * 5), sharex=True)
    if num_layers == 1:
        axs = [axs, ]  # Ensure axs is iterable
    for i, (mean, var) in enumerate(zip(Fmeans, Fvars)):
        axs[i].plot(x.numpy(), mean.numpy(), 'k-', linewidth=1.5, label=f'Layer {i + 1} Mean')
        axs[i].fill_between(x.numpy(), mean.numpy() - var.sqrt().numpy(), mean + var.sqrt().numpy(),
                            alpha=0.2, color='blue', label=f'Layer {i + 1} Confidence')
        axs[i].fill_between(x.numpy(), mean.numpy() - 2 * var.sqrt().numpy(), mean + 2 * var.sqrt().numpy(),
                            alpha=0.1, color='blue')

        if num_print_samples:
            for s in Fs[i][:num_print_samples]:
                axs[i].plot(x.numpy(), s.numpy(), 'b-', alpha=0.5, linewidth=0.6)
            if Fs_nor is not None and i + 1 < num_layers:
                for s_nor in Fs_nor[i + 1][:num_print_samples]:  # not include the input at the first layer
                    axs[i].plot(x.numpy(), s_nor.numpy(), 'r--', alpha=0.5, linewidth=0.6)

        axs[i].set_ylabel(f'Layer {i+1} Output')
        axs[i].grid(True)

    x_train_np, y_train_np = x_train.squeeze(-1).numpy(), y_train.squeeze(-1).numpy()
    x_test_np, y_test_np = x_test.squeeze(-1).numpy(), y_test.squeeze(-1).numpy()
    axs[-1].plot(x_train_np, y_train_np, 'r.', markersize=3, label='Training Points')
    axs[-1].plot(x_test_np, y_test_np, 'g.', markersize=3, label='Test Points')
    axs[-1].set_xlabel('Input')

    for ax in axs:
        ax.legend()
    fig.suptitle(title, fontsize=12)  # Set the overall title for the figure
    plt.tight_layout(rect=(0, 0, 1, 1))  # Adjust the layout to make room for the overall title
    plt.savefig(save_path_name)

