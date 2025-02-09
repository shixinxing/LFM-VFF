import torch
import math


def uniform_sampling(a, b, n):
    return a + (b - a) * torch.rand(n, dtype=torch.float64).to(torch.get_default_dtype())


def multi_step_shift(num_data_frag=140, noise=0.005):
    x_frag_1 = uniform_sampling(0., 0.2, num_data_frag)
    y_frag_1 = torch.ones_like(x_frag_1) * (- 1.) + torch.randn_like(x_frag_1) * noise
    x_frag_2 = uniform_sampling(0.2, 0.4, num_data_frag)
    y_frag_2 = torch.ones_like(x_frag_2) * 1.2 + torch.randn_like(x_frag_2) * noise
    x_frag_3 = uniform_sampling(0.4, 0.6, num_data_frag)
    y_frag_3 = torch.ones_like(x_frag_3) * (-0.25) + torch.randn_like(x_frag_3) * noise
    x_frag_4 = uniform_sampling(0.6, 0.8, num_data_frag)
    y_frag_4 = torch.ones_like(x_frag_4) * 1.2 + torch.randn_like(x_frag_4) * noise
    x_frag_5 = uniform_sampling(0.8, 1, num_data_frag)
    y_frag_5 = torch.ones_like(x_frag_5) * (0.25) + torch.randn_like(x_frag_5) * noise

    x = torch.cat([x_frag_1, x_frag_2, x_frag_3, x_frag_4, x_frag_5])
    y = torch.cat([y_frag_1, y_frag_2, y_frag_3, y_frag_4, y_frag_5])

    index = torch.randperm(x.size(0))
    x_train = x[index[: x.size(0) // 2]].unsqueeze(-1)
    y_train = y[index[: x.size(0) // 2]].unsqueeze(-1)
    x_test = x[index[x.size(0) // 2:]].unsqueeze(-1)
    y_test = y[index[x.size(0) // 2:]].unsqueeze(-1)

    return x_train, y_train, x_test, y_test


def rff_compare_vff(noise_std=0.2):
    def sin_cos_original(t):
        return torch.sin(3 * t) + 0.4 * torch.cos(9 * t)

    def sinc(t):
        return torch.sinc(9 * (t + 0.15) / math.pi) * 0.8 + 0.1

    def sin_cos(t):
        return torch.sin(2 * t) + 0.4 * torch.cos(10 * t) - 0.1

    num_points = 20
    x = torch.cat([uniform_sampling(0, 0.4, num_points // 2), uniform_sampling(0., 1, num_points // 2)], dim=0)
    y = sin_cos(x) + torch.randn_like(x) * noise_std

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)

    return x, y, None, None


if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)
    matplotlib.use('MacOSX')
    x, y, xt, yt = multi_step_shift()

    plt.plot(x.squeeze(-1), y.squeeze(-1), 'r.', markersize=3)
    if xt is not None and yt is not None:
        plt.plot(xt.squeeze(-1), yt.squeeze(-1), 'g.', markersize=3)
    plt.show()

