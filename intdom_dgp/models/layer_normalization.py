import torch
from torch import Tensor
from gpytorch.module import Module


# currently not suitable for learnable boundary
class Affine_SameAsInput_Normalization(Module):
    """
    Affine transformation to [to_min, to_max]
    Intermediate layers use the same normalization params as the inputs layer during both training and evaluation
    """
    def __init__(
            self,
            to_min, to_max,
            Xmin, Xmax,
    ):
        super().__init__()
        self.to_min, self.to_max = to_min, to_max
        self.Xmin, self.Xmax = Xmin, Xmax  # for input data points

    def forward(self, f: Tensor) -> Tensor:
        return (self.to_max - self.to_min) * (f - self.Xmin) / (self.Xmax - self.Xmin) + self.to_min

    def get_dict_info(self):
        info = {
            'type': 'same-as-input',
            'preprocess_a': self.to_min,
            'preprocess_b': self.to_max,
            'min_tensor': self.Xmin,
            'max_tensor': self.Xmax
        }
        return info


class Tanh_Normalization(Module):
    """use tanh: tanh(x / h) * h to squeeze [Xmin, Xmax] to [to_min, to_max]"""
    def __init__(
            self,
            to_min, to_max,
            Xmin, Xmax,
            h_trainable=True
    ):
        super().__init__()
        self.to_min, self.to_max = to_min, to_max
        self.Xmin, self.Xmax = Xmin, Xmax
        amplitude = torch.as_tensor(to_max - to_min) / 2.

        self.h_trainable = h_trainable
        if self.h_trainable:
            self.register_parameter('h', torch.nn.Parameter(amplitude))
        else:
            self.register_buffer('h', amplitude)

    def forward(self, f: Tensor) -> Tensor:
        x_center = (self.Xmax + self.Xmin) / 2.
        y_center = (self.to_max + self.to_min) / 2.
        return self.h * torch.tanh((f - x_center) / self.h) + y_center

    def get_dict_info(self):
        info = {
            'type': 'tanh',
            'preprocess_a': self.to_min,
            'preprocess_b': self.to_max,
            'min_tensor': self.Xmin,
            'max_tensor': self.Xmax,
            'h': self.h,
            'h_trainable': self.h_trainable
        }
        return info


class Batch_MaxMin_Normalization(Module):
    """
    Naively use the max and min of mini-batches at each intermediate layer to normalize each output dimensions;
    Won't keep running estimates of its computed min/max values.
    Use batch statistics instead when evaluation
    """
    def __init__(
            self, to_min, to_max,
            track_gradients=False,
            store_minibatch_min_max=False,  # whether to store min and max of all mini-batches
            eps=1e-8,
    ):
        super().__init__()
        self.to_min, self.to_max = to_min, to_max
        self.track_gradients = track_gradients
        self.store_minibatch_min_max = store_minibatch_min_max
        if self.store_minibatch_min_max:
            self.minibatch_min_max = []  # [(min, max), (min, max),...]
        self.eps = eps
        self.min_tensor, self.max_tensor = None, None

    def forward(self, x: Tensor, training=True) -> Tensor:  # x: [S, B, t]
        if training:
            with torch.set_grad_enabled(self.track_gradients):
                x_processed = x.detach() if not self.track_gradients else x
                min_tensor = torch.min(x_processed, dim=-2, keepdim=True)[0]
                max_tensor = torch.max(x_processed, dim=-2, keepdim=True)[0]
                min_tensor = torch.min(min_tensor, dim=-3, keepdim=True)[0]  # [1, 1, t]
                max_tensor = torch.max(max_tensor, dim=-3, keepdim=True)[0]

                if self.store_minibatch_min_max:
                    self.minibatch_min_max.append((min_tensor, max_tensor))

            self.min_tensor = min_tensor
            self.max_tensor = max_tensor  # store for evaluation

            slope = (self.to_max - self.to_min) / (max_tensor - min_tensor + self.eps)
            return slope * (x - min_tensor) + self.to_min
        else:
            with torch.no_grad():
                slope = (self.to_max - self.to_min) / (self.max_tensor - self.min_tensor + self.eps)
                return slope * (x - self.min_tensor) + self.to_min

    def return_stored_min_max(self):
        return self.minibatch_min_max if self.store_minibatch_min_max else None

    def get_dict_info(self):
        info = {
            'type': 'minibatch-maxmin',
            'preprocess_a': self.to_min,
            'preprocess_b': self.to_max,
            'min_tensor': self.min_tensor.cpu(),
            'max_tensor': self.max_tensor.cpu(),
            'track_gradients': self.track_gradients
        }
        return info


# noinspection PyAttributeOutsideInit
# imitate torch.nn.BatchNorm1d, using moving average to compute max/min estimates,
# use these estimates during evaluation
class MaxMin_MovingAverage_LayerNormalizationMaxMin(Module):
    def __init__(
            self, to_min, to_max, num_features,
            track_gradients=True,
            momentum=0.8,
            store_minibatch_min_max=False,
            eps=1e-8
    ):
        super().__init__()
        self.to_min, self.to_max = to_min, to_max
        self.num_features = num_features
        self.track_gradients = track_gradients
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_min_tensor', torch.ones(self.num_features) * self.to_min)
        self.register_buffer('running_max_tensor', torch.ones(self.num_features) * self.to_max)

        self.store_minibatch_min_max = store_minibatch_min_max
        if self.store_minibatch_min_max:
            self.minibatch_min_max = []  # [(min, max), (min, max),...]

    def forward(self, x, training=True):  # x: [S, B, t]
        if training:
            with torch.set_grad_enabled(self.track_gradients):
                x_processed = x.detach() if not self.track_gradients else x
                min_tensor = torch.min(x_processed, -2)[0]
                max_tensor = torch.max(x_processed, -2)[0]
                if self.store_minibatch_min_max:
                    self.minibatch_min_max.append((min_tensor, max_tensor))

            ndim = max_tensor.ndim
            # [S, t] -> [t]
            current_max_tensor = torch.mean(max_tensor.detach(), dim=[i for i in range(ndim - 1)])  # ⚠️
            current_min_tensor = torch.mean(min_tensor.detach(), dim=[i for i in range(ndim - 1)])
            self.running_max_tensor = (current_max_tensor * (1. - self.momentum)
                                       + self.running_max_tensor * self.momentum)
            self.running_min_tensor = (current_min_tensor * (1. - self.momentum)
                                       + self.running_min_tensor * self.momentum)

            slope = (self.to_max - self.to_min) / (max_tensor - min_tensor + self.eps)
            return slope * (x - min_tensor) + self.to_min
        else:
            with torch.no_grad():
                slope = (self.to_max - self.to_min) / (self.running_max_tensor - self.running_min_tensor + self.eps)
                return slope * (x - self.running_min_tensor) + self.to_min

    def return_stored_min_max(self):
        return self.minibatch_min_max if self.store_minibatch_min_max else None

    def get_dict_info(self):  # give the information of the normalization layers
        info = {
            'type': 'moving-average',
            'preprocess_a': self.to_min,
            'preprocess_b': self.to_max,
            'min_tensor': self.running_min_tensor.cpu(),
            'max_tensor': self.running_max_tensor.cpu(),
            'num_features': self.num_features,
            'track_gradients': self.track_gradients,
            'momentum': self.momentum,
        }
        return info


class BatchNorm1d_LayerNormalization(Module):
    def __init__(self, num_features):
        super(BatchNorm1d_LayerNormalization, self).__init__()
        self.num_features = num_features
        self.bn_layer = torch.nn.BatchNorm1d(num_features)

    def forward(self, x, training=True):  # [S, B, t]
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.num_features)
        x = self.bn_layer(x)
        return x.view(*batch_shape, self.num_features)

    def get_dict_info(self):
        info = {
            'type': 'batch-norm',
            'running_mean': self.bn_layer.running_mean,
            'running_var': self.bn_layer.running_var,
            'num_features': self.num_features,
            'momentum': self.bn_layer.momentum
        }
        return info

