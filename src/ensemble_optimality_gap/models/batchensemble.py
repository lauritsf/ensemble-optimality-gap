import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


def random_sign_(tensor: torch.Tensor, prob: float = 0.5, value: float = 1.0):
    """
    Randomly set elements of the input tensor to either +value or -value.

    Args:
        tensor (torch.Tensor): Input tensor.
        prob (float, optional): Probability of setting an element to +value (default: 0.5).
        value (float, optional): Value to set the elements to (default: 1.0).

    Returns:
        torch.Tensor: Tensor with elements set to +value or -value.
    """
    sign = torch.where(torch.rand_like(tensor) < prob, 1.0, -1.0)

    with torch.no_grad():
        tensor.copy_(sign * value)


class BatchEnsembleMixin:
    def init_ensemble(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        alpha_init: float | None = None,
        gamma_init: float | None = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        self.ensemble_size = ensemble_size
        self.alpha_init = alpha_init
        self.gamma_init = gamma_init

        if not isinstance(self, nn.Module):
            raise TypeError("BatchEnsembleMixin must be mixed with nn.Module or one of its subclasses")

        if alpha_init is None:
            self.register_parameter("alpha_param", None)
        else:
            self.alpha_param = self.init_scaling_parameter(alpha_init, in_features, device=device, dtype=dtype)
            self.register_parameter("alpha_param", self.alpha_param)

        if gamma_init is None:
            self.register_parameter("gamma_param", None)
        else:
            self.gamma_param = self.init_scaling_parameter(gamma_init, out_features, device=device, dtype=dtype)
            self.register_parameter("gamma_param", self.gamma_param)

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(ensemble_size, out_features, device=device, dtype=dtype))
            self.register_parameter("bias_param", self.bias_param)
        else:
            self.register_parameter("bias_param", None)

    def init_scaling_parameter(self, init_value: float, num_features: int, device=None, dtype=None):
        param = torch.empty(self.ensemble_size, num_features, device=device, dtype=dtype)
        if init_value < 0:
            param.normal_(mean=1, std=-init_value)
        else:
            random_sign_(param, prob=init_value, value=1.0)
        return nn.Parameter(param)

    def expand_param(self, x: torch.Tensor, param: torch.Tensor):
        """Expand and match a parameter to a given input tensor.

        Description:
        In BatchEnsemble, the alpha, gamma and bias parameters are expanded to match the input tensor.

        Args:
            x: Input tensor to match the parameter to. Shape: [batch_size, features/classes, ...]
            param: Parameter to expand. Shape: [ensemble_size, features/classes]

        Returns:
            expanded_param: Expanded parameter. Shape: [batch_size, features/classes, ...]
        """
        num_repeats = x.size(0) // self.ensemble_size
        expanded_param = torch.repeat_interleave(param, num_repeats, dim=0)
        extra_dims = len(x.shape) - len(expanded_param.shape)
        for _ in range(extra_dims):
            expanded_param = expanded_param.unsqueeze(-1)
        return expanded_param


class Linear(nn.Linear, BatchEnsembleMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        ensemble_size: int = 1,
        alpha_init: float | None = None,
        gamma_init: float | None = None,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False, device=device, dtype=dtype)
        # nn.init.kaiming_normal_(self.weight)
        self.init_ensemble(in_features, out_features, ensemble_size, alpha_init, gamma_init, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass through the layer.

        self.alpha, self.gamma, and self.bias are applied to the input tensor x.
        If their params are None, the input tensor is returned unchanged.
        """
        if self.alpha_init is not None:
            x = x * self.expand_param(x, self.alpha_param)
        x = F.linear(x, self.weight)
        if self.gamma_init is not None:
            x = x * self.expand_param(x, self.gamma_param)
        if self.bias_param is not None:
            x = x + self.expand_param(x, self.bias_param)
        return x


class Conv2d(nn.Conv2d, BatchEnsembleMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        ensemble_size: int = 1,
        alpha_init: float | None = None,
        gamma_init: float | None = None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        # nn.init.kaiming_normal_(self.weight)
        self.init_ensemble(in_channels, out_channels, ensemble_size, alpha_init, gamma_init, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass through the layer."""
        if self.alpha_init is not None:
            x = x * self.expand_param(x, self.alpha_param)
        x = self._conv_forward(x, self.weight, bias=None)  # Inherited from nn.Conv2d
        if self.gamma_init is not None:
            x = x * self.expand_param(x, self.gamma_param)
        if self.bias_param is not None:
            x = x + self.expand_param(x, self.bias_param)
        return x


class Ensemble_BatchNorm2d(nn.Module):
    # Applies individual batchnorm layers to each model in the batchensemble
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats=True,
        device=None,
        dtype=None,
        ensemble_size: int = 1,
    ):
        super().__init__()
        self._batchnorm_modules = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        x_chunks = torch.chunk(x, len(self._batchnorm_modules), dim=0)
        return torch.cat([bn_module(d) for bn_module, d in zip(self._batchnorm_modules, x_chunks)], dim=0)

    @property
    def weight(self):
        # Stack the weight tensors of each module
        return torch.stack([bn_module.weight for bn_module in self._batchnorm_modules])

    @property
    def bias(self):
        return torch.stack([bn_module.bias for bn_module in self._batchnorm_modules])


def ensemble_bn_init(ensemble_bn_module: nn.Module):
    for module in ensemble_bn_module._batchnorm_modules:
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
