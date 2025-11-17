from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_conv2d(module: nn.Module):
    """Initializes weights of a Conv2d module using Kaiming Normal."""
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_linear(module: nn.Module):
    """Initializes weights of a Linear module using Kaiming Normal."""
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_bn(module: nn.Module):
    """Initializes weights and biases of a BatchNorm module."""
    nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def l2_weight_bias(module: nn.Module) -> torch.Tensor:
    """Calculates the L2 norm of weights and biases (if present) of a module."""
    if module.bias is None:
        return module.weight.pow(2).sum()
    else:
        return module.weight.pow(2).sum() + module.bias.pow(2).sum()


class BasicBlock(nn.Module):
    """
    Implements a basic residual block with pre-activation ordering.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the first convolution.
        dropout_rate: Dropout probability.
        conv_layer: Convolutional layer type (default: nn.Conv2d).
        norm_layer: Normalization layer type (default: nn.BatchNorm2d).
        conv_initializer: Function to initialize convolutional layers.
        norm_initializer: Function to initialize normalization layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float = 0.0,
        conv_layer: Callable[..., nn.Module] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
        conv_initializer: Callable[[nn.Module], None] = init_conv2d,
        norm_initializer: Callable[[nn.Module], None] = init_bn,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        ConvLayer = conv_layer or nn.Conv2d
        NormLayer = norm_layer or nn.BatchNorm2d
        self.conv_initializer = conv_initializer
        self.norm_initializer = norm_initializer

        self.bn1 = NormLayer(in_channels)
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = NormLayer(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_shortcut = (
            ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_initializer(self.conv1)
        self.conv_initializer(self.conv2)
        self.norm_initializer(self.bn1)
        self.norm_initializer(self.bn2)
        if not isinstance(self.conv_shortcut, nn.Identity):
            self.conv_initializer(self.conv_shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = F.dropout(out, p=self.dropout_rate, training=self.training)  # dropout_rate may be 0.
        return out + self.conv_shortcut(x)  # conv_shortcut may be identity

    def l2_loss(self):
        """Calculates the L2 regularization loss for this block."""
        l2_conv = l2_weight_bias(self.conv1) + l2_weight_bias(self.conv2)
        l2_bn = l2_weight_bias(self.bn1) + l2_weight_bias(self.bn2)
        if isinstance(self.conv_shortcut, nn.Identity):
            return l2_conv + l2_bn
        else:
            return l2_conv + l2_bn + l2_weight_bias(self.conv_shortcut)


class WideResNet(nn.Module):
    """
    Implements a Wide Residual Network (WRN) architecture.

    Args:
        in_channels: Number of input channels.
        out_features: Number of output features (classes).
        depth: Total number of layers (should be 6n + 4).
        width_multiplier: Widening factor for the network.
        dropout_rate: Dropout probability.
        conv_layer: Convolutional layer type (default: nn.Conv2d).
        linear_layer: Linear layer type (default: nn.Linear).
        norm_layer: Normalization layer type (default: nn.BatchNorm2d).
        conv_initializer: Function to initialize convolutional layers.
        linear_initializer: Function to initialize linear layers.
        norm_initializer: Function to initialize normalization layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        depth: int,
        width_multiplier: int,
        dropout_rate: float = 0.0,
        conv_layer: Callable[..., nn.Module] | None = None,
        linear_layer: Callable[..., nn.Module] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
        conv_initializer: Callable[[nn.Module], None] = init_conv2d,
        linear_initializer: Callable[[nn.Module], None] = init_linear,
        norm_initializer: Callable[[nn.Module], None] = init_bn,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("Depth should be 6n+4 (e.g., 16, 22, 28, 40).")
        self.blocks_per_group = (depth - 4) // 6
        self.dropout_rate = dropout_rate
        self.conv_initializer = conv_initializer
        self.linear_initializer = linear_initializer
        self.norm_initializer = norm_initializer
        self.ConvLayer = conv_layer or nn.Conv2d
        self.LinearLayer = linear_layer or nn.Linear
        self.NormLayer = norm_layer or nn.BatchNorm2d

        channels = [16, 16 * width_multiplier, 32 * width_multiplier, 64 * width_multiplier]
        strides = [1, 2, 2]

        self.conv1 = self.ConvLayer(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = self._make_group(channels[0], channels[1], strides[0])
        self.group2 = self._make_group(channels[1], channels[2], strides[1])
        self.group3 = self._make_group(channels[2], channels[3], strides[2])
        self.bn1 = self.NormLayer(channels[-1])
        self.fc = self.LinearLayer(channels[-1], out_features)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_initializer(self.conv1)
        self.linear_initializer(self.fc)
        self.norm_initializer(self.bn1)
        for group in [self.group1, self.group2, self.group3]:
            for block in group:
                block.reset_parameters()

    def _make_group(self, in_channels, out_channels, stride) -> nn.Sequential:
        blocks = []
        for block_idx in range(self.blocks_per_group):
            stride_ = stride if block_idx == 0 else 1
            blocks.append(
                BasicBlock(
                    in_channels,
                    out_channels,
                    stride_,
                    self.dropout_rate,
                    conv_layer=self.ConvLayer,
                    norm_layer=self.NormLayer,
                    conv_initializer=self.conv_initializer,
                    norm_initializer=self.norm_initializer,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def l2_loss(self):
        l2_conv = l2_weight_bias(self.conv1)
        l2_bn = l2_weight_bias(self.bn1)
        l2_fc = l2_weight_bias(self.fc)
        l2_groups = sum(block.l2_loss() for group in [self.group1, self.group2, self.group3] for block in group)
        return l2_conv + l2_bn + l2_fc + l2_groups
