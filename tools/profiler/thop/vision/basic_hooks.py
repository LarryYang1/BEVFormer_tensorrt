import argparse
import logging
from .calc_func import *
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

multiply_adds = 1


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = calculate_parameters(m.parameters())


def zero_ops(m, x, y):
    m.total_ops += calculate_zero_ops()

    if isinstance(x, tuple):
        x_tensor = x[0]
    else:
        x_tensor = x

    if isinstance(y, tuple):
        y_tensor = y[0]
    else:
        y_tensor = y

    m.total_memory += (l_prod(x_tensor.shape) + l_prod(y_tensor.shape))


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.weight.shape),
        groups = m.groups,
        bias = m.bias
    )

    m.total_memory += (l_prod(x.shape) + l_prod(y.shape) + l_prod(m.weight.shape))

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # m.total_ops += calculate_conv(
    #     bias_ops,
    #     torch.zeros(m.weight.size()[2:]).numel(),
    #     y.nelement(),
    #     m.in_channels,
    #     m.groups,
    # )


def count_convNd_ver2(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    # # Cout x Cin x Kw x Kh
    # kernel_ops = m.weight.nelement()
    # if m.bias is not None:
    #     # Cout x 1
    #     kernel_ops += + m.bias.nelement()
    # # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    # m.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])
    m.total_ops += calculate_conv(m.bias.nelement(), m.weight.nelement(), output_size)


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    # TODO: add test cases
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2
    m.total_ops += flops
    # m.total_ops += (l_prod(x) + l_prod(y))


# def count_layer_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


# def count_instance_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


def count_prelu(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        m.total_ops += calculate_relu(nelements)
    m.total_ops += (l_prod(x) + l_prod(y))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += calculate_relu_flops(list(x.shape))
    m.total_ops += (l_prod(x) + l_prod(y))
    m.total_meory += (l_prod(x) + l_prod(y))


def count_softmax(m, x, y):
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    m.total_ops += calculate_softmax(batch_size, nfeatures)


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel()
    m.total_ops += calculate_avgpool(num_elements)
    m.total_meory += (l_prod(x) + l_prod(y))


def count_adap_avgpool(m, x, y):
    kernel = torch.div(
        torch.DoubleTensor([*(x[0].shape[2:])]), 
        torch.DoubleTensor([*(y.shape[2:])])
    )
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    m.total_ops += calculate_adaptive_avg(total_add, num_elements)
    m.total_meory += (l_prod(x) + l_prod(y))


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        m.total_ops += 0
    else:
        x = x[0]
        m.total_ops += calculate_upsample(m.mode, y.nelement())

    m.total_memory += (l_prod(x.shape) + l_prod(y.shape))


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)

    m.total_memory += (l_prod(x[0].shape) + l_prod(y[0].shape) + l_prod(m.weight.shape))
