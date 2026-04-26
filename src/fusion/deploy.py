import torch
from fusion.kernel_fusion import fuse_kernels
from blocks.bn_fusion import fuse_bn


def convert_acb_to_single_conv(acb_block, bn_list):
    k3, b3 = fuse_bn(acb_block.conv3x3.weight, None, bn_list[0])
    k1, b1 = fuse_bn(acb_block.conv1x3.weight, None, bn_list[1])
    k2, b2 = fuse_bn(acb_block.conv3x1.weight, None, bn_list[2])

    fused_kernel = fuse_kernels(k3, k1, k2)
    fused_bias = b3 + b1 + b2

    return fused_kernel, fused_bias
