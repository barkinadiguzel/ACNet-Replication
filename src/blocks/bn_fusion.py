import torch

def fuse_bn(conv_w, conv_b, bn):
    if conv_b is None:
        conv_b = torch.zeros(conv_w.size(0), device=conv_w.device)

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)

    w_fused = conv_w * (gamma / std).reshape(-1, 1, 1, 1)
    b_fused = beta + (conv_b - mean) * (gamma / std)

    return w_fused, b_fused
