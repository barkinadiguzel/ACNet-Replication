import torch

def pad_kernel(kernel, target_size=3):
    out = torch.zeros(
        kernel.size(0),
        kernel.size(1),
        target_size,
        target_size,
        device=kernel.device
    )

    kh, kw = kernel.shape[-2:]
    sh = (target_size - kh) // 2
    sw = (target_size - kw) // 2

    out[:, :, sh:sh+kh, sw:sw+kw] = kernel
    return out


def fuse_kernels(k3x3, k1x3, k3x1):
    k1x3 = pad_kernel(k1x3)
    k3x1 = pad_kernel(k3x1)

    return k3x3 + k1x3 + k3x1
