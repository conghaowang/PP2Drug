import torch

def vp_logsnr(t, beta_d, beta_min):
    t = torch.as_tensor(t)
    return - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
    
def vp_logs(t, beta_d, beta_min):
    t = torch.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))