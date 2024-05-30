import torch
import numpy as np
from torch_scatter import scatter_mean

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


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def scatter_mean_flat(tensor, batch):
    """
    Scatter the mean over all non-batch dimensions.
    """
    loss = scatter_mean(tensor.sum(-1), batch, dim=0)
    # print('scatter loss size', loss.size())
    loss = torch.mean(loss)
    # print('mean loss size', loss.size())
    return loss

def scatter_flat(tensor, batch):
    loss = scatter_mean(tensor, batch, dim=0)       # already summed along the final dim
    loss = torch.mean(loss)
    return loss


def center2zero(x, mean_dim=0):
    if x == None:
        return None
    mean = torch.mean(x, dim=mean_dim, keepdim=True)
    assert mean.size(-1) == 3
    x = x - mean
    return x


def center2zero_with_mask(x, node_mask, check_mask=True):
    if x == None:
        return None
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    batch_size = x.size(0)
    x_ = x.detach().clone()
    # for i in range(batch_size):
    #     masked_max_abs_value = (x[i] * (~node_mask[i])).abs().sum().item()
    #     assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    #     N = node_mask[i].sum(1, keepdims=True)

    #     mean = torch.sum(x[i], dim=1, keepdim=True) / N
    #     x_[i] = x[i] - mean[i] * node_mask[i]

    # node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x_ * (~node_mask)).abs().sum().item()
    if check_mask:
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x_, dim=1, keepdim=True) / N
    print(mean)
    assert mean.size(-1) == 3 
    x_ = x_ - mean * node_mask
    return x_


def center2zero_combined_graph(x, node_mask, Gt_mask, mode='individual'):
    GT_mask = node_mask & (~Gt_mask)
    # print(node_mask.size(), Gt_mask.size(), GT_mask.size())
    x_ = x.detach().clone()
    xt_ = center2zero_with_mask(x_, Gt_mask, check_mask=False)
    xT_ = center2zero_with_mask(x_, GT_mask, check_mask=False)
    return xt_*Gt_mask + xT_*GT_mask


def center2zero_sparse_graph(x, Gt_mask, batch_info, mode='GT'):
    if mode == 'GT':
        GT_mask = ~Gt_mask
        mean = scatter_mean(x[GT_mask], batch_info[GT_mask], dim=0)
        assert mean.size(-1) == 3
        x = x - mean[batch_info]
        return x
    elif mode == 'G0':
        mean = scatter_mean(x[Gt_mask], batch_info[Gt_mask], dim=0)
        assert mean.size(-1) == 3
        x = x - mean[batch_info]
        return x
    elif mode == 'individual':
        raise NotImplementedError
    else:
        raise NotImplementedError


def sample_zero_center_gaussian(size, device):
    # assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = center2zero(x)
    return x_projected


def sample_zero_center_gaussian_with_mask(size, device, node_mask):
    # assert len(size) == 3
    x = torch.randn(size, device=device)

    # node_mask = node_mask.unsqueeze(-1)
    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = center2zero_with_mask(x_masked, node_mask)
    return x_projected