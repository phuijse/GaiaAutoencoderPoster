import torch
from torch import Tensor

def nanstd(x: Tensor, dim: int, ddof=1, keepdims=False):
    N = x.shape[dim]
    mean = x.nanmean(dim=dim).unsqueeze(dim)
    res = ((x - mean).pow(2).nansum(dim=dim)/(N-ddof)).sqrt()
    if keepdims:
        res = res.unsqueeze(dim)
    return res

def masked_mean(x: Tensor, mask: Tensor, keepdims: bool=True):
    return (x * mask).sum(dim=-1, keepdims=keepdims) / mask.sum(dim=-1, keepdims=keepdims)

def masked_std(x: Tensor, masked_mean: Tensor, mask: Tensor, keepdims: bool=True):
    return (((x - masked_mean).pow(2) * mask).sum(dim=-1, keepdims=keepdims) / (mask.sum(dim=-1, keepdims=keepdims)-1)).sqrt()

def create_sinusoidal_model(time, freq):
    N = time.shape[0]
    return torch.stack([torch.ones_like(time), (2.*torch.pi*freq*time).cos(), (2.*torch.pi*freq*time).sin()]).T 

def lstsq_fit(X, y, yerr):
    w = 1. / yerr
    return torch.linalg.lstsq(X * w.unsqueeze(-1), (w * y)).solution
    
def compute_amplitude(lc, freq):
    X = create_sinusoidal_model(lc[0], freq)
    theta = lstsq_fit(X, lc[1], lc[2])
    return (theta[1]**2 + theta[2]**2).sqrt()

def batched_masked_quantile(x: Tensor, mask: Tensor, q: float):
    device, dtype = x.device, x.dtype
    n_valid = mask.sum(dim=-1)
    last_idx = (n_valid - 1).clamp_min(0)
    # Push invalids to the end
    inf = torch.tensor(float('inf'), device=device, dtype=dtype) # Move this to register?
    filled = torch.where(mask, x, inf)
    sorted_vals, _ = torch.sort(filled, dim=-1)
    # Find the closest points to the desired quantiles
    r = q * last_idx.to(dtype)
    r0 = torch.floor(r).to(torch.int64)
    r1 = torch.ceil(r).to(torch.int64)
    # Clamp just to be safe
    r0 = torch.minimum(r0, last_idx)
    r1 = torch.minimum(r1, last_idx)
    # Interpolate between the values closest to the q percentile
    vals0 = torch.gather(sorted_vals, -1, r0.unsqueeze(-1)).squeeze(-1)
    vals1 = torch.gather(sorted_vals, -1, r1.unsqueeze(-1)).squeeze(-1)
    t = (r - r0.to(dtype))
    out = vals0 * (1.0 - t) + vals1 * t
    return out.reshape(-1, 1) # preserve original dims

def masked_median(x: Tensor, is_valid: Tensor) -> Tensor:
    return batched_masked_quantile(x, is_valid, 0.5)

def tukey_fence(x: Tensor, is_valid: Tensor, k: float, iqr_lower_bound: float) -> Tensor:
    q25 = batched_masked_quantile(x, is_valid, 0.25)
    q75 = batched_masked_quantile(x, is_valid, 0.75)
    iqr = q75 - q25
    extent = iqr.mul_(k).clamp_min_(iqr_lower_bound)
    return (x < q25 - extent) & is_valid, (x > q75 + extent) & is_valid
