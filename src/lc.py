import torch
from torch import nn, Tensor

from stats import masked_median, tukey_fence

def pad_crop_lc(lc, max_len=200): #This function will not be traced
    n_obs = lc.shape[-1]
    mask = torch.ones(max_len, dtype=torch.bool)
    if n_obs > max_len:
        lc = lc[:, :max_len]
    elif n_obs < max_len:
        lc = nn.functional.pad(lc, (0, max_len - n_obs), mode='constant', value=0.0)  
        mask[n_obs:] = False
    return lc, mask

def flag_extreme_errors(err: torch.Tensor, valid_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    med = masked_median(err, valid_mask)
    mad = masked_median((err - med).abs(), valid_mask)
    outliers = (err > med + threshold * 1.4826 * mad) & valid_mask
    # Protection in case mad == 0
    return torch.where(mad > 0, outliers, torch.zeros_like(valid_mask, dtype=torch.bool))

def flag_errors_by_mag_bin(mag: torch.Tensor, err: torch.Tensor, threshold: float) -> torch.Tensor:
    mag_bin = torch.arange(4, 15, 0.5, device=mag.device, dtype=mag.dtype)
    mag_index = torch.bucketize(mag.contiguous(), mag_bin) - 1
    mag_index = mag_index.clamp_min(0) 
    err_bin_median, err_bin_mad = torch.tensor(
        [(0.0072346226959187085, 0.000556539138304629), # 4-4.5
         (0.007277966762969894, 0.0006341286519172679),
         (0.00751161242916645, 0.0003549648152118088),
         (0.003841922673385825, 0.0009420231103613149),
         (0.0028823807111780318, 0.00028250797805103186),
         (0.0025199820839086813, 0.0003671334743430223),
         (0.002398287605344948, 0.0002502935178378177),
         (0.0022981766524078407, 0.0002689046639815398),
         (0.0023829754751450204, 0.00029003694763449195),
         (0.002103783741402776, 0.00022834709513293414),
         (0.001680953479241577, 0.00017038730690437185),
         (0.0015808278649738481, 0.0001357746865960827),
         (0.001605813753803031, 0.0001559842453087506),
         (0.001559661122420557, 0.00015928125842445138),
         (0.0021289871851236427, 0.00016317128601887779),
         (0.0017530130024650842, 0.00025731447985278253),
         (0.0011007070529042728, 0.00010090888791834001),
         (0.0012090172094871771, 0.00013394119080470477),
         (0.001648936969040879, 9.993663236324522e-05),
         (0.0017209489436539936, 0.00010049517959384276),
         (0.0018836719120070068, 0.00014741960338955857),
         (0.002138300268623284, 0.00019051290448675263)], # 14.5 - inf
        device=mag.device, dtype=mag.dtype
    ).T    
    err_threshold = err_bin_median + threshold * 1.4826 * err_bin_mad
    return err > err_threshold[mag_index]
    
class LightCurveCleaner(nn.Module):

    def __init__(self, 
                 relative_error_threshold: float, 
                 magbin_error_threshold: None | float, 
                 drop_singleton_outliers: bool,
                 sort_invalid_last: bool = True,
                ):
        super().__init__()
        self.err_rel_thresh = relative_error_threshold
        self.err_mag_thresh = magbin_error_threshold
        self.sort_invalid_last = sort_invalid_last
        self.drop_singleton_outliers = drop_singleton_outliers

    def forward(
        self, 
        light_curve: Tensor, 
        non_padded_mask: Tensor, 
        return_all_masks: bool = False
    ) -> dict[str, Tensor]: 
        """
        light_curve is a B x 3 x L float32 tensor, second dim contains obstimes, vals and valerrs
        padding_mask is B x L boolean tensor indicating what points have not been padded with zeros
        """
        torch._assert(light_curve.size(-2) == 3, "expected channel order [N, 3, L]")
        time, mag, err = light_curve.unbind(dim=-2)
        # Initial set of valid points: non padded and non nan/inf
        non_inf_values = torch.isfinite(light_curve).any(dim=-2)
        good_errors = err > 0.0
        is_valid = non_padded_mask & non_inf_values & good_errors
        time, mag, err = light_curve.unbind(dim=-2)
        # Find large errorbars relative to this light curve
        is_rel_err_outlier = flag_extreme_errors(err, is_valid, self.err_rel_thresh)
        if self.err_mag_thresh is not None:
            # Find large errorbars relative to their magnitude bin
            is_magbin_err_outlier = flag_errors_by_mag_bin(mag, err, self.err_mag_thresh)
            mask_err = is_rel_err_outlier & is_magbin_err_outlier
        else:
            is_magbin_err_outlier = None
            mask_err = is_rel_err_outlier
        is_valid = is_valid & ~mask_err
        is_outlier_mag = None
        if self.drop_singleton_outliers:
            bright_outliers, faint_outliers = tukey_fence(mag, is_valid, k=3, iqr_lower_bound=0.015)
            is_outlier_mag = (bright_outliers | faint_outliers)
            # Look for those that have only one outlier
            pardon_all = (is_outlier_mag.to(torch.int64).sum(dim=-1, keepdims=True) > 1)
            is_outlier_mag = is_outlier_mag & ~pardon_all
            is_valid = is_valid & ~is_outlier_mag
        sort_keys = time
        if self.sort_invalid_last:
            big = torch.finfo(light_curve.dtype).max
            sort_keys = torch.where(is_valid, time, torch.as_tensor(big, device=light_curve.device, dtype=light_curve.dtype))
        sort_idx = torch.argsort(sort_keys, dim=-1)
        light_curve = torch.take_along_dim(light_curve, sort_idx.unsqueeze(-2).expand(-1, light_curve.size(1), -1), dim=-1)
        is_valid = torch.take_along_dim(is_valid, sort_idx, dim=-1)
        output = {'light_curve': light_curve, 'mask_valid': is_valid}
        if return_all_masks:
            output['mask_outlier_relative_err'] = torch.take_along_dim(is_rel_err_outlier, sort_idx, dim=-1)
            if is_magbin_err_outlier is not None:
                output['mask_outlier_magbin_err'] = torch.take_along_dim(is_magbin_err_outlier, sort_idx, dim=-1)
            if is_outlier_mag is not None:
                output['mask_outlier_singleton_mag'] = torch.take_along_dim(is_outlier_mag, sort_idx, dim=-1) 
        return output