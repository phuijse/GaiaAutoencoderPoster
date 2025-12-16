import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable

from shared import ResidualBlock, KL_regularization, FullyConnectedNet


def periodic_kernel(phase_train: Tensor, phase_test: Tensor, bandwidth: float):
    delta_phase = phase_train.unsqueeze(-1) - phase_test.unsqueeze(-2)
    return (((2.*torch.pi*delta_phase).cos() - 1.)/bandwidth**2).exp()

def kernel_interpolation(phase_test, phase_train, mag_train, mask_train, bandwidth):
    K = periodic_kernel(phase_test, phase_train, bandwidth)
    W = mask_train.unsqueeze(-2)*K
    W = W/W.sum(dim=-1, keepdims=True)
    return (W*mag_train.unsqueeze(-2)).sum(dim=-1)

def kernel_smoothing(phase, mag, mask, bandwidth):
    return kernel_interpolation(phase, phase, mag, mask, bandwidth)
      
def fold(time: Tensor, period: Tensor) -> Tensor:
    return torch.remainder(time, period)/period

def weighted_min_max_scaling(x, is_valid):
    seq_len = is_valid.size(-1)
    valid_count = is_valid.sum(dim=-1)
    N = torch.arange(x.size(0))
    x_masked = torch.where(is_valid, x, x.new_zeros(()))
    # Zeros will always be at the beginning
    sorted_x = torch.sort(x_masked, dim=-1)[0]
    p_min = sorted_x[N, (seq_len - valid_count)]
    p_max = sorted_x[N, -1]
    loc = p_min.unsqueeze(-1)
    scale = (p_max - p_min).unsqueeze(-1)
    return loc, scale

def weighted_robust_min_max_scaling(x, is_valid):
    seq_len = is_valid.size(-1)
    valid_count = is_valid.sum(dim=-1)
    N = torch.arange(x.size(0))
    x_masked = torch.where(is_valid, x, x.new_zeros(()))
    # Zeros will always be at the beginning
    sorted_x = torch.sort(x_masked, dim=-1)[0]
    p_min = sorted_x[N, (seq_len-valid_count) + ((valid_count-1)*0.02).long()]
    p_max = sorted_x[N, (seq_len-valid_count) + ((valid_count-1)*0.98).long()]
    loc = p_min.unsqueeze(-1)
    scale = (p_max - p_min).unsqueeze(-1)
    return loc, scale

def weighted_mean_std_scaling(x, w):
    sumw = w.sum(dim=-1, keepdims=True)
    loc = (x*w).sum(dim=-1, keepdims=True)/sumw
    scale = ((w*(x - loc).pow(2)).sum(dim=-1, keepdims=True)/sumw).sqrt()
    return loc, scale

def weighted_mean(x, w):
    return (x*w).sum(dim=-1)/w.sum(dim=-1)

def weighted_std(x, w, m):
    return (((x - m).pow(2)*w).sum(dim=-1)/(w.sum(dim=-1)-1)).sqrt()
    
def flag_faint_outliers_relative_phase(phase, mag, valid_mask, w=5, threshold=5):
    idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    mag_v = mag.index_select(0, idx)
    phase_v = phase.index_select(0, idx)
    perm = torch.argsort(phase_v)
    pad = w // 2
    win_phase = F.pad(phase_v[perm].view(1, 1, -1), (pad, pad), mode="circular").view(-1).unfold(0, w, 1)
    win_mag = F.pad(mag_v[perm].view(1, 1, -1), (pad, pad), mode="circular").view(-1).unfold(0, w, 1)
    # Circular distance in phase
    win_phase_minus_center = torch.cat([win_phase[:, :pad], win_phase[:, pad+1:]], dim=-1)
    win_phase_delta = win_phase_minus_center - win_phase[:, pad].unsqueeze(-1)
    win_phase_delta_circular = torch.remainder(win_phase_delta - 0.5, 1.0) - 0.5
    # Epanechnikov kernel
    win_phase_weight = 0.75*(1. - win_phase_delta_circular.pow(2))
    win_mag_minus_center = torch.cat([win_mag[:, :pad], win_mag[:, pad+1:]], dim=-1) 
    win_mag_loc = weighted_mean(win_mag_minus_center, win_phase_weight)
    win_mag_scale = weighted_std(win_mag_minus_center, win_phase_weight, win_mag_loc.unsqueeze(-1))
    win_threshold = win_mag_loc + threshold * win_mag_scale
    sigma_threshold = torch.empty_like(mag_v)
    sigma_threshold.index_copy_(0, perm, win_threshold.squeeze(-1))
    discard_v = mag_v > sigma_threshold
    discard = torch.zeros_like(valid_mask, dtype=torch.bool)
    discard.index_copy_(0, idx, discard_v)
    return discard

class LightCurveFolder(nn.Module):

    def __init__(self, outlier_threshold: float, outlier_window: int, scaler: Callable | None, sort_invalid_last: bool = True):
        super().__init__()
        # self.phase_embedder = FourierSeriesEmbedder(n_harmonics=n_harmonics)
        self.outlier_threshold = outlier_threshold
        torch._assert(outlier_window % 2 != 0, 'Window size has to be odd')
        self.outlier_window = outlier_window
        self.sort_invalid_last = sort_invalid_last
        self.register_buffer('phase_interp',  torch.linspace(0, 1, 50))
        self.scaler = scaler

    def forward(self, light_curve, is_valid, frequency, fold_with_double_period=True, return_all_masks: bool=False):
        time, mag, err = light_curve.unbind(dim=-2)
        bsize, seq_len = time.shape
        # 1) Fold
        period = 1./frequency.unsqueeze(-1)
        if fold_with_double_period:
            period = period * 2.0
        phase = fold(time, period)
        # 2) Remove outliers
        is_faint_outlier = torch.stack([flag_faint_outliers_relative_phase(
            phase[k], mag[k], is_valid[k], w=self.outlier_window, threshold=self.outlier_threshold
        ) for k in range(bsize)]) # TRACE FRIENDLY OR NOT?
        is_valid = is_valid & ~is_faint_outlier
        # 3) Align at min phase
        weights = is_valid
        smooth_mag = kernel_interpolation(self.phase_interp, phase, mag, weights, bandwidth=0.1)        
        phase_shift = self.phase_interp[torch.argmax(smooth_mag, dim=-1)].reshape(-1, 1)
        #smooth_min, smooth_max = smooth_mag.amin(dim=-1, keepdims=True), smooth_mag.amax(dim=-1, keepdims=True)
        phase = phase - phase_shift
        # 4) Sort in phase (put invalid points at the end)
        sort_keys = phase
        if self.sort_invalid_last:
            big = torch.finfo(light_curve.dtype).max
            sort_keys = torch.where(is_valid, phase, torch.as_tensor(big, device=light_curve.device, dtype=light_curve.dtype))
        sort_idx = torch.argsort(sort_keys, dim=-1)
        phase = torch.take_along_dim(phase, sort_idx, dim=-1)
        mag = torch.take_along_dim(mag, sort_idx, dim=-1)
        err = torch.take_along_dim(err, sort_idx, dim=-1)
        is_valid = torch.take_along_dim(is_valid, sort_idx, dim=-1)        
        # 5) Normalize
        mag_loc, mag_scale = 0., 1.
        if self.scaler is not None:
            mag_loc, mag_scale = self.scaler(mag, is_valid)
        #mag_loc = smooth_min
        #mag_scale = smooth_max - smooth_min
        new_mag = (mag - mag_loc)/mag_scale
        new_err = err/mag_scale
        output = {
            'light_curve': torch.stack([phase, new_mag, new_err]).permute(1, 0, 2),
            'mask_valid': is_valid,
            #'phase_embedding': self.phase_embedder(phase),
        }
        if return_all_masks:            
            output['mask_faint_outliers'] = torch.take_along_dim(is_faint_outlier, sort_idx, dim=-1)
        return output

class FourierSeriesEmbedder(nn.Module):

    def __init__(self,
                 n_harmonics: int,
                 fundamental_period: float = 1.0,
                 ):
        super().__init__()
        if n_harmonics < 1:
            raise ValueError("Number of harmonics has to be greater than zero")
        k = torch.arange(1, n_harmonics+1)
        self.register_buffer('k', k.reshape(1, -1, 1))
        self.T = fundamental_period

    def forward(self, time: Tensor) -> Tensor:
        angle = 2.*torch.pi*self.k*time.unsqueeze(1)/self.T
        return torch.cat([angle.cos(), angle.sin()], dim=1).transpose(-2, -1)

class FoldedEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 num_layers: int,
                 dropout_pbb: float,
                 context_strategy: str,
                 bidirectional: bool,
                 split_latent: bool,
                ):
        super().__init__()
        self.n_latent = latent_dim
        self.context = context_strategy
        self.rnn = nn.GRU(num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          input_size=input_dim,
                          dropout=dropout_pbb if num_layers > 1 else 0,
                          hidden_size=hidden_dim)
        context_size = hidden_dim
        if bidirectional:
            context_size *= 2
        if context_strategy == 'attention':
            self.attention = nn.Linear(context_size, 1, bias=False)
        self.latent = nn.Sequential(
            nn.Linear(context_size, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim*2 if split_latent else latent_dim)
        )

    def forward(self,
                data: Tensor,
                valid_mask: Tensor
                ) -> tuple[Tensor, Tensor]:
        
        ht, _ = self.rnn(data)
        if self.context == 'last':
            last_valid_index = valid_mask.sum(dim=-1) - 1
            bsize = ht.shape[0]
            ht = ht[torch.arange(bsize), last_valid_index]
        elif self.context == 'average':
            m = valid_mask.unsqueeze(-1)
            ht = (ht*m).sum(dim=1)/m.sum(dim=1)
        elif self.context == 'attention':
            attn_weights = self.attention(ht).exp()*valid_mask.unsqueeze(-1)
            ht = (attn_weights * ht).sum(dim=1)/attn_weights.sum(dim=1)
        else:
            raise ValueError("Invalid context setting")
        z = self.latent(ht)
        if z.shape[-1] == self.n_latent:
            return z, None
        return z.split(self.n_latent, dim=-1)
        

class FoldedDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 activation: nn.Module,
                 use_layernorm: bool,
                ):
        super().__init__()
        dec_input = input_dim + latent_dim
        self.layers = nn.Sequential(
            nn.Linear(dec_input, hidden_dim),
            activation,
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            nn.Linear(hidden_dim, 1)
        )       
        
    def forward(self,
                phase_embedding: Tensor,
                z: Tensor
                ) -> Tensor:
        seqlen = phase_embedding.shape[1]
        zt = torch.cat([
            phase_embedding,
            z.unsqueeze(1).repeat(1, seqlen, 1)
        ], dim=-1)
        xhat = self.layers(zt).squeeze(-1)
        return xhat
    

class FoldedAutoencoder(nn.Module):

    def __init__(self, n_harmonics, n_latent, activation, n_hidden, context, 
                 vae, model_decoder_variance, use_layernorm, preprocessor_kwargs={}):
        super().__init__()
        self.n_latent = n_latent
        self.vae = vae
        self.phase_embedder = FourierSeriesEmbedder(n_harmonics=n_harmonics)
        self.encoder = FoldedEncoder(
            input_dim=1+2*n_harmonics, hidden_dim=n_hidden, latent_dim=n_latent, split_latent=vae,
            num_layers=2, context_strategy=context, dropout_pbb=0.1, bidirectional=False
        )
        n_dec_input = n_latent if not model_decoder_variance else n_latent - 1
        self.decoder = FoldedDecoder(
            input_dim=2*n_harmonics, hidden_dim=n_hidden, latent_dim=n_dec_input, activation=activation, use_layernorm=use_layernorm
        )
        self.preprocessor = LightCurveFolder(**preprocessor_kwargs)
        self.decoder_variance = None
        if model_decoder_variance:
            self.decoder_variance = FullyConnectedNet(
                input_dim=1, hidden_dim=n_hidden, output_dim=1, 
                n_hidden_layers=1, activation=nn.GELU(), use_layernorm=use_layernorm, use_residual=False,
            )
    
    def reconstruction_loss(self, z, folded_light_curve, valid_mask, eps=1e-6):
        phase, mag, err = folded_light_curve.unbind(dim=-2)
        phase_emb = self.phase_embedder(phase)
        folded_var = err.pow(2)
        if self.decoder_variance is not None:
            xvar = self.decoder_variance(z[:, -1].unsqueeze(-1)).exp()
            folded_var = folded_var + xvar
            folded_rec = self.decoder.forward(phase_emb, z[:, :-1])
        else:
            folded_rec = self.decoder.forward(phase_emb, z)
        with torch.no_grad():
            folded_var.clamp_(min=eps)
        folded_loss = 0.5*(mag - folded_rec).pow(2)/folded_var + 0.5*folded_var.log()
        return (folded_loss * valid_mask).sum(dim=-1)

    def forward(self, light_curve, valid_mask, frequency):
        pre = self.preprocessor(light_curve, valid_mask, frequency)
        return self.infer_latent(pre['light_curve'], pre['mask_valid'])
        
    def infer_latent(self, folded_light_curve, valid_mask):
        phase, mag, err = folded_light_curve.unbind(dim=-2)
        emb = torch.cat([self.phase_embedder(phase), mag.unsqueeze(-1)], dim=-1)
        return self.encoder.forward(emb, valid_mask)

    def criterion(self, folded_light_curve, valid_mask, beta=0.01): 
        data_dim = valid_mask.sum(dim=-1)
        #_, phase_emb, mag_norm, err_norm = self.adapter.preprocess(data, valid_mask, frequency)
        z_mu, z_logsigma = self.infer_latent(folded_light_curve, valid_mask)
        z = z_mu
        loss_reg = torch.zeros(1).sum()
        if self.vae:
            z = z + z_logsigma.exp()*torch.randn_like(z_logsigma, requires_grad=False) 
            loss_reg = KL_regularization(z_mu, z_logsigma).mean()
        loss_rec = (self.reconstruction_loss(z, folded_light_curve, valid_mask)/data_dim).mean(dim=0)
        return {'NLL': loss_rec, 'KL': loss_reg}
