import torch
from torch import nn, Tensor

from shared import ResidualBlock, FullyConnectedNet, KL_regularization
from stats import masked_mean, masked_std

class LightCurveDmDtBinner(nn.Module):

    def __init__(self, frequency_bin_edges: Tensor, magnitude_bin_edges: Tensor):
        super().__init__()
        torch._assert(torch.any(frequency_bin_edges > 0), "There are negative frequencies in frequency_bin_edges")
        dt_edges = 1./frequency_bin_edges
        dt_edges, _ = torch.sort(dt_edges)
        self.register_buffer('dt_edges', dt_edges)
        self.register_buffer('dm_edges', magnitude_bin_edges)
        self.dt_num_bins = self.dt_edges.numel() - 1
        self.dm_num_bins = self.dm_edges.numel() - 1
        self.apply_abs = torch.all(magnitude_bin_edges >= 0.0)

    def forward(self, light_curve: Tensor, valid_mask: Tensor, return_tuples: bool = False):
        time, mag, err = light_curve.unbind(dim=-2)
        loc = masked_mean(mag, valid_mask, keepdims=True)
        scale = masked_std(mag, loc, valid_mask, keepdims=True)
        mag = (mag - loc)/scale
        batch_size, seq_len = time.shape
        # Compute pairwise differences
        i, j = torch.triu_indices(row=seq_len, col=seq_len, offset=1)
        n_tuples = i.numel()
        dt = time[:, j] - time[:, i]
        dm = mag[:, j] - mag[:, i]
        dw = (valid_mask[:, i]*valid_mask[:, j])
        if self.apply_abs:
            dm = dm.abs()
        # Find corresponding bin in dt and dm
        dt_bin_index = torch.bucketize(dt, self.dt_edges, right=True) - 1
        dm_bin_index = torch.bucketize(dm, self.dm_edges, right=True) - 1
        dt_bin_index = dt_bin_index.clamp_(0, self.dt_num_bins - 1)
        dm_bin_index = dm_bin_index.clamp_(0, self.dm_num_bins - 1)
        # bincount requires a 1d tensor, so we offset by the batch size
        flat_index = (dt_bin_index + dm_bin_index * self.dt_num_bins).ravel()
        bins_per_sample = self.dt_num_bins * self.dm_num_bins
        batch_idx = torch.arange(batch_size, device=flat_index.device)[:, None].expand(-1, n_tuples).ravel()
        flat_global = flat_index + batch_idx * bins_per_sample        
        # Note 1: tuples where one component is invalid have zero weight
        # Note 2: We could also use the errorbars to set the weights
        # Note 3: counts will have the same type as weights
        counts = torch.bincount(
            flat_global,
            weights=dw.to(dt.dtype).ravel(),
            minlength=batch_size * bins_per_sample
        )
        dmdt = counts.view(batch_size, self.dm_num_bins, self.dt_num_bins)
        # Marginalize in the delta time dimension
        dt_counts = dmdt.sum(dim=-2, keepdims=True)
        dm_given_dt = dmdt/dt_counts
        uncertainty = (1./dt_counts).sqrt().repeat(1, self.dm_num_bins, 1)
        valid_mask_dmdt = ~torch.isnan(dm_given_dt)
        dm_given_dt[~valid_mask_dmdt] = 0.0
        uncertainty[~valid_mask_dmdt] = 0.0
        dm_given_dt = torch.cat([dm_given_dt.unsqueeze(1), uncertainty.unsqueeze(1)], dim=1)
        if return_tuples:
            return dm_given_dt, valid_mask_dmdt, dm, dt, dw
        else:
           return dm_given_dt, valid_mask_dmdt

class DMDTAutoencoder(nn.Module):

    def __init__(self, n_latent, n_hidden=64, activation=nn.GELU(), vae=False, model_decoder_variance=False, use_layer_norm=False, preprocessor_kwargs={}):
        super().__init__()
        self.preprocessor = LightCurveDmDtBinner(**preprocessor_kwargs)
        self.n_input = self.preprocessor.dm_num_bins * self.preprocessor.dt_num_bins
        self.n_latent = n_latent
        self.vae = vae
        #self.use_transformer = use_transformer
        self.encoder = nn.Sequential(
            nn.Linear(self.n_input, n_hidden), activation, 
            ResidualBlock(dim=n_hidden, activation=activation, use_layernorm=use_layer_norm),
            ResidualBlock(dim=n_hidden, activation=activation, use_layernorm=use_layer_norm),
            nn.Linear(n_hidden, n_latent*2 if vae else n_latent)
        )
        n_dec_input = n_latent if not model_decoder_variance else n_latent - 1
        self.decoder = nn.Sequential(
            nn.Linear(n_dec_input, n_hidden), 
            ResidualBlock(dim=n_hidden, activation=activation, use_layernorm=use_layer_norm),
            nn.Linear(n_hidden, self.n_input), 
        )
        
        self.decoder_variance = None
        if model_decoder_variance:
            self.decoder_variance = FullyConnectedNet(
                input_dim=1, hidden_dim=n_hidden, output_dim=1, 
                n_hidden_layers=1, activation=activation, use_layernorm=False, use_residual=False,
            )

    def forward(self, light_curve, valid_mask):
        dm_given_dt, mask_valid = self.preprocessor(light_curve, valid_mask)
        return self.infer_latent(dm_given_dt, mask_valid)
        
    def infer_latent(self, dmdt, valid_mask):
        x = dmdt.unbind(dim=1)[0].view(-1, self.n_input)
        z = self.encoder(x)
        if z.shape[1] == self.n_latent:
            return z, None
        return torch.split(z, self.n_latent, dim=-1)

    def reconstruction_loss(self, z, dmdt, valid_mask, eps=1e-6):
        dmdt, dmdt_var = dmdt.view(-1, 2, self.n_input).unbind(dim=1)
        valid_mask = valid_mask.view(-1, self.n_input)
        dmdt_var = dmdt_var.pow(2)
        if self.decoder_variance is not None:
            dmdt_rec = nn.functional.sigmoid(self.decoder.forward(z[:, :-1]))
            xvar = self.decoder_variance(z[:, -1].unsqueeze(-1)).exp()
        else:
            dmdt_rec = nn.functional.sigmoid(self.decoder.forward(z))
        with torch.no_grad():
            dmdt_var.clamp_(min=eps)
        rec_loss = 0.5*(dmdt - dmdt_rec).pow(2)/dmdt_var + 0.5*dmdt_var.log()
        #bce = nn.functional.binary_cross_entropy_with_logits(input=xhat, target=x[:, 0, :], reduction='none')
        #rec_loss = (bce*valid_mask).sum(dim=-1)/valid_mask.sum(dim=-1)
        return (rec_loss * valid_mask).sum(dim=-1)

    def criterion(self, dmdt, valid_mask):
        data_dim = valid_mask.sum(dim=[-2, -1])
        z_mu, z_logsigma = self.infer_latent(dmdt, valid_mask)
        loss_reg = torch.zeros(1).sum()
        z = z_mu
        if self.vae:
            loss_reg = KL_regularization(z_mu, z_logsigma).mean()
            z = z + z_logsigma.exp()*torch.randn_like(z, requires_grad=False)
        loss_rec = (self.reconstruction_loss(z, dmdt, valid_mask)/data_dim).mean(dim=0)        
        return {'NLL': loss_rec, 'KL': loss_reg}
