import torch
from torch import nn, Tensor
from gaiaxpy import pwl_to_wl

from shared import FullyConnectedNet, ConvolutionalNet, KL_regularization


class XPMeanSpectraPreprocessor(nn.Module):

    def __init__(self, clip_left: int = 5, clip_right: int = 5, num_wl: int = 50):
        super().__init__()
        # Will clip the first clip_left and the last clip_right pseudo-wavelengths
        self.clip_left = clip_left
        self.clip_right = -clip_right if clip_right > 0 else None
        pwl = torch.arange(clip_left, num_wl - clip_right, 1)
        self.register_buffer('wl_bp', torch.from_numpy(pwl_to_wl(pwl=pwl.numpy(), band='BP')))
        self.register_buffer('wl_rp', torch.from_numpy(pwl_to_wl(pwl=pwl.numpy(), band='RP')))

    def forward(self, xp_spectra: Tensor, fmedian_g: Tensor = None): 
        """
        xp_spectra is a tensor of shape [batch_size, 2, 2, 60]
        second dim represents bp = 0, rp = 1
        third dim represents flux = 0 fluxerr = 1
        """
        xp_spectra = xp_spectra[:, :, :, self.clip_left:self.clip_right]
        if fmedian_g is not None:
            # Scale it to 15 magnitudes
            scale_factor = 1./(10**(-0.4*(15. - fmedian_g))).reshape(-1, 1, 1, 1)
        else:
            # Divide by the maximum flux in both bp and rp to preserve relative difference (color)
            #max_flux = xp_spectra[:, :, 0, :].amax(dim=(1, 2), keepdims=True)
            max_flux = torch.quantile(xp_spectra[:, :, 0, :].flatten(1, 2), 0.995, dim=-1)
            scale_factor = 1./max_flux.view(-1, 1, 1, 1)
        return xp_spectra*scale_factor


class XPMeanSpectraConvEncoder(nn.Module):
    def __init__(self, n_input, n_latent, n_hidden, split_latent=False, filters=[16, 32], kernels=[5, 3]):
        super().__init__()
        self.n_latent = n_latent        
        self.bp_nnet = ConvolutionalNet(n_hidden, filters, kernels, activation=nn.ReLU(), dropout_pbb=0.1)
        self.rp_nnet = ConvolutionalNet(n_hidden, filters, kernels, activation=nn.ReLU(), dropout_pbb=0.1)
        self.xp_mixer = nn.Sequential(
            nn.Linear(n_hidden*2, n_hidden),
            #nn.LayerNorm(n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 2*n_latent if split_latent else n_latent)
        )
        
    def forward(self, bp_norm, rp_norm):
        z_bp = self.bp_nnet(bp_norm.unsqueeze(1))
        z_rp = self.rp_nnet(rp_norm.unsqueeze(1))
        z = self.xp_mixer(torch.cat([z_bp, z_rp], dim=-1))
        if z.shape[-1] == self.n_latent:
            return z, None
        return z.split(self.n_latent, dim=-1)

import torch
import torch.nn as nn

class ConvolutionalDecoder(nn.Module):
    def __init__(self, input_dim, seq_len, filters, kernels, activation):
        super().__init__()
        self.seq_len = seq_len
        self.last_channels = filters[-1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.last_channels * seq_len),
            activation
        )
        layers = []
        rev_filters = list(reversed(filters))
        rev_kernels = list(reversed(kernels))
        in_channels = rev_filters[0]
        for out_channels, k in zip(rev_filters[1:], rev_kernels[:-1]):
            layers.append(nn.Conv1d(in_channels, out_channels,
                                    kernel_size=k, stride=1, padding=k // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation)
            in_channels = out_channels
        k_final = rev_kernels[-1]
        layers.append(
            nn.Conv1d(in_channels, 1, kernel_size=k_final, stride=1, padding=k_final // 2)
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.last_channels, self.seq_len)
        x = self.conv(x)
        return x.squeeze(1)


class XPMeanSpectraConvDecoder(nn.Module):
    def __init__(self, n_latent, n_hidden, seq_len=50, filters=[16, 32], kernels=[5, 3]):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden        
        self.xp_unmixer = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 2 * n_hidden),
            nn.GELU()
        )

        self.bp_decoder = ConvolutionalDecoder(
            input_dim=n_hidden,
            seq_len=seq_len,
            filters=filters,
            kernels=kernels,
            activation=nn.ReLU()
        )
        self.rp_decoder = ConvolutionalDecoder(
            input_dim=n_hidden,
            seq_len=seq_len,
            filters=filters,
            kernels=kernels,
            activation=nn.ReLU()
        )

    def forward(self, z):
        h = self.xp_unmixer(z)
        h_bp, h_rp = h.chunk(2, dim=-1)
        bp_hat = self.bp_decoder(h_bp)
        rp_hat = self.rp_decoder(h_rp)
        return bp_hat, rp_hat
        

class XPMeanSpectraFCEncoder(nn.Module):

    def __init__(self, n_input, n_latent, n_hidden, split_latent, n_residual_layers, use_layernorm, activation=nn.GELU()):
        super().__init__()
        self.n_latent = n_latent
        self.nnet = FullyConnectedNet(
            input_dim=n_input, hidden_dim=n_hidden, output_dim = n_latent if not split_latent else 2*n_latent,
            n_hidden_layers=n_residual_layers, use_residual=True, activation=activation, use_layernorm=use_layernorm, 
        )        
        
    def forward(self, bp_norm, rp_norm):
        z = self.nnet(torch.cat([bp_norm, rp_norm], dim=-1))
        if z.shape[-1] == self.n_latent:
            return z, None
        return z.split(self.n_latent, dim=-1)
        
class XPMeanSpectraFCDecoder(nn.Module):
    def __init__(self, n_input, n_latent, n_hidden, n_residual_layers, use_layernorm, activation=nn.GELU()):
        super().__init__()
        self.n_input = n_input
        self.nnet = FullyConnectedNet(
            input_dim=n_latent, hidden_dim=n_hidden, output_dim=n_input, 
            n_hidden_layers=n_residual_layers, use_residual=True, activation=activation, use_layernorm=use_layernorm
        )
        
    def forward(self, z):
        xhat = self.nnet(z)
        #xhat = nn.functional.gelu(xhat)
        return torch.split(xhat, self.n_input // 2, dim=-1)
    

class XPMeanSpectraAutoencoder(nn.Module):

    def __init__(self, n_latent, n_hidden=128, n_input=100, use_layernorm=False, model_decoder_variance=False, vae=False, preprocessor_kwargs={}):
        super().__init__()
        self.vae = vae
        #self.encoder = XPMeanSpectraConvEncoder(
        #    n_input=n_input, n_latent=n_latent, n_hidden=n_hidden, split_latent=vae
        #)
        self.encoder = XPMeanSpectraFCEncoder(
            n_input=n_input, n_latent=n_latent, n_hidden=n_hidden, split_latent=vae, use_layernorm=use_layernorm, n_residual_layers=3,
        )
        n_dec_input = n_latent if not model_decoder_variance else n_latent - 1
        self.decoder = XPMeanSpectraFCDecoder(
            n_input=n_input, n_latent=n_dec_input, n_hidden=n_hidden, n_residual_layers=3, use_layernorm=use_layernorm
        )
        #self.decoder = XPMeanSpectraConvDecoder(n_dec_input, n_hidden=n_hidden, seq_len=50)
        self.preprocessor = XPMeanSpectraPreprocessor(**preprocessor_kwargs)
        self.decoder_variance = None
        if model_decoder_variance:
            self.decoder_variance = FullyConnectedNet(
                input_dim=1, hidden_dim=n_hidden, output_dim=1, 
                n_hidden_layers=1, activation=nn.GELU(), use_layernorm=use_layernorm, use_residual=False,
            )

    def infer_latent(self, xp_norm): # Assumes that xp has already been preprocessed
        flux_norm = xp_norm[:, :, 0, :]
        bp_norm, rp_norm   = flux_norm.unbind(dim=1)
        return self.encoder.forward(bp_norm, rp_norm)

    def forward(self, xp):
        xp_norm = self.preprocessor(xp)        
        return self.infer_latent(xp_norm)

    def reconstruction_loss(self, z, xp_norm, eps=1e-6):
        flux, err = xp_norm.unbind(dim=2)
        bp_norm, rp_norm   = flux.unbind(dim=1)
        bperr_norm, rperr_norm = err.unbind(dim=1)
        bp_var = bperr_norm.pow(2)
        rp_var = rperr_norm.pow(2)
        if self.decoder_variance is not None:
            bp_rec, rp_rec = self.decoder.forward(z[:, :-1])
            x_var = self.decoder_variance.forward(z[:, -1].unsqueeze(-1)).exp()  
            bp_var = bp_var + x_var
            rp_var = rp_var + x_var
        else:
            bp_rec, rp_rec = self.decoder.forward(z)
        with torch.no_grad():
            bp_var.clamp_(min=eps)
            rp_var.clamp_(min=eps)
        bp_loss = 0.5*(bp_norm - bp_rec).pow(2)/bp_var + 0.5*bp_var.log()
        rp_loss = 0.5*(rp_norm - rp_rec).pow(2)/rp_var + 0.5*rp_var.log()
        return (bp_loss.sum(dim=-1) + rp_loss.sum(dim=-1))

    def criterion(self, xp_norm): # Assumes that xp has already been preprocessed
        dim_data = 2*xp_norm.shape[-1]
        z_mu, z_logsigma = self.infer_latent(xp_norm)
        z = z_mu
        loss2 = torch.zeros(1).sum()
        if self.vae:
            loss2 = KL_regularization(z_mu, z_logsigma).mean()
            z = z + z_logsigma.exp()*torch.randn_like(z_logsigma, requires_grad=False)
        loss1 = self.reconstruction_loss(z, xp_norm).mean(dim=0)/dim_data
        return {'NLL': loss1, 'KL': loss2}
