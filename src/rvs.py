import torch
from torch import nn, Tensor

from shared import FullyConnectedNet, ConvolutionalNet, KL_regularization
from stats import nanstd, masked_mean, masked_std


class RVSMeanSpectraPreprocessor(nn.Module):

    def __init__(self, clip_left, clip_right):
        super().__init__()
        self.clip_left = clip_left
        self.clip_right = -clip_right if clip_right > 0 else None
        self.width = 961 - clip_left - clip_right
        self.register_buffer('rvs_wl', torch.linspace(846, 870, 961)[self.clip_left:self.clip_right])

    def forward(self, rvs_spectra: Tensor):
        # Clip
        rvs_spectra = rvs_spectra[:, :, self.clip_left:self.clip_right]
        rvs_flux, rvs_ferr = rvs_spectra.unbind(dim=1)
        # Flag nans and positive outliers
        non_nan = ~torch.isnan(rvs_flux) & ~torch.isnan(rvs_ferr)
        non_outliers = rvs_flux < 1.0 + 4*nanstd(rvs_flux, dim=1, keepdims=True)
        good_mask = non_nan & non_outliers
        rvs_flux[~good_mask] = 0.0
        rvs_ferr[~good_mask] = 1.0
        # Remove linear trend?    
        """
        X = wl.unsqueeze(1).repeat((1, 2))
        X[:, -1] = 1.0
        XtX_inv = torch.linalg.inv(X.T @ X)
        X_pinv = XtX_inv @ X.T
        coeffs = (X_pinv @ rvs.T).T
        trend = coeffs[:, 0:1] * wl + coeffs[:, 1:2]
        rvs = rvs - trend 
        """
        # Rescale
        loc = masked_mean(rvs_flux, good_mask)
        scale = masked_std(rvs_flux, loc, good_mask)
        rvs_flux = (rvs_flux - loc) / scale
        rvs_ferr = rvs_ferr / scale
        rvs_spectra = torch.concat([rvs_flux.unsqueeze(1), rvs_ferr.unsqueeze(1)], dim=1)
        return rvs_spectra, good_mask


class RVSMeanSpectraConvEncoder(nn.Module):

    def __init__(self, n_latent, n_hidden, split_latent, filters=[8, 16, 32, 64], kernel_size=5, stride=2):
        super().__init__()
        self.n_latent = n_latent
        layers = []
        in_channels = 1
        for out_channels in filters:
            layers.append(nn.Conv1d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=2))
            layers.append(nn.BatchNorm1d(out_channels)),
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.LazyLinear(n_hidden),
            #nn.LayerNorm(n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 2*n_latent if split_latent else n_latent)
        )        
        
    def forward(self, rvs_norm):
        x = self.conv(rvs_norm.unsqueeze(1))
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        z = self.fc(x) 
        if z.shape[1] == self.n_latent:
            return z, None
        return z.split(self.n_latent, dim=-1)

class RVSMeanSpectraDecoder(nn.Module):

    def __init__(self, n_input, n_latent, n_hidden, activation=nn.GELU()):
        super().__init__()
        self.n_input = n_input
        self.nnet = FullyConnectedNet(
            input_dim=n_latent, hidden_dim=n_hidden, output_dim=n_input,
            n_hidden_layers=2, use_layernorm=False, activation=activation, use_residual=True
        )

    def forward(self, z):
        return self.nnet(z)

class RVSMeanSpectraAutoencoder(nn.Module):

    def __init__(self, n_latent, n_hidden=128, vae=False, model_decoder_variance=False, preprocessor_kwargs={}):
        super().__init__()
        self.vae = vae
        self.preprocessor = RVSMeanSpectraPreprocessor(**preprocessor_kwargs)
        #self.encoder = RVSEncoder(n_input=self.adapter.width, n_latent=n_latent, n_hidden=n_hidden, split_latent=vae)
        self.encoder = RVSMeanSpectraConvEncoder(n_latent=n_latent, n_hidden=n_hidden, split_latent=vae)
        n_dec_input = n_latent - 1 if model_decoder_variance else n_latent        
        self.decoder = RVSMeanSpectraDecoder(n_input=self.preprocessor.width, n_latent=n_dec_input, n_hidden=n_hidden)        
        self.decoder_variance = None
        if model_decoder_variance:
            self.decoder_variance = FullyConnectedNet(
                input_dim=1, hidden_dim=n_hidden, output_dim=1, 
                n_hidden_layers=1, activation=nn.GELU(), use_layernorm=False, use_residual=False,
            )
        
    def reconstruction_loss(self, z, rvs_norm, rvs_valid_mask, eps=1e-6):
        rvs_flux_norm, rvs_ferr_norm = rvs_norm.unbind(dim=1)
        rvs_var = rvs_ferr_norm.pow(2)
        if self.decoder_variance is not None:
            x_var = self.decoder_variance.forward(z[:, -1].unsqueeze(-1)).exp()
            rvs_var = rvs_var + x_var
            rvs_rec = self.decoder.forward(z[:, :-1])
        else:
            rvs_rec = self.decoder.forward(z)
        with torch.no_grad():
            rvs_var.clamp_(min=eps)
        rvs_loss = 0.5*(rvs_flux_norm - rvs_rec).pow(2)/rvs_var + 0.5*rvs_var.log()
        return (rvs_loss * rvs_valid_mask).sum(dim=-1)

    def infer_latent(self, rvs_norm, rvs_mask): # Assumes that rvs has already been preprocessed
        rvs_flux_norm = rvs_norm[:, 0, :]
        return self.encoder.forward(rvs_flux_norm)

    def forward(self, rvs):
        rvs_norm, rvs_mask = self.preprocessor(rvs)
        return self.infer_latent(rvs_norm, rvs_mask)

    def criterion(self, rvs_norm, rvs_mask): # Assumes that rvs has already been preprocessed 
        data_dim = rvs_norm.shape[-1]
        z_mu, z_logsigma = self.infer_latent(rvs_norm, rvs_mask)
        z = z_mu
        loss_reg = torch.zeros(1).sum()
        if self.vae:
            z = z + z_logsigma.exp()*torch.randn_like(z_logsigma, requires_grad=False)    
            loss_reg = KL_regularization(z_mu, z_logsigma).mean()
        loss_rec = self.reconstruction_loss(z, rvs_norm, rvs_mask).mean(dim=0)/data_dim
        return {'NLL': loss_rec, 'KL': loss_reg}

