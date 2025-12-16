import torch
from torch import nn, Tensor

from shared import ResidualBlock, KL_regularization


class SimpleMMAE(nn.Module):

    def __init__(self, n_input, n_latent, hidden_dim=128, activation=nn.GELU(), vae=True, use_layernorm=False, use_errorbars=True):
        super().__init__()
        self.n_latent = n_latent
        self.vae = vae
        self.use_errorbars = use_errorbars
        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            activation,
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            nn.Linear(hidden_dim, n_latent*2 if vae else n_latent)
        )
            
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            activation,
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            nn.Linear(hidden_dim, n_input)
        )

    def forward(self, x):
        z = self.encoder(x)
        if z.shape[1] == self.n_latent:
            return z, None
        return torch.split(z, self.n_latent, dim=-1)

    def criterion(self, mus, sigmas, eps=1e-6):
        var = sigmas.pow(2)
        z_mu, z_logsigma = self.forward(mus)
        loss_reg = torch.zeros(1).sum()
        z = z_mu
        if self.vae:
            loss_reg = KL_regularization(z_mu, z_logsigma).mean()
            z = z + z_logsigma.exp()*torch.randn_like(z, requires_grad=False)
        mu_rec = self.decoder(z)
        with torch.no_grad():
            var.clamp_(min=eps)
        if self.use_errorbars:
            loss_rec = (0.5*(mus - mu_rec).pow(2)/var + 0.5*var.log()).mean()
        else:
            loss_rec = 0.5*(mus - mu_rec).pow(2).mean()
        return {'NLL': loss_rec, 'KL': loss_reg}


class FullMMAE(nn.Module):

    def __init__(self, single_modality_models, 
                 n_input, n_latent, hidden_dim=128, activation=nn.GELU(), vae=True, use_layernorm=False, freeze_decoders=True):
        super().__init__()
        self.n_latent = n_latent
        self.vae = vae
        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            activation,
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            nn.Linear(hidden_dim, n_latent*2 if vae else n_latent)
        )
            
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            activation,
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            ResidualBlock(dim=hidden_dim, activation=activation, use_layernorm=use_layernorm),
            nn.Linear(hidden_dim, n_input)
        )
        self.single_models = []
        for model in single_modality_models:
            model.eval()
            if freeze_decoders:
                for param in model.parameters():
                    param.requires_grad = False
            self.single_models.append(model)

    def forward(self, x):
        z = self.encoder(x)
        if z.shape[1] == self.n_latent:
            return z, None
        return torch.split(z, self.n_latent, dim=-1)

    def forward_full(self, targets):
        mu1, _ = self.single_models[0].infer_latent(*targets[0])
        mu2, _ = self.single_models[1].infer_latent(*targets[1])
        mu3, _ = self.single_models[2].infer_latent(*targets[2])
        mus = torch.cat([mu1, mu2, mu3], dim=-1)
        return self.forward(mus)

    def criterion(self, targets, eps=1e-6):
        z_mu, z_logsigma = self.forward_full(targets)
        loss_reg = torch.zeros(1).sum()
        z = z_mu
        if self.vae:
            loss_reg = KL_regularization(z_mu, z_logsigma).mean()
            z = z + z_logsigma.exp()*torch.randn_like(z, requires_grad=False)
        mu_rec = self.decoder(z)
        data_dim1 = targets[0][0].shape[-1]*2
        data_dim2 = targets[1][-1].sum(dim=-1)
        data_dim3 = targets[2][-1].sum(dim=[-2, -1])
        rec1 = self.single_models[0].reconstruction_loss(mu_rec[:, 0*self.n_latent:1*self.n_latent], *targets[0])/data_dim1
        rec2 = self.single_models[1].reconstruction_loss(mu_rec[:, 1*self.n_latent:2*self.n_latent], *targets[1])/data_dim2
        rec3 = self.single_models[2].reconstruction_loss(mu_rec[:, 2*self.n_latent:3*self.n_latent], *targets[2])/data_dim3      
        return {'L1': rec1.mean(), 'L2': rec2.mean(), 'L3': rec3.mean(), 'KL': loss_reg}