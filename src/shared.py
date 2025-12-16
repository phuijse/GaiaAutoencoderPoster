import torch
from torch import nn

def KL_regularization(z_mu, z_logsigma):
    return 0.5*(z_mu.pow(2) + (2*z_logsigma).exp() - 1. - 2.*z_logsigma).sum(dim=-1)      

class FullyConnectedBlock(nn.Module):

    def __init__(self, dim, activation, use_layernorm):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = None
        if use_layernorm:
            self.norm = nn.LayerNorm(dim)
        self.act = activation

    def forward(self, x):
        out = self.fc(x)
        if self.norm is not None:
            out = self.norm(out)
        return self.act(out)

class ResidualBlock(nn.Module):
    
    def __init__(self, dim, activation, use_layernorm):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1, self.norm2 = None, None
        if use_layernorm:
            self.norm1 = nn.LayerNorm(dim)
        self.act = activation
        self.fc2 = nn.Linear(dim, dim)
        if use_layernorm:
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act(out)
        out = self.fc2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.act(out + residual)

class FullyConnectedNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers, use_layernorm, activation, use_residual=False):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation)
        for _ in range(n_hidden_layers):
            if use_residual:
                hidden_layer = ResidualBlock(dim=hidden_dim, use_layernorm=use_layernorm, activation=activation)
            else:
                hidden_layer = FullyConnectedBlock(dim=hidden_dim, use_layernorm=use_layernorm, activation=activation)
            layers.append(hidden_layer)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.nnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.nnet(x)


class ConvolutionalNet(nn.Module):
    def __init__(self, output_dim, filters, kernels, activation, dropout_pbb):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels, k in zip(filters, kernels):
            layers.append(nn.Conv1d(in_channels, out_channels, stride=1, kernel_size=k, padding=k//2)),
            layers.append(nn.BatchNorm1d(out_channels)),
            layers.append(activation)
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.LazyLinear(output_dim), activation)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EarlyStopping:
    def __init__(self, save_model_path, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = torch.inf
        self.best_epoch = 0
        self.early_stop = False
        self.path = save_model_path

    def __call__(self, epoch, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)

class ResidualVAE(nn.Module):

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
        z_mu, z_logsigma = self.forward(mus)
        loss_reg = torch.zeros(1).sum()
        z = z_mu
        if self.vae:
            loss_reg = KL_regularization(z_mu, z_logsigma).mean()
            z = z + z_logsigma.exp()*torch.randn_like(z, requires_grad=False)
        mu_rec = self.decoder(z)        
        if self.use_errorbars:
            var = sigmas.pow(2)
            with torch.no_grad():
                var.clamp_(min=eps)
            loss_rec = (0.5*(mus - mu_rec).pow(2)/var + 0.5*var.log()).mean()
        else:
            loss_rec = 0.5*(mus - mu_rec).pow(2).mean()
        return {'NLL': loss_rec, 'KL': loss_reg}