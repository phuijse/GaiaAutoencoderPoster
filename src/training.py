import torch
from torch import nn
from pathlib import Path
from shared import ResidualVAE
from dmdt import DMDTAutoencoder
from xp import XPMeanSpectraAutoencoder
from rvs import RVSMeanSpectraAutoencoder
from folding import FoldedAutoencoder, weighted_min_max_scaling

def filter_config_with_fn_args(config, fn):
    import inspect
    sig = inspect.signature(fn)
    param_names = [name for name in sig.parameters]
    return {k: v for k, v in config.items() if k in param_names}

xp_preprocessor_kwargs = {'clip_left': 0, 'clip_right': 0, 'num_wl': 55}

lc_cleaner_args = {
    'relative_error_threshold' : 3., 
    'magbin_error_threshold': 10.,
    'drop_singleton_outliers': True
}

lc_folder_args = {
    'outlier_window': 5,
    'outlier_threshold': 3.5,
    'scaler': weighted_min_max_scaling
}

dmdt_binner_args = {
    'frequency_bin_edges': torch.tensor([4e-4, 1e-3, 1/500, 1/200, 1e-2, 1/50, 1/20, 0.5, 3, 8, 25]),
    'magnitude_bin_edges': torch.cat((torch.linspace(0.0, 3.0, 20), torch.tensor([float("inf")])))
}

def instance_unimodal_model(model_prefix, beta, n_latent, seed, use_radam, lr, batch_size, model_decoder_variance, load_weights=True, model_dir=Path('models')):
    model_name = f"AE_{model_prefix}_K{n_latent}_beta{beta}_decvar{model_decoder_variance}_seed{seed}_radam{use_radam}_lr{lr}_bs{batch_size}.ckpt"
    match model_prefix:
        case 'xp':
            model = XPMeanSpectraAutoencoder(
                n_latent=n_latent, n_hidden=64, 
                vae=beta>0.0, model_decoder_variance=model_decoder_variance, 
                use_layernorm=False, preprocessor_kwargs=xp_preprocessor_kwargs)
        case 'fold':
            model = FoldedAutoencoder(
                n_harmonics=12, n_hidden=128, activation=nn.GELU(), n_latent=n_latent, context='attention', 
                vae=beta>0.0, model_decoder_variance=model_decoder_variance, use_layernorm=False, preprocessor_kwargs=lc_folder_args
            )
        case 'rvs':
            model = RVSMeanSpectraAutoencoder(
                n_latent=n_latent, n_hidden=128,
                vae=beta>0.0, model_decoder_variance=model_decoder_variance, preprocessor_kwargs=rvs_preprocessor_kwargs)
        case 'dmdt':
            model = DMDTAutoencoder(n_hidden=128, activation=nn.GELU(), n_latent=n_latent, 
                                    vae=beta>0.0, model_decoder_variance=model_decoder_variance, use_layer_norm=False, preprocessor_kwargs=dmdt_binner_args)
    model_path = model_dir / model_name
    if load_weights:
        if not model_path.exists():
            raise ValueError('The model you are requesting to load does not exist')
        print(f"Model {model_name} found. Loading it")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    # model.to('cuda')
    return model, model_name

def instance_simple_multimodal_model(n_input, beta, n_latent, seed, use_radam, lr, batch_size, load_weights=True, use_errorbars=True, model_dir=Path('models')):
    model_name = f"SimpleMMAE_K{n_latent}_beta{beta}_seed{seed}_radam{use_radam}_lr{lr}_bs{batch_size}_err{use_errorbars}.ckpt"
    model = ResidualVAE(n_input=n_input, n_latent=n_latent, vae=beta>0.0, use_errorbars=use_errorbars)
    model_path = model_dir / model_name
    if load_weights:
        if not model_path.exists():
            raise ValueError("You asked for model but there is none")
        print(f"{model_name} model found. Loading it")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    # model.to('cuda')
    return model, model_name

def instantiate_pumap_model(n_input, hidden_dim, batch_size, lr, seed, use_radam, use_layernorm, load_weights=False, model_dir=Path('models')):
    # n_input = umap_dataset.shape[-1]
    model_name = f'MMAE_pumap_K{n_input}_hd{hidden_dim}_seed{seed}_radam{use_radam}_lr{lr}_bs{batch_size}.ckpt'
    model = ResidualVAE(n_latent=2, n_input=n_input, hidden_dim=hidden_dim, activation=nn.GELU(), vae=False, use_errorbars=False, use_layernorm=False)
    model_path = model_dir / model_name
    if load_weights:
        if not model_path.exists():
            raise ValueError("You asked for model but there is none")
        print(f"{model_name} model found. Loading it")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    # model.to('cuda')
    return model, model_name

def instance_viz_model(n_input, hidden_dim, beta, use_errorbars, use_radam, lr, batch_size, seed, load_weights=False, model_dir=Path('models')):
    model_name = f'MMAE_viz_K{n_input}_beta{beta}_hd{hidden_dim}_evars{use_errorbars}_seed{seed}_radam{use_radam}_lr{lr}_bs{batch_size}.ckpt'
    model = ResidualVAE(n_latent=2, n_input=n_input, vae=beta>0.0, hidden_dim=hidden_dim, use_errorbars=use_errorbars, activation=nn.GELU()) 
    model_path = model_dir / model_name
    if load_weights:
        if not model_path.exists():
            raise ValueError("You asked for model but there is none")
        print(f"{model_name} model found. Loading it")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    # model.to('cuda');
    return model, model_name