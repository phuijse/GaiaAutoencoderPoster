import polars as pl
import json
from pathlib import Path
import numpy as np
import param
from scipy.spatial import cKDTree
import holoviews as hv
from functools import partial
from gaiaxpy import pwl_to_wl
import torch
import panel as pn

import text_blocks as tb
from training import instance_unimodal_model

pretty_labels = {
    'Semi regular': ["SR", "SRA", "SRD", "SRS", "SRB", "SARG"], 
    'Mira variable': ["M"], 
    'Rotational variable': ["SOLAR_LIKE", "ROT", "BY", "RS",], 
    'Eclipsing binary (Algol)': ["EA"],
    'Eclipsing binary (β Lyrae)': ["EB"], 
    'Eclipsing binary (W UMa)': ["EW"],
    'δ Scuti': ["DSCT"], 
    'RR Lyrae (Type ab)': ["RRAB"], 
    'RR Lyrae (Type c)': ["RRC"], 
    'Cepheid (Classical)': ["DCEP"], 
    'Chemically peculiar': ["CP", "MCP", "ACV"], 
    'Young stellar object': ["YSO"], 
    'Cepheid (Type II)': ["T2CEP", "RV", "BLHER"], 
    'Active galactic nuclei': ["AGN", "QSO", "BLAZAR"], 
    'Ellipsoidal variable': ["ELL"],
    #'beta Cephei': ["BCEP"],
}


pretty_labels_inv = {
    v: k
    for k, values in pretty_labels.items()
    for v in values
}


def build_dashboard():
    hv.extension('bokeh')
    pn.extension(design='material', sizing_mode="stretch_width")
    models_save_dir = Path('data')
    with open(models_save_dir / 'xp_model_config.json', 'r') as f:
        xp_training_configuration = json.load(f)

    with open(models_save_dir / 'fold_model_config.json', 'r') as f:
        fold_training_configuration = json.load(f)
        
    model_xp, _ = instance_unimodal_model('xp', **xp_training_configuration, model_dir=models_save_dir)
    model_xp.eval();

    rec_phase = torch.linspace(-1, 1, 200)
    model_fold, _ = instance_unimodal_model('fold', **fold_training_configuration, model_dir=models_save_dir)
    rec_phase_emb = model_fold.phase_embedder(rec_phase.unsqueeze(0))
    model_fold.eval();

    df_data = pl.read_parquet(models_save_dir / 'poster_data*.parquet')
    df_latent = pl.read_parquet(models_save_dir / 'latent_vars_plus_metadata_pumap.parquet')
    df_latent = df_latent.join(df_data.select(['sourceid', 'color']), on='sourceid').with_columns(
        pl.col('frequency').log10().alias('log_frequency'),
        pl.col('magnitude_std').log10().alias('log_amplitude'),
    )

    labeled_subset = df_latent.filter(
        pl.col('label').is_in(pretty_labels_inv.keys())
    ).with_columns(
        pl.col('label').replace(pretty_labels_inv)
    ).group_by('label').map_groups(lambda g: g.sample(n=min(500, g.height), seed=0)).sort('label')

    wl_bp = pwl_to_wl('BP', np.arange(0, 55))
    wl_rp = pwl_to_wl('RP', np.arange(0, 55))

    xname, yname = 'lambda_1_umap', 'lambda_2_umap'

    latent_space = hv.Points(df_latent.select([xname, yname]).to_numpy(), kdims=['λ 1', 'λ 2']).opts(color='k', width=900, height=600, size=1, alpha=0.1)
    latent_space_labeled = hv.Points(hv.Dataset(labeled_subset.to_pandas()), kdims=[xname, yname], vdims=['label']).opts(width=900, height=600, color="label", cmap="glasbey", size=1, legend_position="right")
    pointer = hv.streams.PointerXY(source=latent_space, x=0.0, y=0.0)
    show_labeled = pn.widgets.Checkbox(name="Show labeled overlay", value=True)

    def view(show):
        return latent_space_labeled.opts(alpha=1.0 if show else 0.0, show_legend=bool(show))

    coords = df_latent[[xname, yname]].to_numpy()
    tree = cKDTree(coords)

    class NearestSelector(param.Parameterized):
        idx = param.Integer(default=-1)
        def __init__(self, tree, **params):
            super().__init__(**params)
            self._tree = tree

        def update_from_xy(self, x, y):
            if x is None or y is None or np.isnan(x) or np.isnan(y):
                self.idx = 0
                return
            _, i = self._tree.query([x, y], k=1)
            self.idx = int(i)

    selector = NearestSelector(tree=tree)
    pointer.add_subscriber(lambda x, y, **_: selector.update_from_xy(x, y))

    def plot_mean_xp(idx):
        plot_data = df_data.slice(idx, 1)
        bpflux, rpflux, bperr, rperr = plot_data.select(['bpflux', 'rpflux', 'bpfluxerror', 'rpfluxerror']).explode('*').to_numpy().T    
        bp_plot = hv.Curve((wl_bp, bpflux), kdims='Wavelength [nm]', vdims='Flux').opts(framewise=True)
        rp_plot = hv.Curve((wl_rp, rpflux), kdims='Wavelength [nm]', vdims='Flux')
        return hv.Overlay([bp_plot, rp_plot])
        
    def plot_raw_lc(idx):
        plot_data = df_data.slice(idx, 1)
        sid = plot_data['sourceid'].item()
        time, mag, err = plot_data.select(['obstimes', 'val', 'valerr']).explode('*').to_numpy().T    
        return hv.Scatter((time, mag), 'Time [days]', 'Magnitude')
        
    def plot_folded_lc(idx):
        plot_data = df_data.slice(idx, 1)
        freq = plot_data['frequency'].item()
        time, mag, err = plot_data.select(['obstimes', 'val', 'valerr']).explode('*').to_numpy().T    
        phase = np.mod(time, 2/freq)*freq
        return hv.Scatter((phase, mag), 'Phase', 'Magnitude')

    def create_sliders_xp(width=None):
        mins, meds, maxs = np.percentile(df_latent.select([f'latent_mu_{k}' for k in range(5)]).to_numpy(), [1, 50, 99], axis=0)
        return [pn.widgets.FloatSlider(name=f'z{k}', start=mins[k], end=maxs[k], value=meds[k], width=width) for k in range(5)]

    def create_sliders_fold(width=None):
        mins, meds, maxs = np.percentile(df_latent.select([f'latent_mu_{k}' for k in range(5, 10)]).to_numpy(), [1, 50, 99], axis=0)
        return [pn.widgets.FloatSlider(name=f'z{k}', start=mins[k], end=maxs[k], value=meds[k], width=width) for k in range(5)]

    def recon_xp(width=250, height=180, **kwargs):
        zz_ = torch.tensor([v for v in kwargs.values()]).to(torch.float32)
        with torch.no_grad():
            bp_rec, rp_rec = model_xp.decoder(zz_)
        wl_bp = pwl_to_wl('BP', np.arange(2, 55-3))
        wl_rp = pwl_to_wl('RP', np.arange(2, 55-3))
        curve_bp = hv.Curve((wl_bp, bp_rec.numpy()), kdims=['Wavelength [nm]'], vdims=['Normalized flux'])
        curve_rp = hv.Curve((wl_rp, rp_rec.numpy()), kdims=['Wavelength [nm]'], vdims=['Normalized flux'])
        return (curve_bp*curve_rp).opts(framewise=True, shared_axes=True, ylim=(0, 1.2), invert_yaxis=False, width=width, height=height, toolbar=None)

    def recon_fold(width=250, height=180, **kwargs):
        zz_ = torch.tensor([v for v in kwargs.values()]).to(torch.float32).unsqueeze(0)
        with torch.no_grad():
            folded_rec = model_fold.decoder.forward(rec_phase_emb, zz_[:, :-1])
            folded_rec_std = model_fold.decoder_variance.forward(zz_[:, -1]).exp().sqrt()
        rec_plot =  hv.Curve((rec_phase.numpy(), folded_rec[0].cpu().numpy()), 'Phase ', 'Norm. magnitude') 
        rec_plot = rec_plot * hv.Spread((rec_phase.numpy(), folded_rec[0].cpu().numpy(), folded_rec_std.cpu())).opts(alpha=0.25)
        return rec_plot.opts(framewise=True, shared_axes=True, ylim=(-0.2, 1.2), invert_yaxis=True, width=width, height=height, toolbar=None)

    def preset_buttons(sliders, presets, *, width=250):
        buttons = []
        for label, values in presets.items():
            b = pn.widgets.Button(name=label, button_type="primary", width=width)
            def _make_cb(vals):
                def _cb(event):
                    for s, v in zip(sliders, vals):
                        s.value = float(v)
                return _cb

            b.on_click(_make_cb(values))
            buttons.append(b)
        return pn.Row(*buttons)

    sliders_xp = create_sliders_xp(width=300)
    sliders_fold = create_sliders_fold(width=300)
    button_conf_xp = {}
    button_conf_fold = {}
    for label in ['RR Lyrae (Type ab)', 'Eclipsing binary (β Lyrae)', 'Mira variable']:
        subset = labeled_subset.filter(pl.col('label').eq(label))
        button_conf_xp[label] = np.median(subset.select([f'latent_mu_{k}' for k in range(5)]).to_numpy(), axis=0).tolist()
        button_conf_fold[label] = np.median(subset.select([f'latent_mu_{k}' for k in range(5, 10)]).to_numpy(), axis=0).tolist()

    buttons_xp = preset_buttons(sliders_xp, button_conf_xp)
    buttons_fold = preset_buttons(sliders_fold, button_conf_fold)
    def interactive_reconstruction(func, sliders, buttons, width, height):
        controls = {f'z{k}': s for k, s in enumerate(sliders)}
        bound = pn.bind(partial(func, width=width, height=height), **controls)
        return pn.Column(buttons, pn.Row(pn.Column(*sliders), bound))

    latent_view = hv.DynamicMap(pn.bind(view, show=show_labeled))
    xp_dmap = hv.DynamicMap(pn.bind(plot_mean_xp, idx=selector.param.idx))
    lc_raw_dmap = hv.DynamicMap(pn.bind(plot_raw_lc, idx=selector.param.idx))
    lc_folded_dmap = hv.DynamicMap(pn.bind(plot_folded_lc, idx=selector.param.idx))

    custom_md = partial(pn.pane.Markdown, styles={'font-size': '1.2em'})

    encoding_panel = pn.Column(
        custom_md(f"## Results: Encoding *Gaia* time series and low-res spectra\n{tb.encoding_md}"),
        pn.Row(
            pn.Column(
                xp_dmap.opts(width=280, height=180).opts(framewise=True, toolbar=None, title='Low-res spectra'),
                lc_folded_dmap.opts(width=280, height=180).opts(color='k', framewise=True, invert_yaxis=True, xlim=(0, 2), toolbar=None, title='Folded light curve'),
                lc_raw_dmap.opts(width=280, height=180).opts(color='k', framewise=True, invert_yaxis=True, xlim=(1600, 2800), toolbar=None, title='Raw light curve'),
            ),
            pn.Column(latent_space.opts(width=800, height=180*3, toolbar='below') * latent_view, show_labeled),
        ),
    )

    decoding_panel = pn.Column(
        custom_md(f"## Results: Decoding from the learned representation\n{tb.decoding_md}"),
        interactive_reconstruction(recon_xp, sliders_xp, buttons_xp, width=350, height=230),
        #pn.Spacer(width=20),
        interactive_reconstruction(recon_fold, sliders_fold, buttons_fold, width=350, height=230),
    )    

    classification_panel = pn.Column(
        custom_md('## Results: Classifying variable sources using the learned representation\n'),
        pn.Row(
            custom_md(tb.classification_md),
            pn.pane.PNG('images/cm_lr.png', width=700, align="center"),
        )
    )

    dashboard = pn.Column(
        pn.Row(
            custom_md("# Organizing the Variable Sky with Multi-Instrument Gaia Data and Neural Networks\n ## Pablo Huijse & Joris De Ridder, Instituut voor Sterrenkunde, KU Leuven"),
            pn.pane.PNG('images/Gaia_logo.png', width=100, align="center"),
            pn.pane.PNG('images/4D-STAR_logo_old.png', width=100, align="center"),
            pn.pane.PNG('images/kuleuven.png', width=100, align="center"),
        ),
        pn.Row(
            pn.Column(
                #pn.pane.Markdown(f"## The variable sky\n {tb.tda_intro}"),
                custom_md(f"## The variable sky\n {tb.tda_intro}"),
                #pn.pane.PNG('images/cepheid.gif')
                pn.pane.Image('images/cepheid.gif', width=450, align="center"),
                pn.pane.Image('images/eclipsing_binary.gif', width=450, align="center"),
                pn.pane.Markdown("(Animations credit: [space.fm/astronomy](https://www.space.fm/astronomy))"),
            ),
            pn.Spacer(width=5),
            pn.Column(
                custom_md(f"## The *Gaia* mission\n {tb.gaia_intro}"),
                pn.pane.PNG('images/gaia_universe.png', width=400, align="center"),
                pn.pane.Markdown("(Images credit: ESA)"),
            ),        
            pn.Spacer(width=5),        
            pn.Column(
                custom_md(f"## Autoencoders\n {tb.ae_intro}"),
                pn.pane.PNG('images/AE.png', width=600, align="center")
            )
        ),
        pn.Row(
            pn.Tabs(
                ('Encoding', encoding_panel),
                ('Decoding', decoding_panel),
                ('Classification', classification_panel),
            ),        
            pn.Spacer(width=20),
            pn.Column(
                custom_md(f'## Conclusion and future work\n{tb.conclusions}'),
                pn.pane.PNG('images/paper_qr.png', width=180, align="center"),
                custom_md(f'## Selected references\n{tb.references}'),
            )
        ),
    )
    return dashboard


if __name__.startswith("bokeh"):
    dashboard = build_dashboard()
    dashboard.servable(title='Departamental research day')
