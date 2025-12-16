from matplotlib import pyplot as plt
import numpy as np

compress_name = lambda name : name if len(name) <= 10 else name[:10]+"..."

def plot_losses(losses, use_asinh: bool = False):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax_kl = ax.twinx()
    colors = ['b', 'k']
    for c, (key, val) in zip(colors, losses['train'].items()):
        if key == 'KL':
            ax_kl.plot(torch.stack(val).cpu(), label=f'train_{key}', c=c)
        else:
            ax.plot(torch.stack(val).cpu(), label=f'train_{key}', c=c)
    for c, (key, val) in zip(colors, losses['valid'].items()):
        if key == 'KL':
            ax_kl.plot(torch.stack(val).cpu(), label=f'valid_{key}', c=c, ls='--')
        else:
            ax.plot(torch.stack(val).cpu(), label=f'valid_{key}', c=c, ls='--')
    #ax.set_yscale('log')
    ax.set_ylabel('Negative log likelihood')
    ax_kl.set_ylabel('KL divergence')
    ax.set_xlabel('Epoch')
    ax.legend()
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    total_train_loss = torch.stack([torch.stack(val).cpu() for key, val in losses['train'].items()]).sum(dim=0).cpu()
    total_valid_loss = torch.stack([torch.stack(val).cpu() for key, val in losses['valid'].items()]).sum(dim=0).cpu()
    if use_asinh:
        ax.plot(torch.asinh(total_train_loss), label='train')
        ax.plot(torch.asinh(total_valid_loss), ls='--', label='validation')
        ax.set_ylabel('Total loss (asinh)')
    else:
        ax.plot(total_train_loss, label='train')
        ax.plot(total_valid_loss, ls='--', label='validation')
        ax.set_ylabel('Total loss')
    ax.set_xlabel('Epoch')
    

def two_dim_latent_space_with_labels(zs_mu2, labels, names, title=None, contour=None, delta=None, plot_bkg=False, s=1, legend_loc=3, dpi=120):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_xlabel('Lambda 1')
    ax.set_ylabel('Lambda 2')
    x_robust_min = []
    x_robust_max = []
    y_robust_min = []
    y_robust_max = []
    if plot_bkg:
        ax.scatter(zs_mu2[:, 0], zs_mu2[:, 1], s=1, alpha=0.01, c='k') 
    for name in names:
        if isinstance(name, str):
            mask = labels == name
            plot_label = compress_name(name)
        else:
            mask = np.any(np.stack([(labels == name_) for name_ in name]), axis=0)
            plot_label = compress_name(name[0])
        alpha = 0.5 if np.sum(mask) < 100000 else 0.1
        if np.sum(mask) == 0:
            continue
        #ax.errorbar(zs_mu[mask, 0], zs_mu[mask, 1], zs_sigma[mask, 0], zs_sigma[mask, 1], fmt='', linestyle='', alpha=alpha, label=plot_label)
        ax.scatter(zs_mu2[mask, 0], zs_mu2[mask, 1], s=s, label=plot_label, alpha=alpha) 
        #ax.scatter(zs_mu[mask, 0], zs_mu[mask, 1], s=s, label=plot_label, alpha=alpha) 
        if delta is not None:
            x_robust_min.append(np.percentile(zs_mu2[mask, 0], delta))
            x_robust_max.append(np.percentile(zs_mu2[mask, 0], 100-delta))
            y_robust_min.append(np.percentile(zs_mu2[mask, 1], delta))
            y_robust_max.append(np.percentile(zs_mu2[mask, 1], 100-delta))
    if contour is not None:
        X, Y, Z = contour
        ax.contour(X, Y, Z, levels=[0.0002], alpha=0.2, linewidths=2, colors='k', zorder=-10)
    if delta is not None:
        xlims = [np.amin(np.array(x_robust_min)), np.amax(np.array(x_robust_max))]
        ylims = [np.amin(np.array(y_robust_min)), np.amax(np.array(y_robust_max))]
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    else:
        xlims, ylims = None, None
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
        lh.set_sizes([10.0])
    if title is not None:
        ax.set_title(title)
    return xlims, ylims

def smart_format(x):
    if x == float("inf"):
        return "inf"
    elif x >= 10:
        return f"{x:.0f}"
    else:  
        return f"{x:.2f}"

def plot_dmdt(ax, dmdt, dt_edges, dm_edges):
    ax.pcolormesh(dmdt.T, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_xticks(range(len(dt_edges)))
    ax.set_yticks([0, 3, 6, 9, len(dm_edges)-10, len(dm_edges)-7, len(dm_edges)-4, len(dm_edges)-1])
    ax.set_yticklabels([f"{dm.item():0.2f}" for dm in dm_edges[[0, 3, 6, 9, -10, -7, -4, -1]]])
    ax.set_xticklabels([smart_format(dt) for dt in dt_edges], rotation=45);