import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import seaborn as sns
from . import common_utils

def update_ax(ax, title=None, xlabel=None, ylabel=None,
              legend_loc='best', legend_cols=1, despine=True,
              ticks_fs=10, label_fs=12, legend_fs=12, legend_title=None,
              title_fs=14, hide_xlabels=False, hide_ylabels=False,
              ticks=True, grid=True, grid_kws=None,
              legend_bbox=None, legend_title_fs=None):

    if title: ax.set_title(title, fontsize=title_fs)
    if xlabel: ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel: ax.set_ylabel(ylabel, fontsize=label_fs)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=legend_fs, \
                  ncol=legend_cols, title=legend_title, \
                  bbox_to_anchor=legend_bbox, title_fontsize=legend_title_fs)

    if despine: sns.despine(ax=ax)

    if ticks:
        ax.tick_params(direction='in', length=6, width=2, colors='k', which='major', top=False, right=False)
        ax.tick_params(direction='in', length=4, width=1, colors='k', which='minor', top=False, right=False)
        ax.tick_params(labelsize=ticks_fs)

    if hide_xlabels: ax.set_xticks([])
    if hide_ylabels: ax.set_yticks([])

    grid_kws = grid_kws or dict(ls=':', lw=2, alpha=0.5)
    if grid: ax.grid(True, **grid_kws)
    return ax


def plot_image_table(image_tensor, num_rows, titles=None, labels=None, ylabels=None, flagged=None,
                     add_image_index=True, image_size=2.5, title_fs=14, label_fs=14, flag_color='red', flag_lw=3):
    num_images = len(image_tensor)
    assert num_images % num_rows == 0
    num_cols = num_images // num_rows

    labels = ['']*num_images if labels is None else labels
    ylabels = ['']*num_images if ylabels is None else ylabels
    indices = [f'({r},{c}) ' for r in range(num_rows) for c in range(num_cols)]
    titles = ['']*num_images if titles is None else titles
    if add_image_index:
        titles = [f'{idx}{title}' for idx, title in zip(indices, titles)]
    flagged = [False]*num_images if flagged is None else flagged
    ylabels, labels, titles, flagged = map(np.array, [ylabels, labels, titles, flagged])

    S = splits = {}
    split_fn = lambda a: np.split(a, num_rows)
    _a = [image_tensor, titles, labels, ylabels, flagged]
    S['images'], S['titles'], S['labels'], S['ylabels'], S['flagged'] = map(split_fn, _a)

    figsize = (num_cols*image_size, num_rows*image_size)
    fig, ax_grid = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    if num_rows==1: ax_grid = [ax_grid]

    for idx, (r_images, r_titles, r_labels, r_ylabels, r_flagged) in enumerate(zip(*list(splits.values()))):
        ax_row = ax_grid[idx]
        plot_image_row(r_images, titles=r_titles, labels=r_labels, ylabels=r_ylabels,
                       flagged=r_flagged, axs=ax_row, img_height=image_size, img_width=image_size,
                       title_fs=title_fs, label_fs=label_fs, flag_color=flag_color, flag_lw=flag_lw)

    return fig, ax_grid

def plot_image_row(image_tensor, titles=None, labels=None, ylabels=None,
                   flagged=None, axs=None, img_height=3, img_width=3,
                   title_fs=16, label_fs=12, flag_color='red', flag_lw=3):

    def _add_axis_border(ax):
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(flag_color)
            sp.set_linewidth(flag_lw)

    ncols = len(image_tensor)
    figsize = (ncols*img_width, img_height)

    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
        if ncols==1: axs=[axs]
    else:
        fig = None
        assert len(axs)==len(image_tensor)

    labels = [None]*ncols if labels is None else labels
    ylabels = [None]*ncols if ylabels is None else ylabels
    titles = [None]*ncols if titles is None else titles
    flagged = [False]*ncols if flagged is None else flagged

    num_imgs = image_tensor.shape[0]
    img_size = image_tensor.shape[-1]

    grid = torchvision.utils.make_grid(image_tensor, normalize=True, padding=0,
                                       scale_each=True, nrow=num_imgs)
    grid = grid.permute(1,2,0)

    images = torch.split(grid, img_size, dim=1)

    for ax, img, title, label, ylabel, flag in zip(axs, images, titles, labels, ylabels, flagged):
        ax.imshow(img)
        update_ax(ax, title=title, xlabel=label, ylabel=ylabel, legend_loc=None,
                                hide_xlabels=True, hide_ylabels=True,
                                despine=False, label_fs=label_fs, title_fs=title_fs)

        if flag: _add_axis_border(ax)

    return fig, axs

def plot_grouped_images(cluster_index_map, dataset,
                        samples_per_cluster=10,
                        cluster_label_map=None,
                        random_sample=False,
                        **row_kw):
    """
    - clusters_index_map: [cluster key] -> [list of indices]
    - dataset
    - cluster_label_map [cluster key] -> [cluster name] (None)
    - samples per cluster: number of images to sample per cluster (10)
    - random sample: top-k images or randomly sample k (default is topk)
    """

    # setup
    num_clusters = len(cluster_index_map)
    ax_shape = (num_clusters, samples_per_cluster)
    kw = dict(img_height=3, img_width=3, title_fs=16, label_fs=16)
    kw.update(row_kw)

    # make figure
    figsize = (kw['img_width']*ax_shape[1], kw['img_height']*ax_shape[0])
    fig, ax_grid = plt.subplots(nrows=ax_shape[0], ncols=ax_shape[1], figsize=figsize)


    # plot
    for (cluster_index, all_indices), ax_row in zip(cluster_index_map.items(), ax_grid):
        if random_sample:
            indices = np.random.choice(all_indices, samples_per_cluster, replace=False)
        else:
            indices = all_indices[:samples_per_cluster]

        image_tensor = torch.stack([dataset[idx][0] for idx in indices])
        plot_image_row(image_tensor, titles=None, labels=None, axs=ax_row)

        label = label = f'Cluster #{cluster_index}'
        if cluster_label_map:
            label = f'{cluster_label_map[cluster_index]}'
        ax_row[0].set_ylabel(label, fontweight='bold', fontsize=14)

    return fig, ax_grid

def plot_image_grid(img_tensor, num_images=None, num_per_row=None, shuffle=False,
                    normalize=True, scale_each=False, ax=None, pad_value=0, padding=0, imgsize=2):
    """
    torchvision.utils.make_grid wrapper
    - image tensor: torch image tensor
    - num_images: number of images to sample from tensor (default: all)
    - num_per_row: number of images per row (default: num_images)
    - shuffle: shuffle order of images (default: False)
    - make_grid args: normalize, scale_each, pad_value
    - ax: matplotlib axis
    """
    if type(img_tensor) is np.ndarray:
        img_tensor = torch.from_numpy(img_tensor)

    img_tensor = img_tensor.clone().cpu().float()

    if len(img_tensor.shape)==3: img_tensor = img_tensor.unsqueeze(0)

    num_images = num_images if num_images else len(img_tensor)
    num_per_row = num_per_row if num_per_row else num_images
    fig = None

    if ax is None:
        figsize = (imgsize*num_images//num_per_row, imgsize*num_per_row)
        figsize = (imgsize*num_per_row, imgsize*num_images//num_per_row)
        fig, ax = plt.subplots(1,1,figsize=figsize)

    if shuffle:
        s = np.random.choice(len(img_tensor), size=num_images, replace=False)
        img_tensor = img_tensor[s]
    else:
        img_tensor = img_tensor[:num_images]

    # img_tensor = torch.FloatTensor(img_tensor)
    g = torchvision.utils.make_grid(img_tensor, nrow=num_per_row,
                                    normalize=normalize, scale_each=scale_each,
                                    pad_value=pad_value, padding=padding)
    g = g.permute(1,2,0).numpy()
    ax.imshow(g)
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def add_axis_border(ax, color, lw):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(color)
        sp.set_linewidth(lw)