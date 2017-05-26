'''Module saving functional network connectivity.

'''

import igraph
import matplotlib
from matplotlib import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_fnc(fnc, idx=None, groups=None, transform=None, labels=None,
             out_file=None):
    if groups is None:
        groups = [range(fnc.shape[0])]

    if idx is None:
        idx = [i for idx in groups for i in idx]
    if labels is not None:
        labels = [labels[i] for i in idx]
    else:
        labels = None
    fnc = fnc[idx][:, idx]
    fnc = fnc * (1.0 - np.eye(fnc.shape[0]))
    n_components = len(idx)

    font = matplotlib.font_manager.FontProperties()
    font.set_family('sans-serif')
    font.set_size(8)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=1.0)

    grid = [0]
    for g in groups:
        grid.append(grid[-1] + len(g))
    for g in grid:
        x = [-10, n_components + 10]
        y = [g - 0.4, g - 0.4]
        plt.plot(x, y, linewidth =2, color='black', ls='-')
    for g in grid:
        x = [g - 0.45, g - 0.45]
        y = [-10, n_components + 10]
        plt.plot(x, y, linewidth =2, color='black', ls='-')
    for i, j in enumerate(idx):
        plt.text(i - 0.45, i + 0.4, '%d' % j, fontsize=6)

    cv = fnc.copy()
    if transform is None:
        transform = lambda x: x
    tv = np.vectorize(transform)
    cv = tv(cv)

    vmax = float(max(abs(np.amin(cv)), abs(np.amax(cv))))
    imgplot = ax.imshow(cv, interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax.axis((-0.5, n_components - 0.5, n_components - 0.5, -0.5))

    if labels is not None:
        ax.set_xlim(-0.5, n_components - 0.5)
        ax.set_xticks(range(n_components))
        ax.set_yticks(range(n_components))
        xtick_names = plt.setp(ax, xticklabels=labels)
        plt.setp(xtick_names, rotation=90)
        ytick_names = plt.setp(ax, yticklabels=labels)
        plt.setp(ytick_names, rotation=0)
    else:
        plt.axis('off')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.4)
    c_ticks = (-vmax, -vmax / 2.0, 0, vmax / 2.0, vmax )
    c_ticks_str = ['%.2f' % t for t in c_ticks]
    cbar = plt.colorbar(imgplot, ticks=c_ticks, orientation='horizontal', cax=cax)
    cbar.ax.set_xticklabels(c_ticks_str, fontproperties=font)

    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    plt.close()