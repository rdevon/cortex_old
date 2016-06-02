'''Module saving functional network connectivity.

'''

import igraph
import matplotlib
from matplotlib import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


conf = igraph.Configuration()
conf['general.verbose'] = False


def plot(fnc, idx=None, groups=None, transform=None, labels=None, out_file=None):
    if groups is None:
        groups = [range(fnc.shape[0])]

    if idx is None:
        idx = range(fnc.shape[0])

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
        y = [g - 0.45, g - 0.45]
        plt.plot(x, y, linewidth =2, color='black', ls='-')
    for g in grid:
        x = [g - 0.4, g - 0.4]
        y = [-10, n_components + 10]
        plt.plot(x, y, linewidth =2, color='black', ls='-')

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

    plt.savefig(out_file)
    plt.close()

def group(mat, thr=0.3, idx=None, labels=None, out_file=None):
    max_weight = mat.max()
    thr = thr * max_weight

    if idx is None:
        idx = range(mat.shape[0])

    wheres = np.where(mat > thr)

    edgelist = []
    weights = []

    for x, y in zip(wheres[0], wheres[1]):
        if x < y:
            edgelist.append((x, y))
            weights.append((mat[x, y]))

    if len(weights) > 0:
        weights /= np.std(weights)
    else:
        return [[i] for i in idx]

    g = igraph.Graph(edgelist, directed=False)
    g.vs['label'] = [i if labels is None else labels[i] for i in idx]
    cls = g.community_multilevel(return_levels=True, weights=weights)
    cl = list(cls[0])

    clusters = [[idx[j] for j in i] for i in cl]

    colors = ['blue', 'red', 'green', 'black', 'yellow', 'magenta', 'cyan',
              'white', 'brown', 'tan', 'grey', 'purple']
    for i, c in enumerate(cl):
        g.vs(cl[i])['color'] = colors[i % len(colors)]

    if out_file is not None:
        igraph.plot(g, out_file, weights=weights, edge_width=weights/10,
               vertex_label_size=8)
    return clusters