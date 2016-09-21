'''
Module to view nifti data.
'''

import sys

import argparse
import itertools
import logging
from math import ceil
if not 'matplotlib' in sys.modules:
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib
from matplotlib.patches import FancyBboxPatch
from matplotlib import pylab as plt, rc
import multiprocessing as mp
import nilearn
from nilearn import plotting as nplt
import nipy
from nipy import load_image, save_image
from nipy.core.api import Image, xyz_affine
from nipy.labs.viz import plot_map
import numpy as np
import os
from os import path
import pickle
from progressbar import Bar, ProgressBar, Percentage, Timer
from scipy.signal import detrend

from ..utils import floatX


logger = logging.getLogger(__name__)
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.25, 0.2, 0.2),
                 (0.45, 0.0, 0.0),
                 (0.5, 0.5, 0.5),
                 (0.55, 0.0, 0.0),
                 (0.75, 0.8, 0.8),
                 (1.0,  1.0, 1.0)),
         'green': ((0.0, 0.0, 1.0),
                   (0.25, 0.0, 0.0),
                   (0.45, 0.0, 0.0),
                   (0.5, 0.5, 0.5),
                   (0.55, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0,  1.0, 1.0)),
         'blue':  ((0.0, 0.0, 1.0),
                   (0.25, 0.8, 0.8),
                   (0.45, 0.0, 0.0),
                   (0.5, 0.5, 0.5),
                   (0.55, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0,  0.0, 0.0)),}

cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

def save_image(nifti, anat, cluster_dict, out_path, f, image_threshold=2,
               texcol=1, bgcol=0, iscale=2, text=None, **kwargs):
    '''Saves a single nifti image.

    Args:
        nifti (str or nipy.core.api.image.image.Image): nifti file to visualize.
        anat (nipy.core.api.image.image.Image): anatomical nifti file.
        cluster_dict (dict): dictionary of clusters.
        f (int): index.
        image_threshold (float): treshold for `plot_map`.
        texcol (float): text color.
        bgcol (float): background color.
        iscale (float): image scale.
        text (Optional[str]): text for figure.
        **kwargs: extra keyword arguments

    '''
    if isinstance(nifti, str):
        nifti = load_image(nifti)
        feature = nifti.get_data()
    elif isinstance(nifti, nipy.core.image.image.Image):
        feature = nifti.get_data()
    font = {'size': 8}
    rc('font', **font)

    coords = cluster_dict['top_clust']['coords']
    if coords == None:
        return

    feature /= feature.std()
    imax = np.max(np.absolute(feature))
    imin = -imax
    imshow_args = dict(
        vmax=imax,
        vmin=imin,
        alpha=0.7
    )

    coords = ([-coords[0], -coords[1], coords[2]])

    plt.axis('off')
    plt.text(0.05, 0.8, text, horizontalalignment='center',
             color=(texcol, texcol, texcol))

    try:
        plot_map(feature, xyz_affine(nifti), anat=anat.get_data(),
                 anat_affine=xyz_affine(anat), threshold=image_threshold,
                 cut_coords=coords, annotate=False, cmap=cmap,
                 draw_cross=False, **imshow_args)
    except Exception as e:
        return

    plt.savefig(out_path, transparent=True, facecolor=(bgcol, bgcol, bgcol))

def save_helper(args):
    save_image(*args)

def save_images(nifti_files, anat, roi_dict, out_dir, **kwargs):
    '''Saves multiple nifti images using multiprocessing.

    Uses `multiprocessing`.

    Args:
        nifti_files (list): list of nifti file paths.
        anat (nipy.core.api.image.image.Image): anatomical image.
        roi_dict (dict): dictionary of cluster dictionaries.
        out_dir (str): output directory path.
        **kwargs: extra keyword arguments.

    '''
    p = mp.Pool(30)
    idx = [int(f.split('/')[-1].split('.')[0]) for f in nifti_files]
    args_iter = itertools.izip(nifti_files,
                               itertools.repeat(anat),
                               [roi_dict[i] for i in idx],
                               [path.join(out_dir, '%d.png' % i) for i in idx],
                               idx)

    p.map(save_helper, args_iter)
    p.close()
    p.join()

def montage(nifti, anat, roi_dict, thr=2, fig=None, out_file=None, order=None,
            stats=None, time_courses=None, y=8, global_std=None, clusters=None):
    '''Saves a montage of nifti images.

    Args:
        nifti (list or nipy.core.api.image.image.Image): 4d nifti or list of \
            3D niftis.
        anat (nipy.core.api.image.image.Image): anatomical nifti image.
        roi_dict (dict): dictionary of cluster dictionaries.
        out_file (str): output file path.
        order (list): List of integers. Order of montage.
        stats (Optional[dict]): extra statistics to print on montage as text.

    '''
    if stats is None: stats = dict()
    if isinstance(anat, str): anat = load_image(anat)
    assert nifti is not None
    assert anat is not None
    assert roi_dict is not None

    texcol = 0
    bgcol = 1
    iscale = 2.5
    if isinstance(nifti, list):
        weights = np.array([n.get_data() for n in nifti]).astype(floatX)
        weights = weights.transpose(1, 2, 3, 0)
        nifti = nifti[0]
    else:
        weights = nifti.get_data()

    features = weights.shape[-1]

    if clusters is not None:
        cls = np.unique(clusters).tolist()
        n_clusters = len(clusters)
        order = []
        _clusters = []
        for cl in cls:
            _cl = []
            for f in xrange(features):
                if clusters[f] == cl:
                    order.append(f)
                    _cl.append(f)
            _clusters.append(_cl)
        clusters = _clusters

    if order is None: order = range(features)

    if clusters is None:
        y = min(len(order), y)
        x = int(ceil(1.0 * len(order) / y))
    else:
        y = max([len(c) for c in clusters])
        x = len(clusters)

    indices = [0]
    if time_courses is not None: x *= 2
    font = {'size': 8}
    rc('font',**font)

    if fig is None:
        fig = plt.figure(figsize=[iscale * y, (1.5 * iscale) * x / 2.5])
    fig.set_facecolor((bgcol, bgcol, bgcol))
    plt.subplots_adjust(
        left=0.01, right=0.99, bottom=0.05, top=0.99, wspace=0.05, hspace=0.5)

    for i, f in enumerate(order):
        roi = roi_dict.get(f, None)
        if roi is None: continue

        if 'top_clust' in roi.keys():
            coords = roi['top_clust']['coords']
        else:
            coords = (0., 0., 0.)
        assert coords is not None

        feat = weights[:, :, :, f]

        if global_std is not None:
            feat /= global_std[f]
        else:
            feat /= feat.std()
        imax = np.max(np.absolute(feat)); imin = -imax
        imshow_args = {'vmax': imax, 'vmin': imin}
        coords = ([-coords[0], -coords[1], coords[2]])

        pad = 0
        if clusters is not None:
            for cluster in clusters:
                if f in cluster:
                    break
                else:
                    pad += y - len(cluster)
        i_ = i + pad

        if time_courses is not None:
            j = 2 * y * (i_ // y) + (i_ % y) + 1
        else:
            j = i_ + 1
        ax = fig.add_subplot(x, y, j)

        try:
            plot_map(feat, xyz_affine(nifti), anat=anat.get_data(),
                     anat_affine=xyz_affine(anat), threshold=thr, figure=fig,
                     axes=ax, cut_coords=coords, annotate=False, cmap=cmap,
                     draw_cross=False, **imshow_args)

        except Exception as e:
            logger.error(e)
            pass

        plt.text(0.05, 0.8, str(f),
                 transform=ax.transAxes,
                 horizontalalignment='center',
                 color=(texcol, texcol, texcol))
        for j, r in enumerate(roi['top_clust']['rois']):
            plt.text(0.05, -0.15 * (.5 + j), r[:35],
                     transform=ax.transAxes,
                     horizontalalignment='left',
                     color=(0, 0, 0))

        pos = [(0.05, 0.05), (0.4, 0.05), (0.8, 0.05)]
        colors = ['purple', 'blue', 'green', 'red']
        for j, (k, vs) in enumerate(stats.iteritems()):
            v = vs[f]
            if v is not None:
                plt.text(pos[j][0], pos[j][1], '%s=%.2f' % (k, v),
                         transform=ax.transAxes,
                         horizontalalignment='left',
                         color=colors[j])

        if time_courses is not None:
            j = y * (2 * (i_ // y) + 1) + (i_ % y) + 1
            ax = fig.add_subplot(x, y, j)
            for k, v in time_courses.iteritems():
                if v.ndim == 1:
                    tc = v
                else:
                    tc = v[f]
                ax.plot(tc, label=k)
            if i_ == 1:
                ax.legend()

    if out_file is not None:
        plt.savefig(out_file, facecolor=(bgcol, bgcol, bgcol))
    else:
        plt.show()
    plt.close()

def slice_montage(weights, thr=2, fig=None, out_file=None, order=None, y=8):
    texcol = 0
    bgcol = 1
    iscale = 2.5
    features = weights.shape[-1]
    order = order or range(features)
    y = min(len(order), y)

    indices = [0]
    x = int(ceil(1.0 * len(order) / y))

    font = {'size': 8}
    rc('font',**font)

    if fig is None:
        fig = plt.figure(figsize=[iscale * y, iscale * x])
    fig.set_facecolor((bgcol, bgcol, bgcol))
    plt.subplots_adjust(
        left=0.01, right=0.99, bottom=0.05, top=0.99, wspace=0.05, hspace=0.5)

    for i, f in enumerate(order):
        feat = weights[:, :, :, f]
        ax = fig.add_subplot(x, y, i + 1)
        ax.imshow(feat[:, :, 40], cmap='Greys_r')

        plt.text(0.05, 0.8, str(f), transform=ax.transAxes,
                 horizontalalignment='center', color=(texcol, texcol, texcol))

    if out_file is not None:
        plt.savefig(out_file, facecolor=(bgcol, bgcol, bgcol))
    else:
        plt.show()
    plt.close()

def unfolded_movie(niftis, files, anat, x=20, out_file=None, image_max=None,
                   image_std=None, stimulus=None, tmax=60, **responses):
    if isinstance(anat, str): anat = load_image(anat)

    iscale = 2.5
    y = int(ceil(1.0 * len(niftis[:tmax]) / x))
    cols = y
    y *= 2

    fig = plt.figure(figsize=[iscale * y, (1.5 * iscale) * x / 2.5])
    plt.subplots_adjust(
        left=0.01, right=0.99, bottom=0.05, top=0.99, wspace=0.05, hspace=0.5)

    widgets = ['Unfolding movie for component, ',
               Timer(), Bar()]
    pbar = ProgressBar(widgets=widgets, maxval=len(niftis[:tmax])).start()
    for i, nifti in enumerate(niftis[:tmax]):
        im = nilearn.image.load_img(files[i])
        if image_std is None: image_std = im.get_data().std()
        if image_max is None: image_max = im.get_data().max()
        j = 2 * (i // x) + (y * i) % (y * x) + 1
        ax = fig.add_subplot(x, y, j)
        nplt.plot_glass_brain(im, figure=fig, axes=ax, alpha=0.8,
                              vmax=image_max, symmetric_cbar=True,
                              cmap=plt.cm.RdYlBu_r, plot_abs=False,
                              threshold=2*image_std)
        pbar.update(i)

    if stimulus is not None or responses is not None:
        for c in xrange(cols):
            ax = fig.add_subplot(1, y, (c + 1) * 2)
            mi = c * x
            ma = (c + 1) * x

            if stimulus is not None:
                for k, v in stimulus.iteritems():
                    ax.plot(v[mi:ma] / v[mi:ma].std(),
                            np.arange(mi, ma), label=k)
            if responses is not None:
                for k, v in responses.iteritems():
                    ax.plot(v[mi:ma] / v[mi:ma].std(),
                            np.arange(mi, ma), label=k)

            ax.set_ylim([mi - 0.5, ma - 0.5])
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.legend()

    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

def make_argument_parser():
    '''Creates an ArgumentParser to read the options for this script from sys.argv

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('nifti', help='Nifti file to be processed.')
    parser.add_argument('--anat', default=None, help='Anat file for montage.')
    parser.add_argument('--rois', default=None, help='Pickled roi file.')
    parser.add_argument('--out', default=None, help='Output of montage.')
    parser.add_argument('--thr', default=2, help='Threshold for features.')
    return parser

def main(nifti_file, anat_file, roi_file, out_file, thr=2):
    '''Main function for running as a script.

    Args:
        nifti (str): path to 4D nifti file.
        anat (str): path to anatomical nifti file.
        roi_file (str): path to pickled roi dictionary file.
        out_file (str): path to output file.
        thr (float): threshold for `nipy.labs.viz.plot_map`

    '''
    iscale = 2
    nifti = load_image(nifti_file)
    anat = load_image(anat_file)
    roi_dict = pickle.load(open(roi_file, 'rb'))
    montage(nifti, anat, roi_dict, out_file=out_file)

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    main(args.nifti, args.anat, args.rois, args.out, args.thr)
