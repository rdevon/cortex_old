#!/usr/bin/env python

'''
Module for finding ROIS for MRI niftis
'''

import argparse
import itertools
import logging
import multiprocessing as mp
from nipy import load_image
from nipy import save_image
import numpy as np
import pickle
import pprint
import re
from scipy import (
    reshape, zeros, where, std,
    argmax, sqrt, ceil, floor, sign,
    negative, linspace, double, float16
)
import subprocess
from sys import stdout


# These are general names of regions for use elsewhere.
singles = ['Postcentral Gyrus',
           "Cingulate Gyrus",
           "Thalamus",
           "Superior Frontal Gyrus",
           "Pyramis",
           "Caudate",
           "Declive",
           "Cuneus",
           "Ulvula",
           "Medial Frontal Gyrus",
           "Precuneus",
           "Lingual Gyrus",
           "Paracentral Lobule",
           "Semi-Lunar Lobule",
           "Posterior Cingulate",
           "Culmen",
           "Cerebellar Tonsil",
           "Cingulate Gyrus",
           "Middle Frontal Gyrus",
           "Anterior Cingulate"
           ]

# Larger functional regions. Not used here, but can be referenced.
SC = ["Caudate","Putamen","Thalamus","Caudate Tail","Caudate Body","Caudate Head"]
AUD = ["Transverse Temporal Gyrus","Superior Temporal Gyrus"]
SM = ["Superior Parietal Lobule","Paracentral Lobule","Postcentral Gyrus","Precentral Gyrus"] #missing sma
VIS = ["Fusiform Gyrus","Lingual Gyrus","Middle Occipital Gyrus","Superior Occipital Gyrus","Inferior Occipital Gyrus","Cuneus","Middle Temporal Gyrus"] #missing calcarine gyrus
CC = ["Inferior Temporal Gyrus","Insula","Inferior Frontal Gyrus","Inferior Parietal Lobule","Middle Frontal Gyrus","Parahippocampal Gyrus"] #missing mcc
DM = ["Precuneus","Superior Frontal Gyrus","Posterior Cingulate","Anterior Cingulate","Angular Gyrus"]
CB = ["Cerebellar Tonsil","Pyramis"]

def lat_opposite(side):
    """
    Returns the lateral opposite as defined by the keyword pair {"Right", "Left"}
    """

    if side == "Right": return "Left"
    elif side == "Left": return "Right"
    else: raise ValueError("Lateral side error, (%s)" % side)

def check_pair(toproi, rois, lr_cm):
    toproi_split = toproi.split(" ",1)
    both = False
    if toproi_split[0] in ["Left", "Right"]:
        for roi in rois:
            roi_split = roi.split(" ",1)
            if (roi_split[1] == toproi_split[1]) & (roi_split[0] == lat_opposite(toproi_split[0])):
                both = True

    if both:
        toproi = " ".join(["(L+R)",toproi_split[1]])
    else:
        if abs(lr_cm) < 9:
            toproi = toproi.split(" ",1)[1]

    return toproi

def find_clusters_from_3D(fnifti, thr):
    """
    Function to use afni command line to find clusters from a 3D nifti.
    TODO(dhjelm): change this to use nipy functions.
    Parameters
    ----------
    fnifti: nifti file
        Nifti file to process.
    thr: float
        Threshold used for clusters.
    Returns
    -------
    cluster: a list of floats
    """
    cmd = ("3dclust "
           "-1Dformat -quiet -nosum -2thresh -2 %.2f "
           "-dxyz=1 2 80 2>/dev/null" % thr)
    awk = "awk '{ print $1\"\t\"$2\"\t\"$3\"\t\"$4\"\t\"$5\"\t\"$6\"\t\"$11\"\t\"$14\"\t\"$15\"\t\"$16}'"
    cmdline = cmd + " '%s'| " % fnifti + awk
    proc = subprocess.Popen(cmdline, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if "#**" in out.split(): return []

    cluster = float16(out.split())
    return cluster

def find_clusters_from_4D(fnifti, i, thr):
    """
    Function to use afni command line to find clusters from a 4D nifti.
    TODO(dhjelm): change this to use nipy functions.
    Parameters
    ----------
    fnifti: nifti file
        Nifti file to process.
    i: integer
        Index of the feature in the nifti file.
    thr: float
        Threshold used for clusters.
    Returns
    -------
    clusters: list of tuples of floats
        List of 3d clusters.
    """
    assert isinstance(i, int)
    assert isinstance(thr, (int, float))

    cmd = ("3dclust "
           "-1Dformat -quiet -nosum -1dindex %d -1tindex %d -2thresh -2 %.2f "
           "-dxyz=1 2 80 2>/dev/null" %
           (i, i, thr))
    awk = "awk '{ print $1\"\t\"$2\"\t\"$3\"\t\"$4\"\t\"$5\"\t\"$6\"\t\"$11\"\t\"$14\"\t\"$15\"\t\"$16}'"
    cmdline = cmd + " '%s'| " % fnifti + awk
    proc = subprocess.Popen(cmdline, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if "#**" in out.split(): return []

    clusters = float16(out.split())
    return clusters

def check_grey(coords):
    """
    Function to check if a particular cluster corresponds to grey matter.
    Note: this function uses the CA_N27_GW atlas. Other metrics could be used, but this feature needs
        to be added.
    Parameters
    ----------
    coords: tuple or list of floats
        Coordinates, should have length 3
    Returns
    -------
    prob: float
        probability of grey matter
    """
    assert len(coords) == 3
    atlas = "CA_N27_GW"

    # where am I command.
    waicmd = "whereami -atlas %s -space MNI %d %d %d 2>/dev/null" % ((atlas, ) + tuple(coords))
    proc = subprocess.Popen(waicmd, stdout=subprocess.PIPE, shell=True)
    (out,err) = proc.communicate()

    lines = out.split("\n")
    patt = re.compile("   Focus point: grey   \(p = ([0-9]\.[0-9]*)\)")
    prob = double([m.group(1) for m in [patt.match(line) for line in lines] if m])

    assert len(prob) == 1
    return prob[0]

def return_region(coords, atlas):
    """
    Function to return the region of interest as defined by the atlas.
    Parameters
    ----------
    cluster: tuple or list of floats
        Coordinates.  Should be length 3
    atlas: string
        Name of atlas to use.
    Returns
    -------
    region: list of strings
        This should be a list of all the regions corresponding to a coordinate.
    """
    assert atlas in ["CA_N27_ML", "CA_ML_18_MNIA", "TT_Daemon"], "Atlas %s not supported yet." % atlas
    assert len(coords) == 3

    # where am I command.
    waicmd = "whereami -atlas %s -space MNI %d %d %d 2>/dev/null" % ((atlas, ) + tuple(coords))
    proc = subprocess.Popen(waicmd, stdout=subprocess.PIPE, shell=True)
    (out,err) = proc.communicate()
            #if ROIONAME != "": rf.write(out)

    lines = out.split("\n")
    patt = re.compile("   Focus point: (.*)")
    region = [m.group(1) for m in [patt.match(line) for line in lines] if m]
    for i in range(len(region)):
        region[i] = " ".join(region[i].split())

    return region


def find_region_names(coords):
    """
    Get region names from a set of atlases.
    Note: only 3 atlases are currently supported, but more could be added in the future.
    Parameters
    ----------
    coords: tuple or list of floats
        Coordinates.  Should be length 3.
    Returns
    -------
    rois: list of strings
        List of regions of interest corresponding to a cluster coordinate.
    """
    assert len(coords) == 3

    n27 = "CA_N27_ML"
    mnia = "CA_ML_18_MNIA"
    tt = "TT_Daemon"

    rois = []
    for atlas in [n27, mnia, tt]:
        rois += return_region(coords, atlas)

    rois = list(set(rois))

    return rois

def get_cluster_info(clusters):
    if len(clusters) == 0:
        print("No clusters found for feature")
        return {}

    cluster_dict = {}
    intensity_sum = 0
    # Retrieve information on all the clusters.
    for c in range(len(clusters) // 10):
        cs = clusters[c * 10: (c+1) * 10]
        intensity_sum += abs(cs[0] * cs[6])

        cm = tuple([cs[x] for x in [1, 2, 3]])
        coords = tuple([cs[x] for x in [7, 8, 9]])

        rois = find_region_names(coords)
#            grey_value = check_grey(coords)

        cluster_dict[c] = dict(
            coords = coords,
            volume = cs[0],
            cm = cm,
            mean_intensity = abs(cs[6]),
            rois = rois
            )
#                             "grey_value": grey_value}

    # Given the cluster information found above, we find the "top" cluster.
    # The maximum clister is given by the one with the highest mean intensity * volume.
    max_int_prop = 0
    top_clust = None
    for k, cluster in cluster_dict.iteritems():
        cluster["int_prop"] = (cluster["mean_intensity"] * cluster["volume"] /
                               intensity_sum)

        if cluster["int_prop"] > max_int_prop or np.isnan(cluster["int_prop"]): # why nan?
            max_int_prop = cluster["int_prop"]
            top_clust = cluster

    if top_clust is not None:
        cluster_dict["top_clust"] = top_clust

    return cluster_dict

def cluster_worker(fnifti, thr, roi_dict):
    clusters = find_clusters_from_3D(fnifti, thr)
    try:
        idx = int(fnifti.split("/")[-1].split(".")[0])
    except:
        idx = int(fnifti.split("/")[-1].split(".")[0].split('_')[-1])
    roi_dict[idx] = get_cluster_info(clusters)

def worker_helper(args):
    cluster_worker(*args)

def find_rois(fnifti, thr):
    """
    Function for finding regions of interest from a nifti file.
    Parameters
    ----------
    fnifti: path to the nifti file or list of paths to files
    thr: float
        Threshold for clusters.
    Returns
    -------
    roidict: dictionary of int, dictionary pairs
    """
    print("Finding clusters from niftis")

    if isinstance(fnifti, str):
        nifti = load_image(fnifti)
        num_features = nifti.shape[-1]
        roi_dict = {}

        for i in xrange(num_features):
            clusters = find_clusters_from_4D(fnifti, i, thr)
            roi_dict[i] = get_cluster_info(clusters)

    elif isinstance(fnifti, list):
        num_features = len(fnifti)
        roi_dict = mp.Manager().dict()
        p = mp.Pool(num_features)
        args_iter = itertools.izip(fnifti,
                                   itertools.repeat(thr),
                                   itertools.repeat(roi_dict))
        p.map(worker_helper, args_iter)
        p.close()
        p.join()
        roi_dict = dict(roi_dict)
    else:
        raise NotImplementedError("Type %s not supported" % type(fnifti))

    print("Finished finding clusters")
    return roi_dict

def save_roi_txt(roi_dict, out_file):
    open(out_file, "w").close()
    with open(out_file, "a") as f:
        pprint.pprint(roi_dict, stream=f)

def main(fnifti, thr=0, out_file=None, out_txt=None):
    roi_dict = find_rois(fnifti, thr)

    if out_txt is not None:
        save_roi_txt(roi_dict, out_txt)
    if out_file is not None:
        pickle.dump(roi_dict, open(out_file, "w"))

    return roi_dict

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("nifti", help="Nifti file to be processed.")
    parser.add_argument("--out", default=None, help="Output pickle file of roi dict.")
    parser.add_argument("--txt", default=None, help="Readable txt file of rois.")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    roi_dict = main(args.nifti, out_file=args.out, out_txt=args.txt)
    if args.out is None and args.txt is None:
        print roi_dict