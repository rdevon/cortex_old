'''Extension of nipy plot_map to do multiple maps.

'''

import numpy as np

from nipy.labs.viz_tools.activation_maps import _plot_anat
from nipy.labs.viz_tools.anat_cache import mni_sform, mni_sform_inv, _AnatCache
from nipy.labs.viz_tools.coord_tools import (
    coord_transform, find_maxsep_cut_coords)
from nipy.labs.viz_tools.slicers import SLICERS, _xyz_order
from nipy.labs.viz_tools.edge_detect import _fast_abs_percentile


CMAPS_ = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']

def plot_map(maps, affines, cut_coords=None, anat=None, anat_affine=None,
             slicer='ortho',
             figure=None, axes=None, title=None,
             threshold=None, annotate=True, draw_cross=True,
             do3d=False, threshold_3d=None,
             view_3d=(38.5, 70.5, 300, (-2.7, -12, 9.1)),
             black_bg=False, **imshow_kwargs):
    
    assert len(maps) == len(affines)
    
    for i in xrange(len(maps)):
        map, affine = _xyz_order(maps[i], affines[i])    
        nan_mask = np.isnan(np.asarray(map))
        if np.any(nan_mask):
            map = map.copy()
            map[nan_mask] = 0
        del nan_mask
        
        if threshold == 'auto': threshold = _fast_abs_percentile(map) + 1e-5

        if (cut_coords is None or isinstance(cut_coords, numbers.Number)
            ) and slicer in ['x', 'y', 'z']:
            cut_coords = find_maxsep_cut_coords(map, affine, slicer=slicer,
                                                threshold=threshold,
                                                n_cuts=cut_coords)
        maps[i] = map
        affines[i] = affine

    slicer = SLICERS[slicer].init_with_figure(data=maps[0], affine=affines[0],
                                              threshold=threshold,
                                              cut_coords=cut_coords,
                                              figure=figure, axes=axes,
                                              black_bg=black_bg,
                                              leave_space=False)

    if threshold:
        for i in xrange(len(maps)):
            maps[i] = np.ma.masked_inside(
                maps[i], -threshold, threshold, copy=False)

    _plot_anat(slicer, anat, anat_affine, title=title,
               annotate=annotate, draw_cross=draw_cross)

    for i in xrange(len(maps)):
        slicer.plot_map(maps[i], affines[i], **imshow_kwargs)
    return slicer