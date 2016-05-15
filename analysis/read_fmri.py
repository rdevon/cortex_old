'''
Utility for handling fMRI data
'''

import argparse
from glob import glob
import nibabel as nib
from nipy import save_image, load_image
import numpy as np
import os
from os import path

from load_mri import (
    find_niftis,
    load_niftis
)


def read_niftis(file_list):
    data0 = load_image(file_list[0]).get_data()

    x, y, z, t = data0.shape
    print 'Found %d files with data shape is %r' % (len(file_list), data0.shape)

    data = []

    new_file_list = []
    for i, f in enumerate(file_list):
        print '%d) Loading subject from file: %s' % (i, f)

        nifti = load_image(f)
        subject_data = nifti.get_data()
        if subject_data.shape != (x, y, z, t):
            raise ValueError('Shape mismatch')
        subject_data -= subject_data.mean()
        subject_data /= subject_data.std()
        data.append(subject_data)
        new_file_list.append(f)
    data = np.concatenate(data, axis=3).transpose(3, 0, 1, 2).astype('float32')

    return data, new_file_list

def save_mask(data, out_path):
    print 'Getting mask'

    n, x, y, z = data.shape
    mask = np.zeros((x, y, z))

    mask[np.where(data.mean(axis=0) > data.mean())] = 1

    print 'Masked out %d out of %d voxels' % ((mask == 0).sum(), reduce(
        lambda x_, y_: x_ * y_, mask.shape))

    np.save(out_path, mask)

def load_niftis(source_dir, out_dir, name='mri', patterns=None):
    '''
    Loads niftis from a directory.
    '''

    if patterns is not None:
        file_lists = []
        for i, pattern in enumerate(patterns):
            file_list = glob(path.join(source_dir, pattern))
            file_lists.append(file_list)
    else:
        file_lists = [find_niftis(source_dir)]

    datas = []
    new_file_lists = []
    for i, file_list in enumerate(file_lists):
        data, new_file_list = read_niftis(file_list)
        new_file_lists.append(new_file_list)
        datas.append(data)
        np.save(path.join(out_dir, name + '_%d.npy' % i), data)

    sites = [[0 if 'st' in f else 1 for f in fl] for fl in file_lists]
    sites = sites[0] + sites[1]

    mask = save_mask(np.concatenate(datas, axis=0), path.join(out_dir, name + '_mask.npy'))
    np.save(path.join(out_dir, name + '_file_paths.npy'), new_file_lists)
    np.save(path.join(out_dir, name + '_sites.npy'), sites)

def make_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('source',
                        help='source directory for all subjects.')
    parser.add_argument('out_path',
                        help='output directory under args.name')
    parser.add_argument('-n', '--name', default='fmri')
    parser.add_argument('-p', '--patterns', nargs='+', default=None)

    return parser

if __name__ == '__main__':

    parser = make_argument_parser()
    args = parser.parse_args()

    source_dir = path.abspath(args.source)
    out_dir = path.abspath(args.out_path)

    if not path.isdir(out_dir):
        raise ValueError('No output directory found (%s)' % out_dir)

    load_niftis(source_dir, out_dir, args.name, patterns=args.patterns)
