'''Utility for handling fMRI data.

'''

import argparse
from glob import glob
import nibabel as nib
from nipy import save_image, load_image
import numpy as np
import os
from os import path
import readline
import sys
import yaml

from .load_mri import find_niftis, load_niftis, make_argument_parser
from ..utils.extra import complete_path


def read_niftis(file_list):
    '''Finds nifti files in a directory.

    Args:
        source (str): The source directory for niftis

    Returns:
        list: List of file paths.

    '''
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
    data = np.array(data).transpose(0, 4, 1, 2, 3).astype('float32')

    return data, new_file_list

def save_mask(data, out_path):
    '''Save mask of data.

    Args:
        data (numpy.array): Data to mask
        out_path (str): Output path for mask.

    '''
    print 'Getting mask'

    s, n, x, y, z = data.shape
    mask = np.zeros((x, y, z))
    _data = data.reshape((s * n, x, y, z))

    mask[np.where(_data.mean(axis=0) > _data.mean())] = 1

    print 'Masked out %d out of %d voxels' % ((mask == 0).sum(), reduce(
        lambda x_, y_: x_ * y_, mask.shape))

    np.save(out_path, mask)

def load_niftis(source_dir, out_dir, name='fmri', patterns=None):
    '''Loads niftis from a directory.

    Saves the data, paths, mask, and `sites`.

    Args:
        source_dir (str): Directory of nifti files.
        out_dir (str): Output directory for saving arrays, etc.
        name (str): Name of dataset.
        patterns (Optional[list]): list of glob for filtering files.

    '''

    if patterns is not None:
        file_lists = []
        for i, pattern in enumerate(patterns):
            file_list = glob(path.join(source_dir, pattern))
            file_lists.append(file_list)
    else:
        file_lists = [find_niftis(source_dir)]

    base_file = file_lists[0][0]
    paths_file = path.join(out_dir, name + '_file_paths.npy')
    sites_file = path.join(out_dir, name + '_sites.npy')
    mask_file = path.join(out_dir, name + '_mask.npy')
    yaml_file = path.join(out_dir, name + '.yaml')
    tmp_dir = path.join(out_dir, name + '_tmp')
    if not path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete_path)
    print ('The fMRI dataset requires an anatomical nifti file to visualize'
           ' properly. Enter the path for the anatomical file or leave blank'
           ' if you plan not to use visualization or will enter into the yaml'
           ' file later.')

    anat_file = raw_input('Anat file: ')
    if anat_file == '': yaml_file = None

    datas = []
    new_file_lists = []
    data_paths = []
    for i, file_list in enumerate(file_lists):
        data, new_file_list = read_niftis(file_list)
        new_file_lists.append(new_file_list)
        datas.append(data)
        data_path = path.join(out_dir, name + '_%d.npy' % i)
        data_paths.append(data_path)
        np.save(data_path, data)

    sites = [[0 if 'st' in f else 1 for f in fl] for fl in file_lists]
    sites = sites[0] + sites[1]

    save_mask(np.concatenate(datas, axis=0), mask_file)
    np.save(paths_file, new_file_lists)
    np.save(sites_file, sites)
    with open(yaml_file, 'w') as yf:
        yf.write(
            yaml.dump(
                dict(name=name,
                     data=data_paths,
                     mask=mask_file,
                     nifti=base_file,
                     sites=sites_file,
                     tmp_path=tmp_dir,
                     anat_file=anat_file
                     )
                )
            )

def main(args=None):
    '''Main routine.

    '''
    if args is None:
        args = sys.argv[1:]

        parser = make_argument_parser()
    args = parser.parse_args()

    source_dir = path.abspath(args.source)
    out_dir = path.abspath(args.out_path)

    if not path.isdir(out_dir):
        raise ValueError('No output directory found (%s)' % out_dir)

    load_niftis(source_dir, out_dir, args.name, patterns=args.patterns)

if __name__ == '__main__':

    parser = make_argument_parser()
    args = parser.parse_args()

    source_dir = path.abspath(args.source)
    out_dir = path.abspath(args.out_path)

    if not path.isdir(out_dir):
        raise ValueError('No output directory found (%s)' % out_dir)

    load_niftis(source_dir, out_dir, args.name, patterns=args.patterns)
