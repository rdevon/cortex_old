'''
Module for the MRI dataset
'''

import cPickle
import logging
import nipy
from nipy.core.api import Image
import numpy as np
import os
from os import path
import pprint
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    Timer
)
import random
from sklearn.decomposition import PCA, IncrementalPCA
import warnings
import yaml

from ...analysis.mri import rois
from .. import BasicDataset
from ...analysis import nifti_viewer
from ...utils import floatX
from ...utils.tools import resolve_path


np.seterr(all='raise')


class MRI(BasicDataset):
    _init_steps = 9

    '''Basic MRI dataset iterator.

    Attributes:
        image_shape (tuple): shape of images for visualization.
        pca (Optional[sklearn.decomposition.PCA]): If not None, PCA
            decomposition of the data.
        pca_components (Optional[int]): number of PCA components if self.pca
            is not None
        n_subjects (Optional[int]): number of subjects in dataset.
        tmp_path (str): path for temporary niftis in visualization.
        base_nifti_file (str): path for base nifti for forming niftis from arrays.
        anat_file (str): path for anatomical nifti file for visualization.
        sites (Optional[list]): list of sites where data was collected.
        mask (numpy.array): mask

    '''

    def __init__(self, source=None, name='mri', flip_signs=False,
                 pca_components=0, incremental_pca=False,
                 distribution='gaussian', **kwargs):
        '''Init function for MRI.

        Args:
            source (str): path of the source.
            name (str): name of the dataset.
            pca_components (Optional[int]): if not 0, decompose the data
                using PCA.
            incremental_pca (bool): Use incremental PCA instead of standard.
            distribution (Optional[str]): distribution of the primary data.
                See `models.distributions` for details.
            **kwargs: extra keyword arguments passed to BasicDataset

        '''
        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.logger.info('Loading %s from %s' % (name, source))

        widgets = ['Forming %s dataset: ' % name , ' (', Timer(), ') [', Percentage(), ']']
        self.pbar = ProgressBar(widgets=widgets, maxval=self._init_steps).start()
        self.progress = 0

        source = resolve_path(source)
        X, Y = self.get_data(source)

        self.image_shape = self.mask.shape
        self.update_progress()
        self.pca_components = pca_components

        if self.pca_components:
            X -= X.mean(axis=0)
            if self.pca is None:
                if incremental_pca:
                    self.logger.info('Using incremental PCA')
                    PCAC = IncrementalPCA
                else:
                    self.logger.info('Using PCA')
                    PCAC = PCA
                self.pca = PCAC(pca_components, whiten=True)
                self.logger.info('Fitting PCA... (please wait)')
                self.pca.fit(X)
                if self.pca_file is not None:
                    with open(self.pca_file, 'wb') as pf:
                        cPickle.dump(self.pca, pf)
            self.logger.info('Performing PCA')
            X = self.pca.transform(X)
        self.update_progress()

        data = {name: X, 'group': Y}
        distributions = {name: distribution, 'group': 'multinomial'}

        super(MRI, self).__init__(data, distributions=distributions, name=name,
                                  labels='group', **kwargs)
        self.update_progress()

        if distribution == 'gaussian':
            if self.pca_components == 0:
                self.X -= self.X.mean(axis=0)
                self.X /= self.X.std()
        elif distribution in ['continuous_binomial', 'binomial']:
            self.X -= self.X.min()
            self.X /= (self.X.max() - self.X.min())
        else:
            raise ValueError(distribution)

        self.mean_image = self.X.mean(axis=0)
        self.update_progress()

        self.n = self.X.shape[0]
        self.update_progress(finish=True)

    def slice_data(self, idx):
        for k, v in self.data.iteritems():
            self.data[k] = v[idx]
        self.X = self.data[self.name]
        if self.labels in self.data.keys():
            self.Y = self.data[self.labels]
        self.n_subjects = len(idx)
        self.n = self.X.shape[0]

    @staticmethod
    def factory(C=None, split=None, idx=None, batch_sizes=None, **kwargs):
        if C is None:
            C = MRI
        mri = C(batch_size=10, **kwargs)
        if hasattr(mri, 'pca'):
            logger = mri.logger
            mri.logger = None
        else:
            logger = None

        if idx is None:
            logger.info('Splitting dataset into ratios %r' % split)
            if round(np.sum(split), 5) != 1. or len(split) != 3:
                raise ValueError(split)

            if mri.balance:
                l_idx = [np.where(label == 1)[0].tolist() for label in mri.Y[:, 0, :].T]
                train_idx = []
                valid_idx = []
                test_idx = []
                for l in l_idx:
                    random.shuffle(l)
                    split_idx = []
                    accum = 0
                    for s in split:
                        s_i = int(s * len(l) + accum)
                        split_idx.append(s_i)
                        accum += s_i
                    train_idx += l[:split_idx[0]]
                    valid_idx += l[split_idx[0]:split_idx[1]]
                    test_idx += l[split_idx[1]:]
            else:
                split_idx = []
                accum = 0
                for s in split:
                    s_i = int(s * mri.n_subjects + accum)
                    split_idx.append(s_i)
                    accum += s_i
                idx = range(mri.n_subjects)
                random.shuffle(idx)

                train_idx = idx[:split_idx[0]]
                valid_idx = idx[split_idx[0]:split_idx[1]]
                test_idx = idx[split_idx[1]:]
            idx = [train_idx, valid_idx, test_idx]
        else:
            logger.info('Splitting dataset into ratios %.2f / %.2f /%.2f '
                        'using given indices'
                        % tuple(len(idx[i]) / float(mri.n_subjects)
                                for i in range(3)))

        assert len(batch_sizes) == len(idx)

        datasets = []
        modes = ['train', 'valid', 'test']
        for i, bs, mode in zip(idx, batch_sizes, modes):
            if bs is None:
                dataset = None
            else:
                dataset = mri.copy()
                dataset.slice_data(i)
                dataset.batch_size = bs
                dataset.logger = logger
                dataset.mode = mode
            datasets.append(dataset)

        return datasets + [idx]

    def get_data(self, source):
        '''Fetch the MRI dataset.

        MRI dataset source is a yaml file.
        An example format of said yaml is::

            name: 'aod',
            data: [
                '/Users/devon/Data/AOD/AOD_0.npy',
                '/Users/devon/Data/AOD/AOD_1.npy'
            ],
            mask: '/Users/devon/Data/AOD/AOD_mask.npy',
            nifti: '//Users/devon/Data/VBM/H000A.nii',
            tmp_path: '/Users/devon/Data/mri_tmp/',
            anat_file: '/Users/devon/Data/ch2better_whitebg_aligned2EPI_V4.nii',
            sites: '/Users/devon/Data/AOD/AOD_sites.npy'

        '''
        self.logger.info('Loading file locations from %s' % source)
        source_dict = yaml.load(open(source))
        self.update_progress()

        self.logger.debug('Source locations: \n%s' % pprint.pformat(source_dict))

        nifti_file = source_dict['nifti']
        mask_file = source_dict['mask']
        self.tmp_path = source_dict['tmp_path']
        if not path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)
        self.anat_file = source_dict['anat_file']
        self.pca_file = source_dict.get('pca', None)
        if self.pca_file is not None:
            try:
                with open(pca_file, 'rb') as f:
                    self.pca = cPickle.load(f)
            except (IOError, EOFError):
                self.pca = None
        else:
            self.pca = None
        self.update_progress()

        data_files = source_dict['data']
        if isinstance(data_files, str):
            data_files = [data_files]

        X = []
        Y = []
        for i, data_file in enumerate(data_files):
            self.logger.info('Loading %s' % data_file)
            self.update_progress(progress=False)
            X_ = np.load(data_file)
            X.append(X_.astype(floatX))
            Y.append((np.zeros((X_.shape[0],)) + i).astype(floatX))
        self.update_progress()

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")
        self.update_progress()

        self.mask = mask
        self.base_nifti_file = nifti_file

        if 'sites' in source_dict.keys():
            sites_file = source_dict['sites']
            self.sites = np.load(sites_file).tolist()
            n_sites = len(np.unique(self.sites).tolist())

            if n_sites > 1:
                self.logger.info('Regressing out site')

                for site in xrange(n_sites):
                    idx = [i for i, s in enumerate(self.sites) if s == site]
                    mi = X[idx].mean(axis=0)
                    X[idx] -= mi
        self.update_progress()
        self.n_subjects = X.shape[0]

        X = self._mask(X)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)

        return X, Y

    def update_progress(self, finish=False, progress=True):
        if progress:
            self.progress += 1
        if finish:
            self.pbar.update(self._init_steps)
        else:
            self.pbar.update(self.progress)

    def _mask(self, X, mask=None):
        '''Mask the data.

        Args:
            X (numpy.array): data to be masked
            mask (Optional[numpy.array]): mask

        Return:
            numpy.array: masked array.

        '''
        if mask is None:
            mask = self.mask

        if X.shape[1] == mask.sum():
            self.logger.debug('Data already masked')
            return X

        if X.shape[1:] != mask.shape:
            raise ValueError((X.shape, mask.shape))

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X_masked = np.zeros((X.shape[0], int(mask.sum()))).astype(floatX)

        for i, x in enumerate(X):
            X_masked[i] = x.flatten()[mask_idx]

        return X_masked

    def _unmask(self, X_masked, mask=None):
        '''Unmask data.

        Args:
            X_masked (numpy.array): array to be unmasked.
            mask (Optional[numpy.array]): mask

        Returns:
            numpy.array: unmasked data.

        '''
        if mask is None:
            mask = self.mask

        if X_masked.shape[1] != mask.sum():
            raise ValueError('Masked data does not fit mask %r vs %r' % (X_masked.shape, mask.sum()))

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X = np.zeros((X_masked.shape[0],) + self.image_shape).astype(floatX)
        for i, x_m in enumerate(X_masked):
            x_f = X[i].flatten()
            x_f[mask_idx] = x_m
            X[i] = x_f.reshape(self.image_shape)

        return X

    def make_image(self, X, base_nifti):
        '''Create a nitfi image from array.

        Args:
            X (numpy.array): array from which to make nifti image.
            base_nifti (nipy.core.api.Image): nifti image template.

        Returns:
            nipy.core.api.Image

        '''
        image = Image.from_image(base_nifti, data=X)
        return image

    def save_niftis(self, X):
        '''Save nifti files from array.

        Args:
            X (numpy.array): array from which to make images.

        Returns:
            list: list of nifti images.
            list: list of output files for images.

        '''

        base_nifti = nipy.load_image(self.base_nifti_file)

        images = []
        out_files = []
        for i, x in enumerate(X):
            image = self.make_image(x, base_nifti)
            out_file = path.join(self.tmp_path, 'tmp_image_%d.nii.gz' % i)
            nipy.save_image(image, out_file)
            images.append(image)
            out_files.append(out_file)

        return images, out_files

    def prepare_images(self, x):
        if self.pca is not None and self.pca_components:
            x = self.pca.inverse_transform(x)
        return x

    def save_images(self, x, out_file=None, remove_niftis=True,
                    x_limit=None, roi_dict=None, signs=None, stats=None,
                    **kwargs):
        '''Saves images from array.

        Args:
            x (numpy.array): array from which to make images.
            out_file (str): ouput file for image montage.
            remove_niftis (bool): delete images after making montage.
            x_limit (Optional(int)): if not None, limit the number of images
                along the x axis.
            stats (Optional(dict)): dictionary of statistics.
            **kwargs: keywork arguments for montage.

        '''
        x = self.prepare_images(x)

        if len(x.shape) == 3:
            x = x[:, 0, :]

        if signs is not None:
            x *= signs[:, None]

        x = self._unmask(x)
        images, nifti_files = self.save_niftis(x)

        if roi_dict is None: roi_dict = dict()
        roi_dict.update(**rois.main(nifti_files))

        if stats is None: stats = dict()
        stats['gm'] = [roi_dict[i]['top_clust']['grey_value'] for i in roi_dict.keys()]

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=out_file, stats=stats, **kwargs)

    def visualize_pca(self, out_file, **kwargs):
        '''Saves the PCA component image.

        Args:
            out_file (str): ouput file for image montage.
            **kwargs: keyword arguments for saving images.

        '''
        if self.pca is None:
            raise ValueError('No PCA found.')
        y = np.eye(self.pca_components).astype(floatX)

        self.save_images(y, out_file, **kwargs)