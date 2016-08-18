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

from .ni_dataset import NeuroimagingDataset
from ...analysis.mri import rois
from ...analysis import nifti_viewer
from ...utils import floatX
from ...utils.tools import resolve_path


np.seterr(all='raise')


class MRI(NeuroimagingDataset):
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
        if source is None:
            raise TypeError('`souce` argument must be provided')

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.logger.info('Loading %s from %s' % (name, source))

        widgets = [
            'Forming %s dataset: ' % name ,
            '(', Timer(), ') [', Percentage(), ']']
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

        data = {'input': X, 'labels': Y}
        distributions = {'input': distribution, 'labels': 'multinomial'}

        super(MRI, self).__init__(data, distributions=distributions, name=name,
                                  **kwargs)
        self.update_progress(finish=True)

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
            self.logger.info('Found %d subjects' % X_.shape[0])
        self.logger.info('Found %d groups' % len(X))
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
        X -= X.mean(axis=0, keepdims=True)
        X /= np.sqrt(X.std(axis=0, keepdims=True) ** 2 + 1e-6)

        return X, Y

    @staticmethod
    def mask_image(X, mask):
        if X.ndim == 5:
            reshape = X.shape[:2]
            X = X.reshape(
                (X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        else:
            reshape = None

        if X.shape[1:] != mask.shape:
            raise ValueError((X.shape, mask.shape))

        mask_f = mask.flatten()
        mask_idx = np.where(mask_f == 1)[0].tolist()
        X_masked = np.zeros((X.shape[0], int(mask.sum()))).astype(floatX)

        for i, x in enumerate(X):
            X_masked[i] = x.flatten()[mask_idx]

        if reshape is not None:
            X_masked = X_masked.reshape(
                (reshape[0], reshape[1], X_masked.shape[1]))

        return X_masked

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

        return MRI.mask_image(X, mask)

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

    def make_images(self, x, roi_dict=None, update_rois=True):
        '''Forms images.

        Args:
            x (numpy.array): array from which to make images.
            roi_dict (dict): roi dictionary.
            update_rois (bool): If true, update roi dictionary.
            signs (np.array or list).

        Returns:
            list: images.
            list: paths to nifti files.
            dict: roi dictionary.

        '''
        if roi_dict is None: roi_dict = dict()
        x = self.prepare_images(x)

        if len(x.shape) == 3:
            x = x[:, 0, :]

        x = self._unmask(x)
        images, nifti_files = self.save_niftis(x)

        if update_rois: roi_dict.update(**rois.main(nifti_files))

        return images, nifti_files, roi_dict

    def viz(self, x, out_file=None, remove_niftis=True, roi_dict=None,
            stats=None, update_rois=True, **kwargs):
        '''Saves images from array.

        Args:
            x (numpy.array): array from which to make images.
            out_file (str): ouput file for image montage.
            remove_niftis (bool): delete images after making montage.
            x_limit (Optional(int)): if not None, limit the number of images
                along the x axis.
            roi_dict (dict): roi dictionary.
            stats (Optional(dict)): dictionary of statistics.
            update_rois (bool): If true, update roi dictionary.
            **kwargs: keywork arguments for montage.

        '''
        images, nifti_files, roi_dict = self.make_images(
            x, roi_dict=roi_dict, update_rois=update_rois)

        if stats is None: stats = dict()
        stats['gm'] = [v['top_clust']['grey_value'] for v in roi_dict.values()]

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=resolve_path(out_file), stats=stats,
                             **kwargs)

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

_classes = {'MRI': MRI}
