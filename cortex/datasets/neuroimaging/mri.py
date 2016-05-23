'''
Module for the MRI dataset
'''

import nipy
from nipy.core.api import Image
import numpy as np
import os
from os import path
import pprint
from sklearn.decomposition import PCA
import warnings
import yaml

from ...analysis.mri import rois
from .. import BasicDataset
from . import nifti_viewer
from ...utils import floatX
from ...utils.tools import resolve_path


np.seterr(all='raise')


class MRI(BasicDataset):
    '''Basic MRI dataset iterator.

    Attributes:
        image_shape (tuple): shape of images for visualization.
        pca (Optional[sklearn.decomposition.PCA]): If not None, PCA
            decomposition of the data.
        pca_components (Optional[int]): number of PCA components if self.pca
            is not None
        tmp_path (str): path for temporary niftis in visualization.
        base_nifti_file (str): path for base nifti for forming niftis from arrays.
        anat_file (str): path for anatomical nifti file for visualization.
        sites (Optional[list]): list of sites where data was collected.
        mask (numpy.array): mask

    '''

    def __init__(self, source=None, name='mri', idx=None,
                 pca_components=0, distribution='gaussian', **kwargs):
        '''Init function for MRI.

        Args:
            source (str): path of the source.
            name (str): name of the dataset.
            idx (list): indices from the original dataset.
            pca_components: (Optional[int]): if not 0, decompose the data
                using PCA.
            distribution (Optional[str]): distribution of the primary data.
                See `models.distributions` for details.
            **kwargs: extra keyword arguments passed to BasicDataset

        '''
        print 'Loading %s from %s' % (name, source)
        source = resolve_path(source)
        X, Y = self.get_data(source)

        self.image_shape = self.mask.shape
        X = self._mask(X)
        self.pca_components = pca_components

        if self.pca_components and self.pca is None:
            self.pca = PCA(pca_components)
            print 'Performing PCA...'
            X = self.pca.fit_transform(X)

        data = {name: X, 'group': Y}
        distributions = {name: distribution, 'group': 'multinomial'}

        super(MRI, self).__init__(data, distributions=distributions, name=name,
                                  labels='group', **kwargs)

        if distribution == 'gaussian':
            self.X -= self.X.mean(axis=0)
            self.X /= self.X.std()
        elif distribution in ['continuous_binomial', 'binomial']:
            self.X -= self.X.min()
            self.X /= (self.X.max() - self.X.min())
        else:
            raise ValueError(distribution)

        self.mean_image = self.X.mean(axis=0)

        if idx is not None:
            self.X = self.X[idx]
            self.Y = self.Y[idx]

        self.n = self.X.shape[0]

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
        print('Loading file locations from %s' % source)
        source_dict = yaml.load(open(source))
        print('Source locations: %s' % pprint.pformat(source_dict))

        nifti_file = source_dict['nifti']
        mask_file = source_dict['mask']
        self.tmp_path = source_dict['tmp_path']
        if not path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)
        self.anat_file = source_dict['anat_file']
        pca_file = source_dict.get('pca', None)
        if pca_file is not None:
            with open(pca_file, 'rb') as f:
                self.pca = cPickle.load(f)
        else:
            self.pca = None

        data_files = source_dict['data']
        if isinstance(data_files, str):
            data_files = [data_files]

        X = []
        Y = []
        for i, data_file in enumerate(data_files):
            print 'Loading %s' % data_file
            X_ = np.load(data_file)
            X.append(X_.astype(floatX))
            Y.append((np.zeros((X_.shape[0],)) + i).astype(floatX))

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")

        self.mask = mask
        self.base_nifti_file = nifti_file

        if 'sites' in source_dict.keys():
            sites_file = source_dict['sites']
            self.sites = np.load(sites_file).tolist()
            n_sites = len(np.unique(self.sites).tolist())

            if n_sites > 1:
                print 'Regressing out site'

                for site in xrange(n_sites):
                    idx = [i for i, s in enumerate(self.sites) if s == site]
                    mi = X[idx].mean(axis=0)
                    X[idx] -= mi

        return X, Y

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
            print 'Data already masked'
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

    def save_images(self, x, out_file, remove_niftis=True,
                    order=None, stats=dict(), x_limit=None):
        '''Saves images from array.

        Args:
            x (numpy.array): array from which to make images.
            out_file (str): ouput file for image montage.
            remove_niftis (bool): delete images after making montage.
            order (Optional[list]): reorder the images in the montage.
            stats (Optional[dict]): dictionary of statistics to add to
                montage.
            x_limit (Optional(int)): if not None, limit the number of images
                along the x axis.

        '''
        if self.pca is not None and self.pca_components:
            X = self.pca.inverse_transform(X)

        if len(x.shape) == 3:
            x = x[:, 0, :]

        x = self._unmask(x)
        images, nifti_files = self.save_niftis(x)
        roi_dict = rois.main(nifti_files)

        if remove_niftis:
            for f in nifti_files:
                os.remove(f)
        nifti_viewer.montage(images, self.anat_file, roi_dict,
                             out_file=out_file,
                             order=order,
                             stats=stats)
