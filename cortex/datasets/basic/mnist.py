'''MNIST dataset.

'''

from collections import OrderedDict
import cPickle
import gzip
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

from .. import BasicDataset, Dataset
from ...utils import concatenate, scan, _rng
from ...utils.maths import split_int_into_closest_two
from ...utils.tools import resolve_path
from ...utils.vis_utils import tile_raster_images


logger = logging.getLogger(__name__)


class MNIST(BasicDataset):
    '''MNIST dataset iterator.

    Attributes:
        image_shape (tuple): dimensions of original images.

    '''
    _viz = ['classification_visualization']

    def __init__(self, source=None, restrict_digits=None, mode='train',
                 binarize=False, name='mnist',
                 out_path=None, **kwargs):
        '''Init function for MNIST.

        Args:
            source (str): Path to source gzip file.
            restrict_digits (Optional[list]): list of digits to restrict
                iterator to.
            mode (str): `train`, `test`, or `valid`.
            out_path (Optional[str]): path for saving visualization output.
            name (str): name of dataset.
            **kwargs: eXtra keyword arguments passed to BasicDataset

        '''
        if source is None:
            raise TypeError('No source file provided')

        logger.info('Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source))

        source = resolve_path(source)

        X, Y = self.get_data(source, mode)

        if restrict_digits is not None:
            X = np.array([x for i, x in enumerate(X) if Y[i] in restrict_digits])
            Y = np.array([y for i, y in enumerate(Y) if Y[i] in restrict_digits])

        data = {'input': X, 'labels': Y}
        distributions = {'input': 'binomial', 'labels': 'multinomial'}

        super(MNIST, self).__init__(data, distributions=distributions,
                                    name=name, mode=mode, **kwargs)

        self.image_shape = (28, 28)

        if binarize:
            self.data[name] = _rng.binomial(
                p=self.data[name], size=self.data[name].shape, n=1).astype('float32')

    def get_data(self, source, mode):
        '''Fetch data from gzip pickle.

        Args:
            source (str): path to source.
            mode (str): `train`, `test`, or `valid`.

        '''
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        if mode == 'train':
            X = np.float32(x[0][0])
            Y = np.float32(x[0][1])
        elif mode == 'valid':
            X = np.float32(x[1][0])
            Y = np.float32(x[1][1])
        elif mode == 'test':
            X = np.float32(x[2][0])
            Y = np.float32(x[2][1])
        else:
            raise ValueError()

        return X, Y

    def classification_visualization(self, idx=None, X=None, Y=None,
                                     Y_pred=None, P=None, out_file=None):

        yp_str = ['%d' % y for y in Y_pred]
        yh_str = ['%d' % y for y in Y]
        p_str = ['%.2f' % p for p in P]
        n = len(idx)
        annotations = zip(zip(yh_str, yp_str, p_str), range(n))
        self.save_images(X=X, annotations=annotations,
                         annotation_labels=[
                            'ground truth', 'estimate', 'probability'],
                         spacing=10, out_file=out_file)

    def autoencoder_visualization(self, X_in=None, X_out=None, Y=None, idx=None,
                                  out_file=None):

        Y = np.argmax(Y, axis=-1)
        y_str = ['%d' % y for y in Y]
        y_str = y_str * 2
        n = len(y_str)
        annotations = zip(y_str, range(n))
        self.save_images(X=np.concatenate([X_in, X_out], axis=0),
                         annotations=annotations,
                         annotation_labels=['label'],
                         out_file=out_file)

    def save_images(self, X=None, annotations=None, annotation_labels=None,
                    transpose=False, font_size=10, spacing=1, out_file=None):
        '''Saves visualization.

        Args:
            X (numpy.array): array to be visualized.
            imgfile (str): output file.
            transpose (bool): if True, then transpose images.

        '''
        if len(X.shape) == 2:
            i_x, i_y = split_int_into_closest_two(X.shape[0])
            X = X.reshape((i_x, i_y, X.shape[1]))
        else:
            i_x = X.shape[0]
            i_y = X.shape[1]

        bottom_margin = 40

        if transpose:
            X = X.reshape((X.shape[0], X.shape[1], self.image_shape[0], self.image_shape[1]))
            X = X.transpose(0, 1, 3, 2)
            X = X.reshape((X.shape[0], X.shape[1],
                           self.image_shape[0] * self.image_shape[1]))

        tshape = (X.shape[0], X.shape[1])
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        image = Image.fromarray(
            tile_raster_images(
                X=X, img_shape=self.image_shape, tile_shape=tshape,
                tile_spacing=(spacing, spacing), bottom_margin=bottom_margin))
        image = Image.merge('RGB', (image, image, image))
        if annotations is not None:
            colors = [
                (200, 0, 200), (50, 255, 50), (50, 50, 255), (255, 50, 50)]
            r_x = (3 * self.image_shape[0]) // 4
            r_y = self.image_shape[1]
            rpos = [(0, 0), (r_x, 0), (0, r_y), (r_x, r_y)]
            idr = ImageDraw.Draw(image)

            try:
                ImageFont.truetype('arial.ttf', font_size)
            except IOError:
                font = ImageFont.truetype('/Library/Fonts/arial.ttf', font_size)

            for txt, pos in annotations:
                if isinstance(pos, (list, tuple)):
                    (x_, y_) = pos
                else:
                    x_ = pos % i_y
                    y_ = pos // i_y

                t_x = x_ * (self.image_shape[0] + spacing)
                t_y = y_ * (self.image_shape[1] + spacing)

                if isinstance(txt, str):
                    txt = [txt]
                for t, r, c in zip(txt, rpos, colors):
                    _x = t_x + r[0]
                    _y = t_y + r[1]
                    idr.text((_x, _y), t, fill=c, font=font)

            if annotation_labels is not None:
                t_x = i_x * (self.image_shape[0] + spacing)
                t_y = i_y * (self.image_shape[1] + spacing)
                idr.line((0, t_y,
                          t_x, t_y),
                    fill=(255, 255, 255))

                idr.text((0, t_y), 'Legend: ', fill=(255, 255, 255),
                    font=font)

                for i, (label, c) in enumerate(zip(annotation_labels, colors)):
                    idr.text((((i + 1) * t_x) // 6, t_y), label, fill=c,
                        font=font)

        image.save(out_file)


_classes = {'MNIST': MNIST}