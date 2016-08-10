'''Basic 2D Image class

'''

from collections import OrderedDict
import cPickle
import gzip
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import BasicDataset, Dataset
from ...utils import concatenate, scan, _rng
from ...utils.maths import split_int_into_closest_two
from ...utils.tools import resolve_path
from ...utils.vis_utils import tile_raster_images


class TwoDImageDataset(BasicDataset):
    def __init__(self, data, image_shape=None, greyscale=True, **kwargs):
        if image_shape is None: raise TypeError('`image_shape` not set.')

        self.image_shape = image_shape
        self.greyscale = greyscale

        super(TwoDImageDataset, self).__init__(data, **kwargs)

    def viz(self, X=None, out_file=None):
        self.save_images(X=X, out_file=out_file)

    def register(self):
        super(TwoDImageDataset, self).register()

        from ... import _manager as manager

        if self.greyscale:
            c = 1
        else:
            c = 3
        datasets = manager.datasets
        datasets[self.name]['image_shape'] = (c,) + self.image_shape

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

        if X_in.shape == X_out.shape:
            X = np.concatenate([X_in, X_out], axis=0)
            mul = 2
        elif X_in.shape[1:] == X_out.shape:
            X = np.concatenate([X_in, X_out[None, :, :]], axis=0)
            mul = X_in.shape[0] + 1
        elif X_out.shape[1:] == X_in.shape:
            X = np.concatenate([X_in[None, :, :], X_out], axis=0)
            mul = X_out.shape[0] + 1
        else:
            raise ValueError('Shapes do not match (%s vs %s)'
                             % (X_in.shape, X_out.shape))

        y_str = y_str * mul
        n = len(y_str)
        annotations = zip(y_str, range(n))

        self.save_images(X=X,
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

        if not self.greyscale:
            if X.shape[1] % 3 != 0:
                raise TypeError('X has incorrect shape for color images.')
            div = X.shape[1] // 3
            X = X.reshape((X.shape[0], 3, div))
            X_r = X[:, 0]
            X_b = X[:, 1]
            X_g = X[:, 2]
            arrs = []
            for X in [X_r, X_b, X_g]:
                arr = tile_raster_images(
                    X=X, img_shape=self.image_shape, tile_shape=tshape,
                    tile_spacing=(spacing, spacing), bottom_margin=bottom_margin)
                arrs.append(arr)
            arr = np.array(arrs).transpose(1, 2, 0)
        else:
            arr = tile_raster_images(
                X=X, img_shape=self.image_shape, tile_shape=tshape,
                tile_spacing=(spacing, spacing), bottom_margin=bottom_margin)

        image = Image.fromarray(arr)
        if self.greyscale:
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
                try:
                    font = ImageFont.truetype(
                        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
                        font_size)
                except IOError:
                    font = ImageFont.truetype(
                        '/Library/Fonts/arial.ttf',
                        font_size)

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
                idr.line((0, t_x,
                          t_y, t_x),
                    fill=(255, 255, 255))

                idr.text((0, t_x), 'Legend: ', fill=(255, 255, 255),
                    font=font)

                for i, (label, c) in enumerate(zip(annotation_labels, colors)):
                    idr.text((((i + 1) * t_y) // 6, t_x), label, fill=c,
                        font=font)

        image.save(resolve_path(out_file))