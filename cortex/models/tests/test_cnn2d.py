'''Tests for CNN

'''

from collections import OrderedDict
import logging
import numpy as np
from pprint import pformat, pprint
import theano
from theano import tensor as T

import cortex
from cortex import models
from cortex.datasets.basic.euclidean import Euclidean
from cortex.models import mlp as module
from cortex.utils import floatX, logger as cortex_logger


logger = logging.getLogger(__name__)
cortex_logger.set_stream_logger(2)
_atol = 1e-6
manager = cortex._manager

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softplus = lambda x: np.log(1.0 + np.exp(x))
identity = lambda x: x

def test_fetch_class(c='CNN2D'):
    C = cortex.resolve_class(c)
    return C

def test_make_cnn(input_shape=(3, 19, 19), filter_shapes=((5, 5), (3, 3)),
                  pool_sizes=((2, 2), (2, 2)), n_filters=[100, 200],
                  h_act='softplus'):
    C = test_fetch_class()
    cnn = C(input_shape=input_shape, filter_shapes=filter_shapes,
            pool_sizes=pool_sizes, n_filters=n_filters, h_act=h_act)
    return cnn

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    '''Shamelessly copied from 'CS231n/assignment2/cs231n/im2col.py'

    '''

    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    '''Shamelessly copied from 'CS231n/assignment2/cs231n/im2col.py'

    '''

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def im2col(image, h, w, pad, stride):
    cols = []
    def pad_with_zeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    image_padded = np.lib.pad(image, pad, pad_with_zeros)
    im_y, im_x = image_padded.shape[2:]

    for y in xrange(0, im_y - h, stride):
        for x in xrange(0, im_x - w, stride):
            cols.append(image_padded[:, :, y:y+h, x:x+h])

    return np.array(cols)

def conv_forward_fast(x, w, b, stride=1, pad=0):
    '''Shamelessly copied from 'CS231n/assignment2/cs231n/im2col.py'

    '''
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) / stride + 1
    out_width = (W + 2 * pad - filter_width) / stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    return out

def feed_numpy(cnn, x):
    x = x.reshape((x.shape[0],) + cnn.input_shape)
    conv_outs = []
    pool_outs = []
    outs = []

    for l in xrange(cnn.n_layers):
        W = cnn.params['weights'][l]
        b = cnn.params['biases'][l]
        shape = cnn.filter_shapes[l]
        pool_size = cnn.pool_sizes[l]

        conv_out = conv_forward_fast(x, W, b)
        conv_outs.append(conv_out)
        n, d, h, w = conv_out.shape
        conv_out_r = conv_out.reshape(n * d, 1, h, w)

        col = im2col_indicies(conv_out_r, pool_size, pool_size, padding=0)
        max_idx = np.argmax(col, axis=0)

        pool_out = col[max_idx, range(max_idx.size)]
        pool_out = pool_out.reshape(
            shape[0] // pool_size[0], shape[1] // pool_size[1], n, d)
        pool_out = pool_out.transpose(2, 3, 0, 1)
        pool_outs.append(pool_out)
        x = pool_out
        outs.append(x)

    n, d, h, w = x.shape
    x = x.reshape(n, d * h * w)
    outs.append(x)
    return conv_outs, pool_outs, outs

def test_feed_forward(cnn=None, X=T.matrix('X', dtype=floatX), x=None,
                      batch_size=23):
    logger.debug('Testing CNN feed forward')
    if cnn is None:
        cnn = test_make_cnn()
    if x is None:
        x = np.random.randint(0, 2, size=(batch_size,) + cnn.input_shape)

    outs = cnn(X)
    conv_outs, pool_outs, outs = feed_numpy(cnn, x)