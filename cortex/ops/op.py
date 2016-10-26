'''Ops

'''

import theano


def center(X=None, mean_image=None):
    return (X - theano.shared(mean_image, name='mean_image')).astype(X.dtype)

_ops = {'center': center}
