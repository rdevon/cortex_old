'''Eval script for pretrained RBMs.
'''

import numpy as np
import theano
from theano import tensor as T

from cortex.datasets import load_data
from cortex.models.rbm import unpack
from cortex.utils import floatX
from cortex.utils.training import (
    make_argument_parser_test,
    reload_model
)
from cortex.utils.tools import (
    get_trng,
    load_model,
    print_profile,
    print_section
)


def evaluate(
    model_to_load=None,
    center_input=False,
    dataset_args=dict(),
    out_path=None,
    mode='test',
    **kwargs):

    # ========================================================================
    print_section('Setting up data')
    train_batch_size = None
    test_batch_size = None
    test_batch_size = None
    if mode == 'train':
        train_batch_size = 10
    elif mode == 'test':
        test_batch_size = 10
    elif mode == 'valid':
        valid_batch_size = 10
    train, valid, test = load_data(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        valid_batch_size=valid_batch_size,
        **dataset_args)
    if mode == 'train':
        data_iter = train
    elif mode == 'test':
        data_iter = test
    elif mode == 'valid':
        data_iter = valid

    # ========================================================================
    print_section('Setting model and variables')
    dim_in = data_iter.dims[data_iter.name]

    X = T.matrix('x', dtype=floatX)
    X.tag.test_value = np.zeros((10, dim_in), dtype=X.dtype)
    trng = get_trng()

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print_section('Loading model and forming graph')

    models, _ = load_model(model_to_load, unpack, data_iter=data_iter)
    model = models['rbm']
    tparams = model.set_tparams()
    print_profile(tparams)

    # ========================================================================
    print_section('Testing')
    results, z_updates = model.update_partition_function(M=20, K=10000)
    f_update_partition = theano.function([], results.values(), updates=z_updates)
    outs = f_update_partition()
    out_dict = dict((k, v) for k, v in zip(results.keys(), outs))
    for k, v in out_dict.iteritems():
        if k == 'log_ws':
            print k, v[-10:]
            print v.shape
            print v.mean()
        else:
            print k, v

    nll = model.estimate_nll(X)
    f_nll = theano.function([X], nll)
    print f_nll(data_iter.X)

if __name__ == '__main__':
    parser = make_argument_parser_test()
    args = parser.parse_args()
    exp_dict = reload_model(args)
    evaluate(**exp_dict)