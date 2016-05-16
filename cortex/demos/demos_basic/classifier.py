'''
Demo for training a classifier.

Try with `python classifier.py classifier_mnist.yaml`
'''

from collections import OrderedDict
import numpy as np
from os import path
import pprint
import sys
import theano
from theano import tensor as T
import time

from cortex.datasets import load_data
from cortex.models.mlp import MLP
from cortex.utils import floatX
from cortex.utils.monitor import SimpleMonitor
from cortex.utils.preprocessor import Preprocessor
from cortex.utils.tools import get_trng, print_profile, print_section
from cortex.utils.training import (
    main_loop,
    make_argument_parser,
    set_experiment,
    set_model,
    set_optimizer,
    set_params
)


def init_learning_args(
    learning_rate=0.01,
    l2_decay=0.,
    dropout=0.,
    optimizer='rmsprop',
    optimizer_args=dict(),
    learning_rate_schedule=None,
    batch_size=100,
    valid_batch_size=100,
    epochs=100,
    valid_key='error',
    valid_sign='+'):
    '''Default learning args.

    This method acts as a filter for kwargs.

    Args:
        learning_rate: float.
        l2_decay: float, L2 decay rate.
        dropout: float, dropout_rate.
        optimizer: str, see utils.op
        optimizer_args: dict, extra kwargs for op.
        learning_rate_schedule: OrderedDict, schedule for learning rate.
        batch_size: int
        valid_batch_size: int
        epochs: int
        valid_key: str, key from results to validate model on.
        valid_sign: str, + or -. If -, then sign is switched at validation.
            Good for upperbounds.
    Returns:
        locals().
    '''
    return locals()

def train(
    out_path=None, name='', model_to_load=None, test_every=None,
    classifier=None, preprocessing=None,
    learning_args=None,
    dataset_args=None):
    '''Basic training script.

    Args:
        out_path: str, path for output directory.
        name: str, name of experiment.
        test_every: int (optional), if not None, test every n epochs instead of
            every 1 epoch.
        classifier: dict, kwargs for MLP factory.
        learning_args: dict or None, see `init_learning_args` above for options.
        dataset_args: dict, arguments for Dataset class.
    '''

    # ========================================================================
    if preprocessing is None: preprocessing = []
    if learning_args is None: learning_args = dict()
    if dataset_args is None: raise ValueError('Dataset args must be provided')

    learning_args = init_learning_args(**learning_args)
    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Learning args: %s' % pprint.pformat(learning_args)

    # ========================================================================
    print_section('Setting up data')
    input_keys = dataset_args.pop('keys')
    batch_size = learning_args.pop('batch_size')
    valid_batch_size = learning_args.pop('valid_batch_size')
    train, valid, test = load_data(
        train_batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        **dataset_args)

    # ========================================================================
    print_section('Setting model and variables')
    dim_in = train.dims[input_keys[0]]
    dim_out = train.dims[input_keys[1]]

    X = T.matrix('x', dtype=floatX) # Input data
    Y = T.matrix('y', dtype=floatX) # Lables
    X.tag.test_value = np.zeros((batch_size, dim_in), dtype=X.dtype)
    Y.tag.test_value = np.zeros((batch_size, dim_out), dtype=X.dtype)
    trng = get_trng()

    preproc = Preprocessor(preprocessing)
    X_i = preproc(X, data_iter=train)
    inps = [X, Y]

    # ========================================================================
    print_section('Loading model and forming graph')
    dropout = learning_args.pop('dropout')

    def create_model():
        model = MLP.factory(dim_in=dim_in, dim_out=dim_out,
                            distribution=train.distributions[input_keys[1]],
                            dropout=dropout,
                            **classifier)
        models = OrderedDict()
        models[model.name] = model
        return models

    def unpack(dim_in=None, dim_out=None, mlp=None, **model_args):
        model = MLP.factory(dim_in=dim_in, dim_out=dim_out, **mlp)
        models = [model]
        return models, model_args, None

    models = set_model(create_model, model_to_load, unpack)
    model = models['MLP']
    tparams = model.set_tparams()
    print_profile(tparams)

    # ==========================================================================
    print_section('Getting cost')
    outs = model(X_i)
    p = outs['p']
    base_cost = model.neg_log_prob(Y, p).sum(axis=0)
    cost = base_cost

    updates = theano.OrderedUpdates()

    l2_decay = learning_args.pop('l2_decay')
    if l2_decay > 0.:
        print 'Adding %.5f L2 weight decay' % l2_decay
        l2_rval = model.l2_decay(l2_decay)
        l2_cost = l2_rval.pop('cost')
        cost += l2_cost

    constants = []
    extra_outs = []
    extra_outs_keys = ['cost']

    # ==========================================================================
    print_section('Test functions')
    error = (Y * (1 - p)).sum(axis=1).mean()

    f_test_keys = ['error', 'cost']
    f_test_vals = [error, base_cost]

    if l2_decay > 0.:
        f_test_keys.append('L2 cost')
        f_test_vals.append(l2_cost)
    f_test = theano.function([X, Y], f_test_vals)

     # ========================================================================
    print_section('Setting final tparams and save function')
    tparams, all_params = set_params(tparams, updates)

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())
        d.update(
            dim_in=dim_in,
            dim_out=dim_out,
            mlp=classifier
        )
        np.savez(outfile, **d)

    # ========================================================================
    print_section('Getting gradients and building optimizer.')
    f_grad_shared, f_grad_updates, learning_args = set_optimizer(
        inps, cost, tparams, constants, updates, extra_outs, **learning_args)

    # ========================================================================
    print_section('Actually running (main loop)')
    monitor = SimpleMonitor()

    main_loop(
        train, valid, tparams,
        f_grad_shared, f_grad_updates, f_test, f_test_keys,
        input_keys=input_keys,
        test_every=test_every,
        save=save,
        monitor=monitor,
        out_path=out_path,
        name=name,
        extra_outs_keys=extra_outs_keys,
        **learning_args)

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    exp_dict = set_experiment(args)

    train(**exp_dict)