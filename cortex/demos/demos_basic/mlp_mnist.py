"""
Demo for classification of MNIST data.

Try with 'cortex-run mlp_mnist.py <optional .yamal>'
"""

from collections import OrderedDict
import numpy as np
from os import path
import theano
from theano import tensor as T

from cortex.datasets import resolve as resolve_dataset
from cortex.models.mlp import MLP
from cortex.utils import floatX

# Default arguments
_learning_args = dict(
    learning_rate=0.01,
    learning_rate_scheduler=None,
    optimizer='sgd',
    optimizer_args=dict(),
    epochs=100,
    valid_key='error',
    valid_sign='+',
    excludes=[]
)

_dataset_args = dict(
    train_batch_size=100,
    valid_batch_size=100,
    dataset='mnist',
    distribution='multinomial',
    source='$data/basic/mnist.pkl.gz'
)

_model_args = dict(
    dim_h=None,
    l2_decay=None,
)

mlp_args = dict(
    dim_hs=[200, 100],
    n_layers=None,
    h_act='T.nnet.sigmoid',
    input_layer='mnist',
    output='label',
    dropout=None
)

extra_arg_keys = ['mlp_args']


def _build(module):
    models = OrderedDict()
    dataset = module.dataset
    mlp_args = module.mlp_args
    dim_in = dataset.dims[dataset.name]
    dim_out = dataset.dims['label']
    distribution = dataset.distributions[mlp_args['output']]

    model = MLP.factory(dim_in=dim_in, dim_out=dim_out, distribution=distribution, **mlp_args)

    models['mlp'] = model
    return models


def _cost(module):
    models = module.models

    X = module.inputs[module.dataset.name]
    Y = module.inputs[module.mlp_args['output']]
    used_inputs = [module.dataset.name, module.mlp_args['output']]

    model = models['mlp']
    outputs = model(X)

    results = OrderedDict()
    p = outputs['p']
    base_cost = model.neg_log_prob(Y, p).sum(axis=0)
    cost = base_cost

    updates = theano.OrderedUpdates()
    constants = []

    l2_decay = module.l2_decay
    if l2_decay is not False and l2_decay > 0.:
        module.logger.info('Adding %.5f L2 weight decay' % l2_decay)
        l2_rval = model.l2_decay(l2_decay)
        l2_cost = l2_rval.pop('cost')
        cost += l2_cost
        results['l2_cost'] = l2_cost

    results['error'] = (Y * (1 - p)).sum(axis=1).mean()
    results['-sum log p(x | y)'] = base_cost
    results['cost'] = cost

    return used_inputs, results, updates, constants, outputs
