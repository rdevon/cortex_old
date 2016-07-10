'''Unit tests for trainer.

'''

import cortex
from cortex.layers.tests import test_mlp
from cortex import layers

layer_manager = cortex.layer_manager

def test_layer_manager_simple():
    mlp_args = {
        'layer_type': 'MLP',
        'dim_in': 11,
        'dim_out': 17,
        'dim_h': 23,
        'h_act': 'T.nnet.softplus',
        'distribution': 'binomial'
    }

    layer_manager.add('mlp', **mlp_args)
    for k, v in mlp_args.iteritems():
        if k == 'layer_type':
            continue
        if not k in layer_manager.layers['mlp']._args:
            continue
        assert layer_manager.layer_args['mlp'][k] == v
        assert layer_manager.layers['mlp'].__dict__[k] == v

def test_links():
    mlp1_args = {
        'layer_type': 'MLP',
        'dim_in': 11,
        'dim_out': 17,
        'dim_h': 23,
        'h_act': 'T.nnet.softplus',
        'distribution': 'binomial'
    }

    mlp2_args = {
        'layer_type': 'MLP',
        'dim_out': 17,
        'dim_h': 23,
        'h_act': 'T.tanh',
        'distribution': 'gaussian'
    }

    layer_manager.add('mlp1', **mlp1_args)
    layer_manager('mlp2', **mlp2_args)