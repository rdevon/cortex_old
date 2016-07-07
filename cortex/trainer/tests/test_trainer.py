'''Unit tests for trainer.

'''

from cortex.layers.tests import test_mlp
from cortex.trainer import layer_manager


def test_layer_manager():
    mlp_args = {
        'dim_in': 11,
        'dim_out': 17,
        'dim_h': 23,
        'h_act': 'T.nnet.softplus',
        'distribution': 'binomial'
    }

    lm = layer_manager.LayerManager(mlp=mlp_args)
    assert False, lm.layer_kwargs