'''MLP with a distribution on top.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from .. import distributions
from .. import Cell
from ...utils import concatenate, floatX


class DistributionMLP(Cell):
    '''MLP with a distribution as the output.

    Attributes:
        distribution (Distribution): distribution of output. Used for
            sampling, density calculations, etc.
        mlp (MLP): multi-layer perceptron. Used for feed-forward operation.

    '''

    _required = ['distribution_type']
    _components = {
        'mlp': {
            'cell_type': 'MLP',
            '_required': {'out_act': 'identity'},
            '_passed': ['dim_h', 'n_layers', 'dropout', 'weight_noise',
                        'h_act', 'dim_hs', 'batch_normalization']
        },
        'distribution': {
            'cell_type': '&distribution_type',
            '_required': {'conditional': True},
            '_passed': ['has_kl', 'neg_log_prob', 'kl_divergence',
                        'simple_sample']
        },
    }
    _links = [('mlp.output', 'distribution.input')]
    _dim_map = {
        'input': 'dim_in',
        'output': 'dim',
        'P': 'dim',
        'samples': 'dim',
        'X': 'dim_in',
        'Y': 'dim',
        'Z': 'dim',
    }
    _dist_map = {'P': 'distribution_type'}
    _costs = {
        'nll': '_cost',
        'negative_log_likelihood': '_cost'
    }
    _sample_tensors = ['P']

    def __init__(self, distribution_type, name=None, **kwargs):
        if name is None:
            name = '%s_%s' % (distribution_type, 'MLP')
        self.distribution_type = distribution_type
        super(DistributionMLP, self).__init__(name=name, **kwargs)

    @classmethod
    def set_link_value(C, key, distribution_type=None, **kwargs):
        from ... import _manager as manager

        if key != 'output':
            return super(DistributionMLP, C).set_link_value(
                key, distribution_type=distribution_type, **kwargs)

        if distribution_type is None:
            raise ValueError
        DC = manager.resolve_class(distribution_type)
        return DC.set_link_value(key, dim=dim, **kwargs)

    @classmethod
    def get_link_value(C, link, key):
        from ... import _manager as manager

        if key != 'output':
            return super(DistributionMLP, C).get_link_value(link, key)
        if link.distribution is None:
            raise ValueError
        DC = manager.resolve_class(link.distribution)
        return DC.get_link_value(link, key)

    def _feed(self, X, *params):
        inps = self.mlp.init_args(X)
        outs = self.mlp._feed(*(inps + params))
        Y = outs['output']
        outs['output'] = self.distribution(Y)
        outs['P'] = outs['output']
        return outs

    def _cost(self, X=None, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[self.name + '.' + 'P']
        return self.distribution._cost(X=X, P=P)

    def generate_random_variables(self, shape, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[self.name + '.' + 'P']

        return self.distribution.generate_random_variables(shape, P=P)

    def _sample(self, epsilon, P=None):
        session = self.manager.get_session()
        if P is None:
            if _p(self.name, 'P') not in session.tensors.keys():
                raise TypeError('%s.P not found in graph nor provided'
                                % self.name)
            P = session.tensors[_p(self.name, 'P')]
        return self.distribution._sample(epsilon, P=P)

    def viz(self, mean=None, perm=None):
            params = self.get_params()
            P0 = self._feed(mean, *params)['output']
            P = self._feed(perm, *params)['output']
            return self.distribution.viz(P0, P=P)


_classes = {'DistributionMLP': DistributionMLP}