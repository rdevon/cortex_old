'''Base Cell class.

'''

from collections import OrderedDict
import copy
import logging
import numpy as np
from pprint import pprint
import random
import re
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .. import get_manager, Link
from ..utils import floatX, _rng
from ..utils.tools import warn_kwargs, _p


logger = logging.getLogger(__name__)


def init_rngs(cell):
    '''Initialization function for RNGs.

    Args:
        cell (Cell).

    '''
    cell.rng = _rng
    cell.trng = RandomStreams(random.randint(1, 10000))

def ortho_weight(ndim, rng=None):
    '''Make ortho weight tensor.

    '''
    if not rng:
        rng = _rng
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    '''Make normal weight tensor.

    '''
    if not rng:
        rng = _rng
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype(floatX)

class NoiseSwitch(object):
    '''Object to control noise of model.

    '''
    _instance = None
    def __init__(self):
        if NoiseSwitch._instance is None:
            NoiseSwitch._instance = self
        self.noise = True

    def switch(self, switch_to):
        ons = ['on', 'On', 'ON']
        offs = ['off', 'Off', 'OFF']
        if switch_to in ons:
            self.noise = True
        elif switch_to in offs:
            self.noise = False

    def __call__(self):
        return self.noise

def get_noise_switch():
    if NoiseSwitch._instance is None:
        return NoiseSwitch()
    else:
        return NoiseSwitch._instance


class Cell(object):
    '''Base class for all models.

    Attributes:
        name (str): name identifier of cell.
        params (dict): dictionary of numpy.arrays
        learn (bool): if False, do not change params.
        n_params (int): number of parameters

    '''
    _components = {}    # Cells that this cell controls.
    _options = {}       # Dictionary of optional arguments and default values
    _required = []      # Required arguments for __init__
    _args = ['name']    # Arguments necessary to uniquely id the cell. Used for
                        #   save.
    _dim_map = {}       #
    _links = []

    def __init__(self, name='layer_proto', **kwargs):
        '''Init function for Cell.

        Args:
            name (str): name identifier of cell.

        '''
        kwargs = self.set_options(**kwargs)
        self.manager = get_manager()
        self.noise_switch = get_noise_switch()
        self.name = name
        self.passed = {}
        init_rngs(self)

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.logger.debug(
            'Forming model cell %r with name %s' % (self.__class__, name))
        self.logger.debug('Formation parameters: %s' % self.get_args())

        kwargs = self.set_components(**kwargs)
        self.init_params(**kwargs)
        self.logger.debug('Paramters and shapes: %s' % self.profile_params())
        self.register()

        self.n_params = 0
        for param in self.params:
            if isinstance(param, list):
                self.n_params += len(param)
            else:
                self.n_params += 1
        self.set_tparams()

    def set_options(self, **kwargs):
        for k, v in self._options.iteritems():
            self.__dict__[k] = kwargs.pop(k, v)
        return kwargs

    def profile_params(self):
        d = OrderedDict()
        for k, p in self.params.iteritems():
            if isinstance(p, list):
                for i, pp in enumerate(p):
                    d['%s[%d]' % (k, i)] = pp.shape
            else:
                d[k] = p.shape
        return d

    @classmethod
    def set_link_value(C, key, **kwargs):
        if key in C._dim_map.keys():
            value = kwargs.get(key, None)
            if not isinstance(value, Link) and value is not None:
                return value
            else:
                raise ValueError
        else:
            raise KeyError

    @classmethod
    def get_link_value(C, link, key, kwargs):
        if key in C._dim_map.keys():
            if link.value is None:
                raise ValueError
            else:
                kwargs[C._dim_map[key]] = link.value
        else:
            raise KeyError

    @classmethod
    def factory(C, cell_type=None, **kwargs):
        '''Cell factory.

        Convenience function for building Cells.

        Args:
            **kwargs: construction keyword arguments.

        Returns:
            C

        '''
        reqs = OrderedDict(
            (k, v) for k, v in kwargs.iteritems() if k in C._required)
        options = dict((k, v) for k, v in kwargs.iteritems() if not k in C._required)

        for req in C._required:
            if req not in reqs.keys():
                raise TypeError('Required argument %s not provided for '
                                'constructor of %s' % (req, C))

        return C(*reqs.values(), **options)

    def register(self):
        self.manager[self.name] = self

    def set_components(self, **kwargs):
        from ..utils.tools import _p

        for k, v in self._components.iteritems():
            args = {}
            args.update(**v)
            passed = args.pop('_passed', dict())
            for p in passed:
                self.passed[p] = k
            required = args.pop('_required', dict())
            args.update(**required)
            passed_args = dict((kk, kwargs[kk])
                for kk in passed
                if kk in kwargs.keys())
            kwargs = dict((kk, kwargs[kk]) for kk in kwargs.keys() if kk not in passed)
            args.update(**passed_args)
            final_args = {}
            for kk, vv in args.iteritems():
                if isinstance(vv, str) and vv.startswith('&'):
                    final_args[kk] = self.__dict__[vv[1:]]
                else:
                    final_args[kk] = vv
            self.manager.prepare(name=k, requestor=self, **final_args)

        for f, t in self._links:
            f = _p(self.name, f)
            t = _p(self.name, t)
            self.manager.add_link(f, t)

        for k, v in self._components.iteritems():
            self.manager.build(_p(self.name, k))
            self.__dict__[k] = self.manager[_p(self.name, k)]

        return kwargs

    def copy(self):
        '''Copy the cell.

        '''
        return copy.deepcopy(self)

    def init_params(self, **init_kwargs):
        '''Initialize the parameters.

        '''
        self.params = OrderedDict()

    def get_args(self):
        return dict((k, self.__dict__[k]) for k in self._args)

    def feed(self, *args):
        '''Basic feed method.

        This is the identity graph. Generally it is `scan` safe.

        Args:
            args (list): list of tensor inputs.

        Returns:
            OrderedDict: theano tensor variables.

        '''
        return OrderedDict(('X_%d' % i, args[i]) for i in range(len(args)))

    def __call__(self, *args, **kwargs):
        '''Call function.

        Args:
            args (list): list of tensor inputs.

        Returns:
            OrderedDict: theano tensor variables.

        '''
        params = tuple(self.get_params())
        return self.feed(*(args + params))

    def get_components(self):
        '''Gets cell components.

        '''
        components = []
        for k in self._components:
            component = getattr(self, k)
            if isinstance(component, list):
                components += [c for c in component if c is not None]
            elif component is not None:
                components.append(component)

        c_components = []
        for component in components:
            c_components += component.get_components()
        components += c_components
        return components

    def set_tparams(self):
        '''Sets the tensor parameters.

        '''
        if self.params is None:
            raise ValueError('Params not set yet')
        tparams = OrderedDict()

        for k, p in self.params.iteritems():
            if isinstance(p, list):
                self.__dict__[k] = []
                for i, pp in enumerate(p):
                    kk = '%s[%d]' % (k, i)
                    name = _p(self.name, kk)
                    tp = theano.shared(pp, name=name)
                    self.manager.tparams[name] = tp
                    self.__dict__[k].append(tp)
            else:
                name = _p(self.name, k)
                tp = theano.shared(pp, name=name)
                self.manager.tparams[name] = tp
                self.__dict__[k] = tp

    def help(self):
        pprint(self._help)

    @classmethod
    def get_arg_reference(C, key, kwargs):
        try:
            k = C._arg_map[key]
        except KeyError:
            raise KeyError('cell %s has no argument %s. Available arguments: '
                           '%s' % (C, key, C._arg_map))
        return kwargs[k]

    def __getattr__(self, key):
        if hasattr(self, key) and key == 'passed':
            return object.__getattr__(self, key)
        if key in self.passed:
            return self.__dict__[self.passed[key]].__getattribute__(key)
        return object.__getattr__(self, key)

    def __str__(self):
        attributes = self.__dict__
        attributes['params'] = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,))
            for k, a in attributes['params'].iteritems())
        for k in ['trng', 'rng', 'logger']:
            attributes.pop(k, None)

        attr_str = ''
        for k, a in attributes.iteritems():
            if k in self._components:
                c_str = a.__str__()
                new_str = '\n\t'
                for i in range(0, len(c_str) - 1):
                    new_str += c_str[i]
                    if c_str[i] != '\t' and c_str[i + 1] == '\t':
                        new_str += '\t'
                new_str += c_str[-1]
            else:
                new_str = '\n\t%s: %s' % (k, a)
            attr_str += new_str

        s = ('<Cell %s: %s>' % (self.__class__.__name__, attr_str))
        return s

_classes = {'Cell': Cell}
from . import mlp, distributions
_modules = [mlp, distributions]
for module in _modules: _classes.update(**module._classes)