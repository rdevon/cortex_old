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

from cortex.utils import floatX, _rng
from cortex.utils.tools import warn_kwargs, _p


logger = logging.getLogger(__name__)

def resolve_class(cell_type):
    try:
        C = _classes[cell_type]
    except KeyError:
        raise KeyError('Unexpected cell subclass `%s`, '
                       'available classes: %s' % (cell_type, _classes.keys()))
    return C

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

def get_cell_manager():
    if CellManager._instance is None:
        return CellManager()
    else:
        return CellManager._instance


class CellManager(object):
    '''cortex Cell manager.

    Ensures that connected objects have the right dimensionality as well as
        manages passing the correct tensors as input and cost.

    '''
    _instance = None

    def __init__(self):
        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        if CellManager._instance is not None:
            logger.warn('New `CellManager` instance. Old one will be lost.')
        CellManager._instance = self
        self.cells = OrderedDict()
        self.cell_args = OrderedDict()
        self.cell_classes = OrderedDict()

        self.links = []
        self.class_maps = _classes
        self.tparams = {}

    @staticmethod
    def split_ref(ref):
        l = ref.split('.')
        cell_id = '.'.join(l[:-1])
        arg = l[-1]
        return cell_id, arg

    def add_class(self, key, value):
        self.class_maps[key] = value

    def register(self, name=None, cell_type=None, **layer_args):
        if name is None:
            name = cell_type
        if name in self.cells.keys():
            self.logger.warn(
                'Cell with name `%s` already found: overwriting. '
                'Use `cortex.cell_manager.remove` to avoid this warning' % key)
        try:
            self.cell_classes = self.classes[cell_type]
        except KeyError:
            raise TypeError('`cell_type` must be provided. Got %s. Available: '
                            '%s' % (cell_type, self.classes))

        if name in self.cell_args.keys():
            self.cell_args[name].update(**cell_args)
        else:
            self.cell_args[name] = cell_args

    def remove(self, key):
        del self.cells[key]
        del self.cell_args[key]

    def build(self, key=None):
        if key is not None and key not in self.cells:
            build_cell(**self.cell_args[key])
        for key, kwargs in self.cell_args.iteritems():
            if key not in self.cells:
                CellManager.build_cell(**kwargs)

    def build_cell(cell_type=None, **kwargs):
        if cell_type is None:
            raise TypeError('Argument `cell_type` not provided.')
        try:
            factory = self.classes[cell_type].factory
        except KeyError:
            raise KeyError('cell class %s not found. Available: %s'
                           % (cell_type, self.classes))
        factory(**kwargs)

    def __getitem__(self, key):
        return self.cells[key]

    def __setitem__(self, key, cell):
        if key in self.cells.keys():
            self.logger.warn(
                'Cell with name `%s` already found: overwriting. '
                'Use `cortex.cell_manager.remove` to avoid this warning' % key)
        if isinstance(cell, Cell):
            self.cells[key] = cell
            self.cell_args[key] = cell.get_args()
        else:
            raise TypeError('`cell` must be of type %s, got %s'
                            % (Cell, type(cell)))

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
    _arg_map = {}       # Map of tensor variables to parameters.
    _decay_params = []  # Parameters subject to decay.

    def __init__(self, name='layer_proto', **kwargs):
        '''Init function for Cell.

        Args:
            name (str): name identifier of cell.

        '''
        kwargs = self.set_options(**kwargs)
        self.cell_manager = get_cell_manager()
        self.noise_switch = get_noise_switch()
        self.name = name
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
    def factory(C, **kwargs):
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
        self.cell_manager[self.name] = self

    def set_components(self, **kwargs):
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

    def get_component(self, name, cell_type=None, **kwargs):
        name = self.name + '.' + name
        try:
            cell = self.cell_manager[name]
        except KeyError:
            if cell_type is None:
                self.cell_manager.cells[name] = None
            else:
                if name in self._cell_manager.cell_args.keys():
                    kwargs.update(self.layer_args[name])
                self.set_links(name)
                self.layers[name] = build_layer(layer_type, name=name, **kwargs)
            cell = self._cell_manager

        return cell

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

    def get_decay_params():
        '''Return parameters used in L1 and L2 decay.

        Returns:
            OrderedDict: dictionary of parameters.

        '''
        params = dict((k, self.__dict__[k]) for k in self.params.keys())
        decay_params = {}
        for k in self._decay_params:
            p = self.__dict__[k]
            if isinstance(p, list):
                for i in range(len(p)):
                    kk = '%s[%d]' % (k, i)
                    name = _p(self.name, kk)
                    decay_params[name] = self.cell_manager.tparams[name]
            else:
                name = _p(self.name, k)
                decay_params[name] = self.cell_manager.tparams[name]

        return decay_params

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
                    self.cell_manager.tparams[name] = tp
                    self.__dict__[k].append(tp)
            else:
                name = _p(self.name, k)
                tp = theano.shared(pp, name=name)
                self.cell_manager.tparams[name] = tp
                self.__dict__[k] = tp

    def l1_decay(self, rate, **kwargs):
        '''L1 decay.

        Args:
            rate (float): decay rate.
            kwargs: keyword arguments of parameter name and rate.

        Returns:
            dict: dictionary of l1 decay costs for each parameter.

        '''
        decay_params = self.get_decay_params()

        cost = T.constant(0.).astype(floatX)
        rval = OrderedDict()
        if rate <= 0:
            return rval

        for k, v in decay_params.iteritems():
            if k in kwargs.keys():
                r = kwargs[k]
            else:
                r = rate
            self.logger.debug('Adding %.4g L1 decay to parameter %s' % (r, k))
            p_cost = r * (abs(v)).sum()
            rval[k + '_l1_cost'] = p_cost
            cost += p_cost

        rval = OrderedDict(
            cost = cost
        )

        return rval

    def l2_decay(self, rate, **kwargs):
        '''L2 decay.

        Args:
            rate (float): decay rate.
            kwargs: keyword arguments of parameter name and rate.

        Returns:
            dict: dictionary of l2 decay costs for each parameter.

        '''
        decay_params = self.get_decay_params()

        cost = T.constant(0.).astype(floatX)
        rval = OrderedDict()
        if rate <= 0:
            return rval
        for k, v in decay_params.iteritems():
            if k in kwargs.keys():
                r = kwargs[k]
            else:
                r = rate
            self.logger.debug('Adding %.4g L2 decay to parameter %s' % (r, k))
            p_cost = r * (v ** 2).sum()
            rval[k + '_l2_cost'] = p_cost
            cost += p_cost

        rval = OrderedDict(
            cost = cost
        )

        return rval

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