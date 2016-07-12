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
_resolve = '@resolve'

def resolve_class(cell_type, classes=None):
    if classes is None:
        classes = _classes
    try:
        C = classes[cell_type]
    except KeyError:
        raise KeyError('Unexpected cell subclass `%s`, '
                       'available classes: %s' % (cell_type, classes.keys()))
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

        self.links = []
        self.classes = _classes
        self.tparams = {}

    @staticmethod
    def split_ref(ref):
        l = ref.split('.')
        cell_id = '.'.join(l[:-1])
        arg = l[-1]
        return cell_id, arg

    def reset(self):
        self.links = []
        self.tparams = {}
        self.cells = OrderedDict()
        self.cell_args = OrderedDict()

    def resolve_class(self, cell_type):
        return resolve_class(cell_type, self.classes)

    def match_args(self, cell_name, **kwargs):
        fail_on_mismatch = bool(cell_name in self.cells.keys())

        if fail_on_mismatch and (cell_name not in self.cell_args):
            raise KeyError('Cell args of %s not found but cell already set'
                           % cell_name)
        else:
            self.cell_args[cell_name] = {}

        args = self.cell_args[cell_name]
        for k, v in kwargs.iteritems():
            if k not in args.keys():
                if fail_on_mismatch:
                    raise KeyError('Requested key %s not found in %s and cell '
                                    'already exists.' % (k, cell_name))
                else:
                    args[k] = v
            if args[k] is not None and args[k] != v:
                raise ValueError('Key %s already set and differs from '
                                 'requested value (% vs %s)' % (k, args[k], v))

    def build(self, name=None):
        if name is not None and name not in self.cells:
            self.build_cell(name)
        else:
            for k, kwargs in self.cell_args.iteritems():
                if k not in self.cells:
                    self.build_cell(k)

    def build_cell(self, name):
        kwargs = self.cell_args[name]
        C = self.resolve_class(kwargs['cell_type'])
        self.resolve(name)

        for k in kwargs.keys():
            if isinstance(kwargs[k], CellManager.Link):
                C.get_link_value(kwargs[k], k, kwargs)
                kwargs.pop(k)

        C.factory(name=name, **kwargs)

    def prepare(self, cell_type, requestor=None, name=None, **kwargs):
        C = self.resolve_class(cell_type)

        if name is None and requestor is None:
            name = cell_type + '_cell'
        elif name is None:
            name = _p(requestor.name, cell_type)
        elif name is not None and requestor is not None:
            name = _p(requestor.name, name)

        self.match_args(name, cell_type=cell_type, **kwargs)

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

        self.cell_args[name] = cell_args

    def remove(self, key):
        del self.cells[key]
        del self.cell_args[key]

    def resolve(self, name):
        for link in self.links:
            if name in link.members:
                link.resolve()

    class Link(object):
        class Node(object):
            def __init__(self, name, C, key):
                self.name = name
                self.C = C
                self.key = key

        def __init__(self, cm, f, t):
            self.value = None
            self.nodes = {}
            self.cm = cm

            def split_arg(arg):
                s = arg.split('.')
                name = '.'.join(s[:-1])
                arg = s[-1]
                return name, arg

            f_name, f_key = split_arg(f)
            t_name, t_key = split_arg(t)

            f_class = self.cm.resolve_class(cm.cell_args[f_name]['cell_type'])
            t_class = self.cm.resolve_class(cm.cell_args[t_name]['cell_type'])

            self.nodes[f] = self.Node(f_name, f_class, f_key)
            self.nodes[t] = self.Node(t_name, t_class, t_key)
            self.members = [n.name for n in self.nodes.values()]

        def resolve(self):
            for node in self.nodes.values():
                kwargs = self.cm.cell_args[node.name]
                try:
                    self.value = node.C.set_link_value(node.key, **kwargs)
                except ValueError:
                    pass
            if self.value is None:
                raise ValueError

        def __repr__(self):
            if self.value is not None:
                return '%s' % self.value
            else:
                keys = self.nodes.keys()
                return ('<link>(%s, %s)' % (keys[0], keys[1]))

    def add_link(self, f, t):
        link = CellManager.Link(self, f, t)
        self.links.append(link)
        for node in link.nodes.values():
            if self.cell_args[node.name].get(node.key, None) is None:
                self.cell_args[node.name][node.key] = link

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
    _dim_map = {}       #
    _links = []

    def __init__(self, name='layer_proto', **kwargs):
        '''Init function for Cell.

        Args:
            name (str): name identifier of cell.

        '''
        kwargs = self.set_options(**kwargs)
        self.cell_manager = get_cell_manager()
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
            if not isinstance(value, CellManager.Link) and value is not None:
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
        self.cell_manager[self.name] = self

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
            self.cell_manager.prepare(name=k, requestor=self, **final_args)

        for f, t in self._links:
            f = _p(self.name, f)
            t = _p(self.name, t)
            self.cell_manager.add_link(f, t)

        for k, v in self._components.iteritems():
            self.cell_manager.build(_p(self.name, k))
            self.__dict__[k] = self.cell_manager[_p(self.name, k)]

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
                    self.cell_manager.tparams[name] = tp
                    self.__dict__[k].append(tp)
            else:
                name = _p(self.name, k)
                tp = theano.shared(pp, name=name)
                self.cell_manager.tparams[name] = tp
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