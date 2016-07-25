'''Module for cortex manager.

'''

from collections import OrderedDict
import logging

from ..utils.tools import _p


def get_manager():
    if Manager._instance is None:
        return Manager()
    else:
        return Manager._instance

def split_arg(arg, idx=-1):
    s = arg.split('.')
    name = '.'.join(s[:idx])
    arg = '.'.join(s[idx:])
    return name, arg

def resolve_tensor_arg(arg, manager=None):
    if manager is None: manager = get_manager()
    if not is_tensor_arg(arg):
        raise TypeError('Arg %s is not a tensor argument.' % arg)
    cell_args = manager.cell_args

    if arg.split('.')[0] in manager.datasets.keys():
        name_, key = split_arg(arg)
        C = None
    elif '.'.join(arg.split('.')[:-1]) in cell_args.keys():
        name_, key = split_arg(arg)
        C = manager.resolve_class(cell_args[name_]['cell_type'])
    else:
        raise KeyError('Cell or data %s not found' % arg)
    return name_, key, C

def is_tensor_arg(arg):
    return (isinstance(arg, str) and '.' in arg)


class Manager(object):
    '''cortex manager.

    Ensures that connected objects have the right dimensionality as well as
        manages passing the correct tensors as input and cost.

    '''
    _instance = None
    _current_session = None

    def __init__(self):
        from ..models import _classes
        from ..datasets import _classes as _dataset_classes
        from ..costs import _costs

        if Manager._instance is not None:
            logger.warn('New `Manager` instance. Old one will be lost.')
        Manager._instance = self

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.classes = _classes
        self.dataset_classes = _dataset_classes
        self.cost_functions = _costs
        self.reset()

    # General methods
    def add_cell_class(name, C):
        self.classes[name] = C

    def add_dataset_class(name, C):
        self.dataset_classes[name] = C

    def add_cost_function(name, f):
        self.cost_functions[name] = f

    @staticmethod
    def split_ref(ref):
        l = ref.split('.')
        cell_id = '.'.join(l[:-1])
        arg = l[-1]
        return cell_id, arg

    def reset(self):
        self.cells = OrderedDict()
        self.cell_args = OrderedDict()
        self.links = []
        self.datasets = {}
        self.steps = []
        self.costs = []
        self.tparams = {}
        self.samples = {}
        self.reset_sessions()

    def resolve_class(self, cell_type):
        from .. import resolve_class
        return resolve_class(cell_type, self.classes)

    def build(self):
        for k, kwargs in self.cell_args.iteritems():
            if k not in self.cells:
                self.build_cell(k)

    def create_session(self):
        from .session import Session
        self._current_session = Session()

    def build_session(self, test=False):
        if self._current_session is None:
            self.create_session()
        self._current_session.build(test=test)
        return self._current_session

    def get_session(self):
        return self._current_session

    def reset_sessions(self):
        from .session import Session
        Session._reset()
        self._current_session = None

    # Data methods
    def prepare_data(self, dataset, **kwargs):
        from .. import resolve_class
        C = resolve_class(dataset, self.dataset_classes)
        C(**kwargs)

    # Cell methods
    def build_cell(self, name):
        from .link import Link
        kwargs = self.cell_args[name]
        C = self.resolve_class(kwargs['cell_type'])

        for k in kwargs.keys():
            if isinstance(kwargs[k], Link):
                link = kwargs[k]
                value = link.query(name, k)
                kwargs[k] = value

        C.factory(name=name, **kwargs)

    def prepare_cell(self, cell_type, requestor=None, name=None, **kwargs):
        C = self.resolve_class(cell_type)

        if name is None and requestor is None:
            name = cell_type + '_cell'
        elif name is None:
            name = _p(requestor.name, cell_type)
        elif name is not None and requestor is not None:
            name = _p(requestor.name, name)

        self.match_args(name, cell_type=cell_type, **kwargs)

    def register_cell(self, name=None, cell_type=None, **layer_args):
        if name is None:
            name = cell_type
        if name in self.cells.keys():
            self.logger.warn(
                'Cell with name `%s` already found: overwriting. '
                'Use `cortex.manager.remove_cell` to avoid this warning' % key)
        try:
            self.cell_classes = self.classes[cell_type]
        except KeyError:
            raise TypeError('`cell_type` must be provided. Got %s. Available: '
                            '%s' % (cell_type, self.classes))

        self.cell_args[name] = cell_args

    def remove_cell(self, key):
        del self.cells[key]
        del self.cell_args[key]

    # Sample methods
    def prepare_samples(self, arg, shape, name='samples'):
        if isinstance(shape, int):
            shape = (shape,)

        if arg in self.cell_args.keys():
            cell_name = arg
            key = None
        elif is_tensor_arg(arg):
            cell_name, key, _ = resolve_tensor_arg(arg)

        if cell_name not in self.cell_args.keys():
            raise KeyError('Cell %s not found' % cell_name)

        C = self.resolve_class(self.cell_args[cell_name]['cell_type'])
        if not hasattr(C, '_sample'):
            raise ValueError('Cell type %s does not support sampling'
                             % C.__name__)

        if key is not None and key not in C._sample_tensors:
            raise KeyError('Cell %s does not support sample tensor %s'
                           % (cell_name, key))

        name = _p(cell_name, name)
        if key is not None:
            key = arg

        if name in self.samples.keys():
            self.logger.warn('Overwriting samples %s' % name)
        self.samples[name] = {'cell_name': cell_name, 'P': key, 'shape': shape}

    # Methods for building graph
    def test_op_args(self, op, args, kwargs):
        if hasattr(op, '__call__'):
            pass
        else:
            raise TypeError('Op must be callable '
                            'or a string of form `cell_name` or '
                            '`cell_name.op`')

        for arg in list(args) + kwargs.values():
            if is_tensor_arg(arg):
                n, k, C = resolve_tensor_arg(arg)
                if n in self.datasets.keys():
                    if not k in self.datasets[n]['dims'].keys():
                        raise KeyError('Dataset %s has no key `%s`' % (n, k))
                elif n in self.cell_args.keys():
                    pass
                else:
                    raise KeyError('No cell nor dataset with name `%s`' % n)

    def add_step(self, op, *args, **kwargs):
        if isinstance(op, str):
            op_s = op.split('.')
            if len(op_s) == 1:
                cell_name = op_s[0]
                op_name = None
            elif len(op_s) == 2:
                cell_name, op_name = op_s
            else:
                raise TypeError('Op must be callable '
                                'or a string of form `cell_name` or '
                                '`cell_name.op`')
            if cell_name in self.cell_args.keys():
                cell_type = self.cell_args[cell_name]['cell_type']
                C = self.resolve_class(cell_type)
                if op_name is None:
                    op_name = '__call__'
                if hasattr(C, op_name):
                    op = getattr(C, op_name)
                else:
                    raise TypeError('Cell %s of type %s has no method %s.'
                                    % (cell_name, C, op_name))
                if op_name == '__call__':
                    arg_keys = C._call_args
                else:
                    arg_keys = getattr(C, '_%s_args' % op_name)
                if len(args) != len(arg_keys):
                    raise TypeError('%d operation (%s) args provided, but %d '
                                    'arg_keys available. (%s given, %s needed)'
                                    % (len(args), C, len(arg_keys),
                                       args, arg_keys))

                for arg, key in zip(args, arg_keys):
                    if (key in C._dim_map.keys() and is_tensor_arg(arg)):
                        self.match_dims(arg, '.'.join([cell_name, key]))
        else:
            cell_name = None

        self.test_op_args(op, args, kwargs)

        self.steps.append(dict(
            cell_name=cell_name,
            op=op,
            args=args,
            kwargs=kwargs))

    def add_cost(self, op, *args, **kwargs):
        cell_name = None
        if isinstance(op, str):
            if is_tensor_arg(op):
                cell_name, cost_name, _ = resolve_tensor_arg(op)
                cell_type = self.cell_args[cell_name]['cell_type']
                C = self.resolve_class(cell_type)
                if cost_name == 'cost' and hasattr(C, '_cost'):
                    op = getattr(C, '_cost')
                else:
                    if not cost_name in C._costs.keys():
                        raise AttributeError('cell type %s for cell `%s` has no '
                                             'cost %s' % (C, cell_name, cost_name))
                    op = getattr(C, C._costs[cost_name])
            else:
                if op not in self.cost_functions.keys():
                    raise TypeError('Cost function `%s` not found.' % op)
                op = self.cost_functions[op]

        self.test_op_args(op, args, kwargs)
        self.costs.append(dict(
            cell_name=cell_name,
            op=op,
            args=args,
            kwargs=kwargs))

    # Methods for linking and dim matching
    def resolve_links(self, name):
        self.logger.debug('Resolving %s' % name)
        for link in self.links:
            if name in link.members:
                self.logger.debug('Resolving link %s' % link)
                link.resolve()

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

    def match_dims(self, f, t):
        from .link import Link
        link = Link(f, t)
        for name, node in link.nodes.iteritems():
            if name in self.datasets.keys():
                pass
            else:
                if (self.cell_args[name].get(node.dim_key, None) is None
                    and node.C._dim_map.get(node.link_key, None) is not None):
                    self.cell_args[name][node.dim_key] = link

                if (self.cell_args[name].get(node.dist_key, None) is None
                    and node.C._dist_map.get(node.link_key, None) is not None):
                    self.cell_args[name][node.dist_key] = link

    # Misc
    def __getitem__(self, key):
        return self.cells[key]

    def __setitem__(self, key, cell):
        from ..models import Cell

        if key in self.cells.keys():
            self.logger.warn(
                'Cell with name `%s` already found: overwriting. '
                'Use `cortex.manager.remove_cell` to avoid this warning' % key)
        if isinstance(cell, Cell):
            self.cells[key] = cell
            self.cell_args[key] = cell.get_args()
        else:
            raise TypeError('`cell` must be of type %s, got %s'
                            % (Cell, type(cell)))