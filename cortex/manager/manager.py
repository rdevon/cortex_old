'''Module for cortex manager.

'''

from collections import OrderedDict
import logging

from .. import costs
from .. import datasets
from .. import models
from ..utils.tools import _p, print_section
from ..training import Evaluator, Trainer, Visualizer
from ..training.monitor import BasicMonitor


def get_manager():
    if Manager._instance is None:
        return Manager()
    else:
        return Manager._instance

def _resolve_class(cell_type, classes):
    try:
        C = classes[cell_type]
    except KeyError:
        raise KeyError('Unexpected cell subclass `%s`, '
                       'available classes: %s' % (cell_type, classes.keys()))
    return C

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
    elif '.'.join(arg.split('.')[:-1]) in manager.nodes.keys():
        name_, key = split_arg(arg)
        C = manager.nodes[name_]
    elif '.'.join(arg.split('.')[:-1]) in cell_args.keys():
        name_, key = split_arg(arg)
        C = manager.resolve_class(cell_args[name_]['cell_type'])
    elif '.'.join(arg.split('.')[:-1]) in _ops.keys():
        name_, key = split_arg(arg)
        C = None
    else:
        raise KeyError('Cell or data %s not found' % arg)
    return name_, key, C

def is_tensor_arg(arg):
    manager = get_manager()
    if arg in manager.nodes.keys():
        return True
    if not isinstance(arg, str) or '.' not in arg:
        return False
    if '/' in arg:
        return False
    return True


class Manager(object):
    '''cortex manager.

    Ensures that connected objects have the right dimensionality as well as
        manages passing the correct tensors as input and cost.

    '''
    _instance = None
    _current_session = None

    def __init__(self):
        if Manager._instance is not None:
            logger.warn('New `Manager` instance. Old one will be lost.')
        Manager._instance = self

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))

        self.classes = models._classes
        self.dataset_classes = datasets._classes
        self.cost_functions = costs._costs
        self.stat_functions = costs._stats
        self.ops = _ops

        self.reset()

    # General methods
    def add_cell_class(name, C):
        self.classes[name] = C

    def add_dataset_class(name, C):
        self.dataset_classes[name] = C

    def add_cost_function(name, f):
        self.cost_functions[name] = f

    def add_stat_function(name, f):
        self.stat_functions[name] = f

    @staticmethod
    def split_ref(ref):
        l = ref.split('.')
        cell_id = '.'.join(l[:-1])
        arg = l[-1]
        return cell_id, arg

    def reset(self):
        self.cells = OrderedDict()
        self.cell_args = OrderedDict()
        self.datasets = {}
        self.nodes = {}
        self.links = []

        self.steps = []
        self.costs = {}
        self.stats = {}
        self.samples = {}

        self.tparams = {}
        self.reset_sessions()
        self.trainer = None
        self.tester = None
        self.visualizer = None

    def resolve_class(self, cell_type, classes=None):
        if classes is None:
            classes = self.classes
        return _resolve_class(cell_type, self.classes)

    def build(self):
        for k, kwargs in self.cell_args.iteritems():
            if k not in self.cells:
                self.build_cell(k)

    # Sessions -----------------------------------------------------------------
    def create_session(self, noise=True, batch_size=None):
        from .session import Session
        session = Session(noise=noise, batch_size=batch_size)
        self._current_session = session
        return session

    def build_session(self, test=False, idx=None):
        if idx is not None:
            session = self.get_session(idx=idx)
        elif self._current_session is None:
            session = self.create_session()
        else:
            session = self._current_session
        session.build(test=test)

    def get_session(self, idx=None):
        from .session import Session
        if idx is not None:
            session = Session.sessions[idx]
        else:
            session = self._current_session
        return session

    def reset_sessions(self):
        from .session import Session
        Session._reset()
        self._current_session = None

    # Main objects -------------------------------------------------------------
    def setup_trainer(self, session, **kwargs):
        self.trainer = Trainer(session, **kwargs)
        return self.trainer

    def setup_evaluator(self, session, **kwargs):
        self.evaluator = Evaluator(session, **kwargs)
        return self.evaluator

    def setup_monitor(self, session, **kwargs):
        self.monitor = BasicMonitor(**kwargs)
        self.monitor.add_section('cost', keys=['total_cost']+session.costs.keys())
        self.monitor.add_section('stats', keys=session.stats.keys())
        return self.monitor

    def setup_visualizer(self, session, **kwargs):
        self.visualizer = Visualizer(session, **kwargs)
        return self.visualizer

    def train(self, eval_modes=None, validation_mode=None, eval_every=10,
              monitor_grads=False):
        if eval_modes is None: eval_modes=['train', 'valid']
        if validation_mode is None: validation_mode = 'valid'
        if len(self.trainer.f_grads) == 0:
            self.trainer.set_optimizer()
        if monitor_grads:
            self.monitor.add_section(
                'Grads', ['_grad_' + k for k in self.trainer.tparams])

        try:
            while True:
                br = False

                for mode in eval_modes:
                    r = self.evaluator(data_mode=mode)
                    if mode == validation_mode:
                        self.evaluator.validate(r, self.trainer.epoch)
                    self.monitor.update(mode, **r)

                self.monitor.display()
                if self.visualizer is not None:
                    self.visualizer()

                try:
                    grads = self.trainer.next_epoch(n_epochs=eval_every)
                    if monitor_grads:
                        self.monitor.update('train', **grads)
                except StopIteration:
                    br = True

                if br:
                    break
        except KeyboardInterrupt:
            print 'Interrupting training...'

    # Data methods -------------------------------------------------------------
    def prepare_data(self, dataset, **kwargs):
        C = _resolve_class(dataset, self.dataset_classes)
        C(**kwargs)

    def prepare_data_split(self, dataset, **kwargs):
        C = _resolve_class(dataset, self.dataset_classes)
        if not hasattr(C, 'factory'):
            raise TypeError('Dataset class `%s` needs a factory to be split '
                            'automatically.')
        C.factory(**kwargs)

    # Cell methods -------------------------------------------------------------
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

    # Methods for building graph -----------------------------------------------
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
                elif n in self.nodes.keys():
                    pass
                else:
                    raise KeyError('No cell nor dataset with name `%s`' % n)

    def add_step(self, op, *args, **kwargs):
        name = kwargs.pop('name', None)
        constants = kwargs.pop('constants', None)
        if isinstance(op, str) and op in self.ops:
            if name is None: name = op
            op = self.ops[op]
            cell_name = None
            self.nodes[name] = dict(dim=None)
            if len(args) > 0:
                self.match_dims(args[0], name + '.input')
        elif isinstance(op, str):
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

            if name is None: name = cell_name

            cell_type = self.cell_args[cell_name]['cell_type']
            C = self.resolve_class(cell_type)
            if op_name is None: op_name = '__call__'

            if hasattr(C, op_name):
                op = getattr(C, op_name)
            else:
                raise TypeError('Cell %s of type %s has no method %s.'
                                % (cell_name, C, op_name))

            if len(args) > 0:
                if op_name == '__call__':
                    arg_keys = C._call_args
                elif hasattr(C, '_%s_args' % op_name):
                    arg_keys = getattr(C, '_%s_args' % op_name)
                else:
                    arg_keys = None

                if arg_keys is not None:
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
            if name is None: name = op.__name__

        #self.test_op_args(op, args, kwargs)

        self.steps.append(dict(
            cell_name=cell_name, name=name, op=op, constants=constants,
            args=args, kwargs=kwargs))

        return name

    def add_cost(self, *args, **kwargs):
        self.add('cost', *args, **kwargs)

    def add_stat(self, *args, **kwargs):
        self.add('stat', *args, **kwargs)

    def add(self, what, op, *args, **kwargs):
        cell_name = None
        name = kwargs.pop('name', None)
        if isinstance(op, str):
            if is_tensor_arg(op):
                cell_name, n, _ = resolve_tensor_arg(op)
                cell_type = self.cell_args[cell_name]['cell_type']
                C = self.resolve_class(cell_type)

                if name is None: name = cell_name

                if what == 'cost':
                    if n == 'cost' and hasattr(C, '_cost'):
                        op = getattr(C, '_cost')
                    else:
                        if not n in C._costs.keys():
                            raise AttributeError(
                                'cell type %s for cell `%s` has no '
                                'cost %s' % (C, cell_name, n))
                        op = getattr(C, C._costs[n])
                        name = _p(name, n)

                elif what == 'stat':
                    if not n in C._stats.keys():
                        raise AttributeError('cell type %s for cell `%s` has no '
                                             'stat %s' % (C, cell_name, n))
                    op = getattr(C, C._stats[n])
                    name = _p(name, n)

            else:
                if name is None: name = op
                if what == 'cost':
                    if op not in self.cost_functions.keys():
                        raise TypeError('Cost function `%s` not found. '
                                        'Avalilable: %s'
                                        % (op, self.cost_functions.keys()))
                    op = self.cost_functions[op]

                elif what == 'stat':
                    if op not in self.stat_functions.keys():
                        raise TypeError('Stat function `%s` not found. '
                                        'Available: %s'
                                        % (op, self.stat_functions.keys()))
                    op = self.stat_functions[op]

        elif callable(op):
            if name is None: name = op.__name__

        else:
            raise TypeError

        #self.test_op_args(op, args, kwargs)
        if what == 'cost':
            if name in self.costs.keys():
                i = 1
                while True:
                    new_name = '%s_%d' % (name, i)
                    if not new_name in self.costs.keys():
                        break
                self.logger.warn('Cost `%s` already found. Changing to %s.'
                                 % (name, new_name))
                name = new_name
            else:
                self.logger.debug('Adding costs `%s`' % name)
            self.costs[name] = dict(
                cell_name=cell_name, op=op, args=args, kwargs=kwargs)

        elif what == 'stat':
            if name in self.stats.keys():
                i = 1
                while True:
                    new_name = '%s_%d' % (name, i)
                    if not new_name in self.stats.keys():
                        break
                self.logger.warn('Stat `%s` already found. Changing to %s.'
                                 % (name, new_name))
                name = new_name
            self.stats[name] = dict(
                cell_name=cell_name, op=op, args=args, kwargs=kwargs)

    def prepare_samples(self, arg, shape, name='samples', **kwargs):
        if isinstance(shape, int):
            shape = (shape,)

        if arg in self.cell_args.keys():
            cell_name = arg
            dist_key = None
        elif is_tensor_arg(arg):
            cell_name, dist_key, _ = resolve_tensor_arg(arg)

        if cell_name not in self.cell_args.keys():
            raise KeyError('Cell %s not found' % cell_name)

        C = self.resolve_class(self.cell_args[cell_name]['cell_type'])
        if not hasattr(C, '_sample'):
            raise ValueError('Cell type %s does not support sampling'
                             % C.__name__)

        if dist_key is not None and dist_key not in C._sample_tensors:
            raise KeyError('Cell %s does not support sample tensor %s'
                           % (cell_name, dist_key))

        s_name = _p(cell_name, name)
        if dist_key is not None: dist_key = arg

        if s_name in self.samples.keys():
            self.logger.warn('Overwriting samples %s' % name)
        else:
            self.logger.debug('Adding samples `%s`' % name)

        self.samples[s_name] = dict(
            cell_name=cell_name, name=name, dist_key=dist_key, shape=shape,
            kwargs=kwargs)

    # Methods for linking and dim matching -------------------------------------
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
        try:
            link = Link(f, t)
        except TypeError:
            return

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

    # Misc ---------------------------------------------------------------------
    def profile(self):
        self.logger.info('Profiling cells and params')
        for k, v in self.cells.iteritems():
            print_section('Profiling %s' % v.name)
            print v
            print 'Params:'
            for k, vv in self.tparams.iteritems():
                name, kk = split_arg(k)
                if name == v.name:
                    print '\t%s: %s' % (k, vv.get_value().shape)

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


_ops = {}
from .. import ops, visualization
modules = [ops, visualization]
for module in modules:
    _ops.update(module._ops)