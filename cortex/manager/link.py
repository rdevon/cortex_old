'''Module for links.

'''
from . import resolve_tensor_arg, get_manager
from ..models import Cell
from ..utils.logger import get_class_logger


class Node(object):
    def __init__(self, C, key):
        if key not in C._dim_map.keys():
            raise TypeError('Class %s has no key `%s`' % (C, key))
        self.C = C
        self.link_key = key
        self.dim_key = C._dim_map[key]
        self.dist_key = C._dist_map.get(key)

class Link(object):

    def __init__(self, f, t, manager=None):
        if manager is None: manager = get_manager()
        self.manager = manager
        self.value = None
        self.distribution = None
        self.nodes = {}
        self.name = f + '->' + t
        self.logger = get_class_logger(self)

        f_name, f_key, f_class = resolve_tensor_arg(f)
        t_name, t_key, t_class = resolve_tensor_arg(t)

        dataset_name = None
        dataset_key = None

        if f_name in manager.datasets.keys():
            if t_name in manager.datasets.keys():
                raise ValueError('Cannot link 2 datasets')
            dataset_name = f_name
            dataset_key = f_key
        elif t_name in manager.datasets.keys():
            dataset_name = t_name
            dataset_key = t_key

        if dataset_name is not None:
            self.value = manager.datasets[dataset_name]['dims'][dataset_key]
            self.distribution = manager.datasets[
                dataset_name]['distributions'][dataset_key]

        elif f_name in manager.nodes.keys():
            if f_class['dim'] is None:
                raise ValueError('from: %s, to: %s, op dim: %s'
                                 % (f_name, t_name, f_class))
            self.value = f_class['dim']
        else:
            f_args = manager.cell_args[f_name]
            t_args = manager.cell_args[t_name]

            try:
                self.value = t_class.set_link_value(t_key, **t_args)
            except ValueError:
                self.logger.debug(
                    'Link value with class `%s`, key `%s`, and args %s failed'
                    % (t_class.__name__, t_key, t_args))
                try:
                    self.value = f_class.set_link_value(f_key, **f_args)
                except ValueError:
                    self.logger.debug(
                        'Link value with class `%s`, key `%s`, and args %s failed'
                        % (f_class.__name__, f_key, f_args))
            try:
                self.distribution = t_class.set_link_distribution(
                    t_key, **t_args)
            except (KeyError, ValueError):
                try:
                    self.distribution = f_class.set_link_distribution(
                        f_key, **f_args)
                except KeyError:
                    pass

        if self.value is None:
            raise TypeError('Link between %s and %s requires a resolvable '
                            'dimension' % (f, t))

        if t_name in manager.nodes.keys():
            manager.nodes[t_name]['dim'] = self.value

        if f_name in manager.nodes.keys():
            manager.nodes[f_name]['dim'] = self.value

        if isinstance(f_class, type) and issubclass(f_class, Cell):
            self.nodes[f_name] = Node(f_class, f_key)
        if isinstance(t_class, type) and issubclass(t_class, Cell):
            self.nodes[t_name] = Node(t_class, t_key)
        manager.links.append(self)

    def query(self, name, key):
        if not name in self.nodes.keys():
            raise KeyError('Link does not have node `%s`' % name)
        node = self.nodes[name]
        if key == node.dim_key:
            (vk_, value) = node.C.get_link_value(self, node.link_key)
            if value is None:
                raise ValueError
            return value
        elif key == node.dist_key:
            if self.distribution is None:
                raise ValueError
            return self.distribution
        else:
            raise KeyError('Link with node `%s` does not support key `%s`'
                           % (name, key))

    def __repr__(self):
        return ('<link>(%s)' % self.name)