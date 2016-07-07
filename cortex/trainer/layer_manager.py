'''Module for LayerManager class.

LayerManeger links dims and distributions between objects.

'''

from collections import OrderedDict
import copy
import logging

from ..layers import build_layer


class LayerManager(object):
    '''LayerManager for managing linking and tensor passing.

    Ensures that connected objects have the right dimensionality as well as
        manages passing the correct tensors as input and cost.

    '''

    def __init__(self, **layer_kwargs):
        self.layer_kwargs = dict()
        self.layer_kwargs.update(**layer_kwargs)
        self.layers = OrderedDict()


    def build_layers(self):
        for l_name, kwargs in self.layer_kwargs.iteritems():
            self.layers[l_name] = build_layer(**kwargs)

    def link(self, from_id, to_id):
        pass