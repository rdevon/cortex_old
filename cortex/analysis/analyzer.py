'''Basic analyzer class for reloading and analyzing models.

'''

from os import path
import theano

import cortex
from ..utils.logger import get_class_logger


class Analyzer(object):
    def __init__(self, session=None, model_file=None, visualizer=None,
                 mode='test', build=True, out_path=None, **extra_classes):
        self.logger = get_class_logger(self)
        
        for k, v in extra_classes.items():
            cortex.add_cell_class(k, v)
        if model_file is not None: cortex.load(model_file)
        self.features = {}
        self.visualizer = visualizer
        self.session = session
        self.mode = mode
        if out_path is None:
            out_path = path.join(path.dirname(model_file), 'analysis')
        cortex.set_path(out_path)
        if build: self.build()
        
    def add_step(k, **kwargs):
        cortex.add_step(k, **kwargs)
        
    def build(self):
        if self.session is None:            
            cortex.build()
            self.session = cortex.create_session(noise=False)
            cortex.build_session()
        if self.visualizer is None:
            self.visualizer = cortex.setup_visualizer(self.session)
        self.tensors = self.session.tensors
        
    def get_data(self):
        self.session.reset_data(mode=self.mode)
        batch_size = self.session.get_dataset_size(mode=self.mode)
        inputs = self.session.next_batch(mode=self.mode, batch_size=batch_size)
        return inputs
        
    def get_tensor(self, k, batch_size=10):
        rval = self.tensors[k]
        f = theano.function(self.session.inputs, rval,
                            updates=self.session.updates,
                            on_unused_input='ignore')
        return f(*self.get_data())
    
    def get_tensor_names(self):
        return self.tensors.keys()
    
    def add_features(self, **kwargs):
        self.features.update(**kwargs)