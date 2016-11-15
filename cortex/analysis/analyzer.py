'''Basic analyzer class for reloading and analyzing models.

'''

import theano

import cortex


class Analyzer(object):
    def __init__(self, model_file, **extra_classes):
        for k, v in extra_classes.items():
            cortex.add_cell_class(k, v)
        cortex.load(model_file)
        
    def add_step(k, **kwargs):
        cortex.add_step(k, **kwargs)
        
    def build(self):
        cortex.build()
        self.session = cortex.create_session(noise=False)
        cortex.build_session()
        self.visualizer = cortex.setup_visualizer(self.session)
        self.tensors = self.session.tensors
        
    def get_tensor(self, k, data='data', mode='test', batch_size=10):
        rval = self.tensors[k]
        f = theano.function(self.session.inputs, rval,
                            updates=self.session.updates,
                            on_unused_input='ignore')
        data = cortex.get_datasets()[data][mode]
        d = data.next(batch_size)
        d = [d[k.name.split('.')[-1]] for k in self.session.inputs]
        
        return f(*d)
    
    def get_tensor_names(self):
        return self.tensors.keys()