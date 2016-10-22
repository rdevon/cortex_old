'''Basic analyzer class for reloading and analyzing models.

'''

import cortex


class Analyzer(object):
    def __init__(self, model_file, **extra_classes):
        for k, v in extra_classes.items():
            cortex.add_cell_class(k, v)
        cortex.load(model_file)
        
    def add_step(k, **kwargs):
        cortex.add_step(k, **kwargs)
        
    def build(self):
        pass