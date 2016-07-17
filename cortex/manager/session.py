'''Module for the Session class.

'''


def make_inputs(module):
    '''Forms the inputs from the dataset

    '''
    dataset = module.dataset

    print_section('Setting inputs')
    d = dataset.next()
    inps = OrderedDict()
    for k, v in d.iteritems():
        logger.debug('Data mode %s has batch shape %s.' % (k, v.shape))
        if v.ndim == 1:
            C = T.vector
        elif v.ndim == 2:
            C = T.matrix
        elif v.ndim == 3:
            C = T.tensor3
        else:
            raise ValueError('Data dim over 3 not supported.')

        if v.dtype == floatX:
            dtype = floatX
        elif v.dtype == intX:
            dtype = intX
        else:
            raise ValueError('dtype %s not supported' % v.dtype)

        X = C(k, dtype=dtype)
        inps[k] = X
    logger.debug('Dataset has the following inputs: %s with types %s'
                 % (inps, [inp.dtype for inp in inps.values()]))
    dataset.reset()
    module.inputs = inps


class Session(object):
    def __init__(self, idx, manager):
        self.tensors = {}
        self.manager = manager


