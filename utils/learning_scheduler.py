'''Scheduler for learning rates.
'''

from collections import OrderedDict


def unpack(learning_rate=None, decay_rate=None, schedule=None):
    return learning_rate, decay_rate, schedule


class Scheduler(object):
    '''Scheduler for learning rates.

    Attributes:
        d: OrderedDict, dictionary of learning rates and decays, schedules.
    '''
    def __init__(self, verbose=True, **kwargs):
        '''Init function of Scheduler.'''
        self.d = OrderedDict()
        self.verbose = verbose
        for k, v in kwargs.iteritems():
            self.d[k] = OrderedDict()
            if isinstance(v, float):
                self.d[k]['learning_rate'] = v
            elif isinstance(v, (dict, OrderedDict)):
                def unpack(learning_rate=None, decay_rate=None, schedule=None):
                    return learning_rate, decay_rate, schedule

                learning_rate, decay_rate, schedule = unpack(**v)

                if learning_rate is None:
                    raise ValueError('Must includes learning rate for %s' % k)

                if (decay_rate is not None) and (schedule is not None):
                    raise ValueError('Provide either decay rate OR scheduler OR neither'
                                     ', not both.')
                self.d[k]['decay_rate'] = decay_rate
                self.d[k]['schedule'] = schedule
                self.d[k]['learning_rate'] = learning_rate

    def __getitem__(self, k):
        return self.d[k]

    def __call__(self, e):
        '''Update the learning rates and return list.'''

        for k, v in self.d.iteritems():
            learning_rate, decay_rate, schedule = unpack(**v)

            if decay_rate is None and schedule is None: continue
            if decay_rate is not None:
                self.d[k]['learning_rate'] *= decay_rate
            elif schedule is not None and e in schedule.keys():
                self.d[k]['learning_rate'] = schedule[e]
            if self.verbose:
                print 'Changing learning rate for %s to %.5f' % (k, self.d[k]['learning_rate'])

        return [v['learning_rate'] for v in self.d.values()]
