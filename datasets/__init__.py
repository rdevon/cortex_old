'''
Generic dataset class
'''


class Dataset(object):
    def __init__(self, batch_size=None, shuffle=True, inf=False, name='dataset',
                 stop=None, **kwargs):
        if batch_size is None:
            raise ValueError('Batch size argument must be given')

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inf = inf
        self.name = name
        self.pos = 0

        return kwargs

    def randomize(self):
        return

    def reset(self):
        self.pos = 0
        if self.shuffle:
            self.randomize()
