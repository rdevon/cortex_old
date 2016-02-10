'''
Dataset for breaking data into pieces.
'''


class Pieces(object):
    def __init__(self, D, batch_size=10,
                 size=None, stride=None,
                 out_path=None,
                 **kwargs):
        self.dataset = D(**kwargs)
        self.batch_size = batch_size

        if isinstance(size, int):
            size = tuple([size for _ in self.dataset.dims])
        assert len(size) == len(self.dataset.dims)
        self.size = size

        if isinstance(stride, int):
            stride = tuple([stride for _ in self.dataset.dims])
        assert len(stride) == len(self.dataset.dims)
        self.stride = stride

        self.out_path = out_path

        self.next = self._next
        self.ppos = -1
        init_rngs(self, **kwargs)

    def next(self):
        raise NotImplementedError()
