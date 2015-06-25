import cPickle
import gzip
import numpy as np
import PIL
from vis_utils import tile_raster_images

def get_iter(inf=False, batch_size=128):
    return mnist_iterator(inf=inf, batch_size=batch_size)

class mnist_iterator:
    def __init__(self, batch_size=128, data_file='/Users/devon/Data/mnist.pkl.gz',
                 restrict_digits=None, mode='train', shuffle=True, inf=False,
                 repeat=1):
        # load MNIST
        with gzip.open(data_file, 'rb') as f:
            x = cPickle.load(f)

        if mode == 'train':
            X = np.float32(x[0][0])
            Y = np.float32(x[0][1])
        elif mode == 'valid':
            X = np.float32(x[1][0])
            Y = np.float32(x[1][1])
        elif model == 'test':
            X = np.float32(x[2][0])
            Y = np.float32(x[2][1])
        else:
            raise ValueError()

        if restrict_digits is None:
            n_classes = 10
        else:
            n_classes = len(restrict_digits)

        O = np.zeros((X.shape[0], n_classes), dtype='float32')
        if restrict_digits is None:
            for idx in xrange(X.shape[0]):
                O[idx, Y[idx]] = 1.;
        else:
            new_X = []
            i = 0
            for j in xrange(X.shape[0]):
                if Y[j] in restrict_digits:
                    new_X.append(X[j])
                    c_idx = restrict_digits.index(Y[j])
                    O[i, c_idx] = 1.;
                    i += 1
            X = np.float32(new_X)

        self.restict_digits = restrict_digits
        self.shuffle = shuffle
        self.n = X.shape[0]
        self.dim = X.shape[1]

        self.pos = 0
        self.bs = batch_size

        self.inf = inf
        self.repeat = repeat

        self.X = X
        self.O = O

        # randomize
        if self.shuffle:
            self.randomize()

    def __iter__(self):
        return self

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.X.shape[0], 1));
        self.X = self.X[rnd_idx, :];
        self.O = self.O[rnd_idx, :];

    def next(self):
        cpos = self.pos
        if cpos == -1:
            # reset
            self.pos = 0
            if self.shuffle:
                self.randomize()
            if not self.inf:
                raise StopIteration

        x = self.X[cpos:cpos+self.bs]
        y = self.O[cpos:cpos+self.bs]
        x = np.concatenate([x for _ in xrange(self.repeat)], axis=0)
        y = np.concatenate([y for _ in xrange(self.repeat)], axis=0)

        cbatch = (x, y)

        if cpos + self.bs >= self.n:
            self.pos = -1
        else:
            self.pos += self.bs

        return cbatch

    def save_images(self, x, imgfile, transpose=False):
        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape, transpose=transpose)
        #print 'Saving to ', imgfile
        image.save(imgfile)

    def show(self, image, tshape, transpose=False):
        fshape = (28, 28)
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))