import cPickle
import gzip
import multiprocessing as mp
import numpy as np
from os import path
import PIL
import random
import sys
from sys import stdout
import theano
from theano import tensor as T
import traceback
from vis_utils import tile_raster_images

def get_iter(inf=False, batch_size=128):
    return mnist_iterator(inf=inf, batch_size=batch_size)

class mnist_iterator(object):
    def __init__(self, batch_size=128, source='/Users/devon/Data/mnist.pkl.gz',
                 restrict_digits=None, mode='train', shuffle=True, inf=False,
                 out_mode='', repeat=1, chain_length=20, reset_chains=False,
                 chain_build_batch=100, stop=None, out_path=None, chain_stride=None):
        # load MNIST
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        X, Y = self.get_data(x, mode)

        self.dims = (28, 28)
        self.f_energy = None
        self.pairs = None
        self.chain = None
        self.chain_idx = None
        self.chain_length = chain_length
        self.chains = None
        self.counts = None
        self.mode = out_mode
        self.chains_build_batch = chain_build_batch
        self.reset_chains = reset_chains
        self.out_path = out_path
        if chain_stride is None:
            self.chain_stride = self.chain_length
        else:
            self.chain_stride = chain_stride

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
        if stop is not None:
            X = X[:stop]
            self.n = stop
        self.dim = X.shape[1]

        self.pos = 0
        self.bs = batch_size

        self.inf = inf
        self.repeat = repeat

        self.X = X
        self.O = O

        if self.mode == 'pairs':
            self.next = self._next_pairs
            self.pairs = np.load('pairs.npy').tolist()
        elif self.mode == 'chains':
            self.next = self._next_chains
            self.chain = np.load('chain.npy').tolist()
            self.chain_idx = range(0, len(self.chain) - chain_length)
            random.shuffle(self.chain_idx)
        elif self.mode == 'multi_chains':
            self.next = self._next_multi_chains
            self.chains = np.load('chains.npy').tolist()
            self.chain_offset = 0
            self.chain_idx = []
            for i, chain in enumerate(self.chains):
                self.chain_idx += [(i, j)
                    for j in range(0, (len(chain) - 2 * self.chain_length), self.chain_length)]
            random.shuffle(self.chain_idx)
        elif self.mode in ['model_chains', 'big_model_chains']:
            self.chains = [[] for _ in xrange(self.bs)]
            self.next = self._next_model_chains
            self.counts = [0 for _ in xrange(self.n)]
            self.chain_pos = 0
            if self.mode == 'big_model_chains':
                self.next = self._next_model_chains_big
        else:
            self.next = self._next

        # randomize
        if self.shuffle and not self.mode in ['pairs', 'chains', 'multi_chains']:
            print 'Shuffling mnist'
            self.randomize()

    def get_data(self, x, mode):
        if mode == 'train':
            X = np.float32(x[0][0])
            Y = np.float32(x[0][1])
        elif mode == 'valid':
            X = np.float32(x[1][0])
            Y = np.float32(x[1][1])
        elif mode == 'test':
            X = np.float32(x[2][0])
            Y = np.float32(x[2][1])
        else:
            raise ValueError()

        return X, Y

    def process(self, X, Y, restrict_digits=None):
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

        return X, O

    def __iter__(self):
        return self

    def set_f_energy(self, f_energy_fn, model):
        self.f_energy = f_energy_fn(model)
        self.dim_h = model.dim_h

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.O = self.O[rnd_idx, :]

    def next(self):
        raise NotImplementedError()

    def _next_pairs(self):
        assert self.pairs is not None
        cpos = self.pos
        if cpos == -1:
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                random.shuffle(self.pairs)
            if not self.inf:
                raise StopIteration

        pairs = [self.pairs[i] for i in xrange(cpos, cpos + self.bs)] * self.repeat

        if cpos + 2 * self.bs >= len(self.pairs):
            self.pos = -1
        else:
            self.pos += self.bs

        x = np.zeros((2, self.bs * self.repeat, self.dim)).astype('float32')
        y = []
        for i, p in enumerate(pairs):
            x[0, i] = self.X[p[0]]
            x[1, i] = self.X[p[1]]
            y.append((np.argmax(self.O[p[0]]), np.argmax(self.O[p[1]])))

        return x, y, pairs

    def _next_chains(self):
        assert self.chain is not None
        cpos = self.pos
        if cpos == -1:
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                random.shuffle(self.chain_idx)
            if not self.inf:
                raise StopIteration

        chains = [[self.chain[i]
                   for i in xrange(self.chain_idx[j],
                                   self.chain_idx[j] + self.chain_length)]
            for j in xrange(cpos, cpos + self.bs)]

        if cpos + 2 * self.bs >= len(self.chain_idx):
            self.pos = -1
        else:
            self.pos += self.bs

        x = np.zeros((self.chain_length, self.bs, self.dim)).astype('float32')
        for i, c in enumerate(chains):
            x[:, i] = self.X[c]

        return x, None

    def _next_multi_chains(self):
        assert self.chains is not None
        cpos = self.pos
        if cpos == -1:
            print 'EOE'
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                random.shuffle(self.chain_idx)
                self.chain_offset = random.randint(0, self.chain_length)
            if not self.inf:
                raise StopIteration
        ijs = [(i, j) for i, j in [self.chain_idx[k] for k in xrange(cpos, cpos + self.bs)]]

        try:
            chains = [[self.chains[i][j + t + self.chain_offset]
                       for t in range(self.chain_length)] for i, j in ijs]
        except IndexError as e:
            print ijs, self.chain_offset
            raise e

        if cpos + 2 * self.bs >= len(self.chain_idx):
            self.pos = -1
        else:
            self.pos += self.bs

        x = np.zeros((self.chain_length, self.bs, self.dim)).astype('float32')
        for i, c in enumerate(chains):
            x[:, i] = self.X[c]

        return x, None

    def _build_chains(self, length=None, pick_discarded=False, x_p=None, h_p=None):
        if length is None:
            length = self.chain_length

        x, _ = self._next(batch_size=self.chains_build_batch)
        x_idx = range(self.pos - self.chains_build_batch,
                      self.pos)

        if pick_discarded:
            max_count = np.max([self.counts[i] for i in x_idx])
            idx = [i for i in x_idx if self.counts[i] == max_count]
        else:
            idx = x_idx[:]

        if len(self.chains[0]) == 0:
            pass
        else:
            counts = [self.counts[i] for i in idx]
            energies, x_p, h_p = self.f_energy(x, x_p, h_p)
            cidx = np.argmin(energies, axis=1)
            idx = [idx[i] for i in cidx]

        picked_idx = np.unique(idx).tolist()
        for i in x_idx:
            if i not in picked_idx:
                self.counts[i] += 1

        recurse = False

        for i in xrange(len(self.chains)):
            chain = self.chains[i]
            j = idx[i]
            self.chains[i].append(j)
            if len(chain) > length:
                chain.pop(0)
            elif len(chain) < length:
                recurse = True

        if recurse:
            self._build_chains(length=length, pick_discarded=pick_discarded,
                               x_p=x_p, h_p=h_p)
        else:
            return

    def _next_model_chains(self):
        assert self.f_energy is not None

        if self.reset_chains:
            self.chains = [[] for _ in xrange(self.bs)]
        self._build_chains()
        x = self._load_chains()

        return x, None

    def _next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.bs

        cpos = self.pos
        if cpos + batch_size > self.n:
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                self.randomize()
            if not self.inf:
                raise StopIteration

        x = self.X[cpos:cpos+batch_size]
        y = self.O[cpos:cpos+batch_size]
        x = np.concatenate([x] * self.repeat, axis=0)
        y = np.concatenate([y] * self.repeat, axis=0)

        self.pos += batch_size

        if self.mode == 'symmetrize_and_pair':
            new_x = np.zeros((2, batch_size * batch_size, x.shape[1])).astype('float32')
            for i in xrange(batch_size):
                for j in xrange(batch_size):
                    batch = np.concatenate([x[i][None, :], x[j][None, :]])
                    new_x[:, batch_size * j + i] = batch
            x = new_x

        return x, y

    def save_images(self, x, imgfile, transpose=False, x_limit=None):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))

        if x_limit is not None and x.shape[0] > x_limit:
            x = np.concatenate([x, np.zeros((x_limit - x.shape[0] % x_limit,
                                             x.shape[1],
                                             x.shape[2])).astype('float32')],
                axis=0)
            x = x.reshape((x_limit, x.shape[0] * x.shape[1] // x_limit, x.shape[2]))

        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape, transpose=transpose)
        #print 'Saving to ', imgfile
        image.save(imgfile)

    def show(self, image, tshape, transpose=False):
        fshape = self.dims
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))

    def translate(self, x):
        return x


class MNIST_Chains(mnist_iterator):
    def __init__(self, batch_size=1, source='/Users/devon/Data/mnist.pkl.gz',
                 restrict_digits=None, mode='train', shuffle=True,
                 window=20, chain_length=5000,  chain_build_batch=1000,
                 stop=None, out_path=None, chain_stride=None):
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        X, Y = self.get_data(x, mode)

        self.dims = (28, 28)
        self.f_energy = None
        self.chain_length = chain_length
        self.window = window
        self.chains_build_batch = chain_build_batch
        self.bs = batch_size

        self.out_path = out_path
        if chain_stride is None:
            self.chain_stride = self.chain_length
        else:
            self.chain_stride = chain_stride

        self.shuffle = shuffle

        X, O = self.process(X, Y, restrict_digits)

        if stop is not None:
            X = X[:stop]
        self.n, self.dim = X.shape
        self.chain_length = min(self.chain_length, self.n)
        self.chains = [[] for _ in xrange(self.bs)]
        self.chain_pos = 0
        self.pos = 0

        self.X = X
        self.O = O

        if self.shuffle:
            print 'Shuffling mnist'
            self.randomize()

    def _load_chains(self, chains=None):
        if chains is None:
            chains = self.chains

        x = np.zeros((len(chains[0]), len(chains), self.dim)).astype('float32')
        for i, c in enumerate(chains):
            x[:, i] = self.X[c]
        return x

    def _build_chain(self, x_p=None, h_p=None):
        n_chains = len(self.chains)
        l_chain = min(self.chain_length, self.n - self.pos)
        n_samples = min(self.chains_build_batch, l_chain)
        chain_idx = [range(l_chain) for _ in xrange(n_chains)]

        print('Resetting chains with length %d and %d samples per query. '
              'Position in data is %d'
              % (l_chain, n_samples, self.pos))

        for c in xrange(n_chains):
            rnd_idx = np.random.permutation(np.arange(0, l_chain, 1))
            chain_idx[c] = [chain_idx[c][i] for i in rnd_idx]

        counts = [[True for _ in xrange(l_chain)] for _ in xrange(n_chains)]
        n = l_chain
        pos = 0

        while n > 0:
            stdout.write('\r%d         ' % n); stdout.flush()
            x_idx = []

            for c in xrange(n_chains):
                idx = [j for j in chain_idx[c] if counts[c][j]]
                x_idx.append([idx[i] for i in range(pos, pos + min(n_samples, n))])

            uniques = np.unique(x_idx).tolist()
            x = self.X[[u + self.pos for u in uniques]]
            wheres = [dict((i, j) for i, j in enumerate(uniques) if j in x_idx[c])
                for c in xrange(n_chains)]

            if n == l_chain:
                picked_idx = [random.choice(idx) for idx in x_idx]
            else:
                assert x_p is not None
                energies, _, h_p = self.f_energy(x, x_p, h_p)
                picked_idx = []
                for c in xrange(n_chains):
                    idx = wheres[c].keys()
                    chain_energies = energies[c, idx]
                    i = np.argmin(chain_energies)
                    j = wheres[c][idx[i]]
                    picked_idx.append(j)

            for c in xrange(n_chains):
                j = picked_idx[c]
                counts[c][j] = False
                self.chains[c].append(j + self.pos)

            x_p = self.X[[i + self.pos for i in picked_idx]]
            n -= 1

            if pos + n_samples > n:
                pos = 0
                for c in xrange(n_chains):
                    rnd_idx = np.random.permutation(np.arange(0, l_chain, 1))
                    chain_idx[c] = [chain_idx[c][i] for i in rnd_idx]

    def next(self):
        assert self.f_energy is not None

        chain_length = min(self.chain_length, self.n - self.pos)
        cpos = self.chain_pos
        if cpos == -1 or len(self.chains[0]) == 0:
            self.chain_pos = 0
            cpos = self.chain_pos
            self.chains = [[] for _ in xrange(self.bs)]
            x_p = None
            h_p = np.random.normal(loc=0, scale=1, size=(self.bs, self.dim_h)).astype('float32')
            self._build_chain(x_p=x_p, h_p=h_p)
            assert len(np.unique(self.chains[0])) == len(self.chains[0]), (len(np.unique(self.chains[0])), len(self.chains[0]))
            for i in self.chains[0]:
                assert i >= self.pos and i < self.pos + self.chain_length

            if self.out_path:
                self.save_images(self._load_chains(), path.join(self.out_path, 'training_chain.png'))

            self.pos += self.chain_length
            if self.pos >= self.n:
                self.pos = 0
                self.randomize()

        chains = [[chain[j] for j in xrange(cpos, cpos+self.window)]
            for chain in self.chains]

        x = self._load_chains(chains=chains)

        if cpos + self.chain_stride + self.window >= chain_length:
            self.chain_pos = -1
        else:
            self.chain_pos += self.chain_stride

        return x, None


def print_graph(mat, ys, thr=0.1, graph_file=None, mode='multilevel'):
    colors = {
        0: 'blue',
        1: 'red',
        2: 'orange',
        3: 'green',
        4: 'purple',
        5: 'brown',
        6: 'white',
        7: 'yellow',
        8: 'cyan',
        9: 'tan',
        10: 'magenta',
        11: 'grey',
        12: 'black'}

    max_weight = np.max(mat)
    thr = thr * max_weight
    wheres = np.where(mat > thr)

    edgelist = []
    weights = []
    for x, y in zip(wheres[0], wheres[1]):
        if x < y:
            edgelist.append([x, y])
            weights.append(mat[x, y])
    weights = weights / np.std(weights)
    graph = Graph(edgelist, directed=False)
    graph.vs['label'] = ys

    if mode == 'eigenvector':
        cl = graph.community_leading_eigenvector(clusters=10)
    elif mode == 'multilevel':
        cls = graph.community_multilevel(return_levels=True, weights=weights)
        cl = list(cls[0])

    print cl
    print [[ys[c] for c in cls] for cls in cl]
    for c, g in enumerate(cl):
        graph.vs(cl[c])['color'] = colors[c % 13]

    igraph.plot(graph, graph_file, weights=weights, edge_width=weights/10, vertex_label_size=8)

counter = None

def init(_counter):
    global counter
    counter = _counter

def make_chain((chain, X)):
    global counter
    idx = range(50000)
    random.shuffle(idx)
    for k in idx:
        if k != chain[-1] and np.corrcoef(X[chain[-1]], X[k])[0, 1] >= 0.8:
            chain.append(k)
        if len(chain) == 10:
            break
    if len(chain) == 10:
        counter.value += 1
        sys.stdout.write('\r%d' % counter.value); sys.stdout.flush()
    else:
        chain = []
    return chain

def make_chains():
    m = mnist_iterator(out_mode='chains', batch_size=1)
    rnd_idx = np.random.permutation(np.arange(0, m.X.shape[0], 1))

    counter = mp.Value('i', 0)
    p = mp.Pool(100, initializer=init, initargs=(counter, ))
    try:
        i = p.map_async(make_chain, [([i], m.X) for i in rnd_idx])
        i.wait()
        chains = i.get()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
    finally:
        p.terminate()
        p.join()
    return chains
