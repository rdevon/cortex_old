import cPickle
import gzip
import multiprocessing as mp
import numpy as np
import PIL
import random
import sys
import theano
from theano import tensor as T
import traceback
from vis_utils import tile_raster_images

def get_iter(inf=False, batch_size=128):
    return mnist_iterator(inf=inf, batch_size=batch_size)

class mnist_iterator:
    def __init__(self, batch_size=128, data_file='/Users/devon/Data/mnist.pkl.gz',
                 restrict_digits=None, mode='train', shuffle=True, inf=False,
                 out_mode='', repeat=1, chain_length=20):
        # load MNIST
        with gzip.open(data_file, 'rb') as f:
            x = cPickle.load(f)

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

        self.f_energy = None
        self.pairs = None
        self.chain = None
        self.chain_idx = None
        self.chain_length = chain_length
        self.chains = None
        self.mode = out_mode

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
        elif self.mode == 'model_chains':
            self.next = self._next_model_chains
        else:
            self.next = self._next

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
        if self.shuffle and not self.mode in ['pairs', 'chains', 'multi_chains']:
            print 'Shuffling mnist'
            self.randomize()

    def __iter__(self):
        return self

    def set_f_energy(self, f_energy):
        self.f_energy = f_energy

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.X.shape[0], 1))
        self.X = self.X[rnd_idx, :];
        self.O = self.O[rnd_idx, :];

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

    def _next_model_chains(self):
        assert self.f_energy is not None

        data = np.zeros((self.chain_length, self.bs, self.dim)).astype('float32')
        last_batch, _ = self._next()
        data[0] = last_batch

        energies = np.zeros((self.bs, self.bs))

        for i, x in enumerate(batch):
            for j, y in enumerate(batch):
                if i != j:
                    energies[i, j] = self.f_energy(x, y)

        for y in xrange(1, self.chain_length):
            last_batch = data[y - 1]
            batch, _ = self._next()
            batch_idx = range(last_batch.shape[0])
            random.shuffle(batch_idx)

            for x in batch:
                min_e = float('inf')
                for k, j in enumerate(batch_idx):
                    energy = self.f_energy(last_batch[j], x)
                    if energy < min_e:
                        min_e = energy
                        pop_idx = k
                        min_idx = j
                energies[min_idx] += min_e
                data[y, min_idx] = x
                batch_idx.pop(pop_idx)

        idx = np.argsort(energies).tolist()
        data = data[:, idx]

        return data, None

    def _next(self):
        cpos = self.pos
        if cpos == -1:
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                self.randomize()
            if not self.inf:
                raise StopIteration

        x = self.X[cpos:cpos+self.bs]
        y = self.O[cpos:cpos+self.bs]
        x = np.concatenate([x] * self.repeat, axis=0)
        y = np.concatenate([y] * self.repeat, axis=0)

        if cpos + 2 * self.bs >= self.n:
            self.pos = -1
        else:
            self.pos += self.bs

        if self.mode == 'symmetrize_and_pair':
            new_x = np.zeros((2, self.batch_size * self.batch_size, x.shape[1])).astype('float32')
            for i in xrange(batch_size):
                for j in xrange(batch_size):
                    batch = np.concatenate([x[i][None, :], x[j][None, :]])
                    new_x[:, self.batch_size * j + i] = batch
            x = new_x

        return x, y

    def save_images(self, x, imgfile, transpose=False):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))
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

    def translate(self, x):
        return x

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
