'''Extra operations for fMRI and MRI analysis.

'''

import igraph
import numpy as np
from scipy.stats import ttest_1samp
from scipy.interpolate import UnivariateSpline
import theano
import theano.tensor as T
from theano.gof import Apply, Op

from cortex.utils import floatX, intX


class Detrender(theano.Op):
    '''Detrender for time courses.

    Detrends along the 0-axis. Wraps numpy.polyfit and numpy.polyval.

    Attributes:
        order (int): order of the polynomial fit.

    '''
    __props__ = ()

    itypes = [T.ftensor3]
    otypes = [T.ftensor3]

    def __init__(self, order=4):
        '''Initializer for Detrender.

        Args:
            order (int): order of the polynomial fit.

        '''
        self.order=order
        super(Detrender, self).__init__()

    def perform(self, node, inputs, output_storage):
        '''Perform detrending.

        '''
        data = inputs[0]
        z = output_storage[0]
        x = np.arange(data.shape[0])
        if len(data.shape) == 3:
            reshape = data.shape
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
        elif len(data.shape) > 3:
            raise ValueError('Detrending over 3 dims not supported')
        else:
            reshape = None
        fit = np.polyval(np.polyfit(x, data, deg=self.order),
                         np.repeat(x[:, None], data.shape[1], axis=1))
        data = data - fit
        if reshape is not None:
            data = data.reshape(reshape)
        z[0] = data.astype(floatX)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes
detrender = Detrender()


class Despiker(theano.Op):
    __props__ = ()

    itypes = [T.ftensor3]
    otypes = [T.ftensor3]

    def perform(self, node, inputs, output_storage):
        data = inputs[0]
        z = output_storage[0]

        def despike_one(tc, bpfrac=0.5):
            tlen = tc.shape[0]
            maxid = np.argmax(tc)
            minid = np.argmin(tc)

            x = np.setdiff1d(range(tlen), [maxid, minid])
            y = tc[x]

            mad_res = np.median(abs(y - np.median(y)))
            sig = mad_res * np.sqrt(np.pi / 2)
            s = tc / sig
            idx = np.where(abs(s) > 2.5)[0]

            x = np.setdiff1d(range(tlen), idx).flatten()
            y = tc[x]
            edges = np.setdiff1d([0, tlen - 1],x)

            if edges.shape[0] > 0:
               f0 = UnivariateSpline(x, y, k=1)
               y0 = f0.__call__(range(tlen))
               tc[edges] = y0[edges].astype(np.float32)

            idx = np.setdiff1d(idx, edges)

            f3 = UnivariateSpline(x, y, k=3)
            f2 = UnivariateSpline(x, y, k=2)
            f1 = UnivariateSpline(x, y, k=1)
            y3 = f3.__call__(range(0, tlen))
            y2 = f2.__call__(range(0, tlen))
            y1 = f1.__call__(range(0, tlen))
            for i in idx:
                if (abs(tc[i]) > abs(y3[i])):
                    tc[i] = y3[i]
                elif (abs(tc[i]) > abs(y2[i])):
                    tc[i] = y2[i]
                else:
                    tc[i] = y1[i]
            return tc

        shape = data.shape
        data = data.reshape((shape[0], shape[1] * shape[2]))

        tcs = []
        for i in xrange(data.shape[1]):
            tcs.append(despike_one(data[:, i]))

        tcs = np.array(tcs).astype(floatX)
        tcs = tcs.T.reshape(shape)
        z[0] = tcs

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

class PCASign(Op):
    '''Gets the sign of a feature that was preprocessed with PCA.

    '''
    __props__ = ()

    def __init__(self, pca):
        self.pca = pca
        super(PCASign, self).__init__()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        o = T.matrix(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, output_storage):
        '''Get signs

        '''
        (x,) = inputs
        (z,) = output_storage
        x = self.pca.inverse_transform(x)
        x /= x.std(axis=1, keepdims=True)
        x[abs(x) < 2] = 0
        signs = 2 * (x.sum(axis=1) >= 0).astype(floatX) - 1
        z[0] = signs


class PCAInverse(Op):
    __props__ = ()

    def __init__(self, pca):
        self.pca = pca
        super(PCAInverse, self).__init__()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        o = T.vector(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, output_storage):
        '''Get signs

        '''
        (x,) = inputs
        (z,) = output_storage
        z[0] = self.pca.inverse_transform(x)


class CorrCoef(Op):
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim >= 2
        o = T.matrix(dtype=floatX)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        (z,) = output_storage
        '''
        if x.ndim == 3:
            x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        elif x.ndim == 4:
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        else:
            raise TypeError(x.ndim)
        '''
        if x.ndim == 2:
            cc = np.corrcoef(x.T)
        else:
            cc = []
            for i in range(x.shape[1]):
                cc.append(np.corrcoef(x[:, i].T))
            cc = np.array(cc).mean(0).astype(floatX)
        z[0] =  cc


class Cluster(Op):
    __props__ = ()

    def __init__(self, thr=0.2):
        self.thr = thr
        super(Cluster, self).__init__()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        o = T.vector(dtype=intX)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, output_storage):
        (mat,) = inputs
        (z,) = output_storage
        mat = abs(mat)

        max_weight = mat.max()
        thr = self.thr * max_weight
        idx = range(mat.shape[0])
        wheres = np.where(mat > thr)

        edgelist = []
        weights = []

        for x, y in zip(wheres[0], wheres[1]):
            if x < y:
                edgelist.append((x, y))
                weights.append((mat[x, y]))

        if len(weights) > 0:
            weights /= np.std(weights)
        else:
            return idx

        g = igraph.Graph(edgelist, directed=False)
        g.vs['label'] = idx
        cls = g.community_multilevel(return_levels=True, weights=weights)
        cl = list(cls[0])
        assert cl is not None

        clusters = []
        n_clusters = len(cl)
        for i in idx:
            found = False
            for j, cluster in enumerate(cl):
                if i in cluster:
                    clusters.append(j)
                    found = True
                    break

            if not found:
                clusters.append(n_clusters)
                n_clusters += 1

        clusters = np.array(clusters).astype(intX)
        z[0] = clusters


class OLS(Op):
    __props__ = ()

    def make_node(self, x, y):
        x = T.as_tensor_variable(x)
        y = T.as_tensor_variable(y)
        assert x.ndim == 3
        assert y.ndim == 2
        o = T.tensor3(dtype=floatX)
        return Apply(self, [x, y], [o])

    def perform(self, node, inputs, output_storage):
        (x, y) = inputs
        (z,) = output_storage

        betas = []
        for c in xrange(x.shape[2]):
            beta, _, _, _ = np.linalg.lstsq(x[:, :, c], y)
            betas.append(beta)
        betas = np.asarray(betas).astype(floatX)
        assert betas.ndim == 3, betas.ndim

        z[0] = betas


class TTest(Op):
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 3
        o = T.matrix(dtype=floatX)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        (z,) = output_storage

        test = ttest_1samp(x, 0., axis=1)
        test = np.concatenate([test[0][:, :, None], test[1][:, :, None]], axis=2).astype(floatX)
        z[0] = test
