'''
Module for RBM class
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Layer
from distributions import Bernoulli
from utils import floatX
from utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    log_sum_exp,
    log_mean_exp,
    norm_weight,
    ortho_weight,
    scan
)


def unpack(dim_h=None,
           dim_in=None,
           data_iter=None,
           **model_args):

    dim_in = int(dim_in)
    dim_h = int(dim_h)

    rbm = RBM(dim_in, dim_h, mean_image=data_iter.mean_image)
    models = [rbm]

    return models, model_args, None


class RBM(Layer):
    '''
    RBM class.

    Currently supports only binary hidden units.

    Attributes:
        dim_v: int, number of visible units.
        dim_h: int, number of hidden units.
        h_act: str, Theano function name for hidden activation.
        v_act: str, Theano function name for visible activation.
        W: T.tensor, weights
        b: T.tensor, visible bias
        c: T.tensor, hidden bias.
        log_Z: T.tensor, current approximation of the log marginal.
        std_log_Z: T.tensor, current std of the approximate log marginal.
        mean_image: T.tensor, used for marginal approximation.
    '''
    def __init__(self, dim_v, dim_h, mean_image=None, name='rbm', **kwargs):
        '''Init method for RBM class.

        Args:
            dim_v: int, number of visible layer units
            dim_h: int, number of hidden layer units
            mean_image: np.array (optional), used for marginal approximation.
                if None, then set to 0.5.
        '''
        self.dim_v = dim_v
        self.dim_h = dim_h
        self.h_dist = Bernoulli(dim_h)
        self.v_act = 'T.nnet.sigmoid'

        if mean_image is None:
            mean_image = np.zeros(self.dim_v,).astype(floatX) + 0.5
        self.mean_image = theano.shared(np.clip(mean_image, 1e-7, 1 - 1e-7),
                                        name='mean_image')

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)
        super(RBM, self).__init__(name=name, excludes=['log_Z', 'std_log_Z'],
                                  **kwargs)

    @staticmethod
    def factory(dim_v=None, dim_h=None, **kwargs):
        '''Convenience factory method'''
        return RBM(dim_v, dim_h, **kwargs)

    def set_params(self):
        W = norm_weight(self.dim_v, self.h_dist.dim)
        b = np.zeros((self.dim_v,)).astype(floatX)
        log_Z = (self.dim_h * np.log(2.)).astype(floatX)
        std_log_Z = (np.array(0.)).astype(floatX)

        self.params = OrderedDict(W=W, b=b, log_Z=log_Z, std_log_Z=std_log_Z)

    def get_params(self):
        return [self.W, self.b] + self.h_dist.get_params()

    def pv_h(self, h):
        '''Function for probility of v given h'''
        return self.step_pv_h(h, self.W, self.b)

    def ph_v(self, x):
        '''Function for probability of h given v'''
        return self.step_ph_v(x, self.W, *self.h_dist.get_params())

    def step_pv_h(self, h, W, b):
        '''Step function for probility of v given h'''
        return eval(self.v_act)(T.dot(h, W.T) + b)

    def step_sv_h(self, r, h, W, b):
        '''Step function for samples from v given h'''
        p = self.step_pv_h(h, W, b)
        return (r <= p).astype(floatX), p

    def step_ph_v(self, x, W, *params):
        '''Step function for probability of h given v'''
        z = T.dot(x, W) + self.h_dist.get_energy_bias(z)
        p = self.h_dist(z)
        return p

    def step_sh_v(self, r, x, W, c):
        '''Step function for sampling h given v'''
        p = self.step_ph_v(x, W, c)
        return (r <= p).astype(floatX), p

    def step_gibbs(self, r_h, r_v, h, W, b, c):
        '''Step Gibbs sample'''
        v, pv = self.step_sv_h(r_v, h, W, b)
        h, ph = self.step_sh_v(r_h, v, W, c)
        return h, v, ph, pv

    def sample(self, h0, n_steps=1):
        '''Gibbs sampling function.

        Sampling starts from hidden state (arbitrary design choice).

        Args:
            h0: T.tensor, Initial hidden layer state.
            n_steps: int, number of Gibbs steps.
        Returns:
            results: OrderedDict, all of the visible and hidden states as well
                as the probability densities from Gibbs sampling.
            updates: OrderedUpdates, from scan
        '''

        r_vs = self.trng.uniform(size=(n_steps, h0.shape[0], self.dim_v), dtype=floatX)
        r_hs = self.trng.uniform(size=(n_steps, h0.shape[0], self.dim_h), dtype=floatX)

        seqs = [r_hs, r_vs]
        outputs_info = [h0, None, None, None]
        non_seqs = self.get_params()

        if n_steps == 1:
            inps = seqs + outputs_info[:1] + non_seqs
            hs, vs, phs, pvs = self.step_gibbs(*inps)
            updates = theano.OrderedUpdates()
        else:
            (hs, vs, phs, pvs), updates = scan(
                self.step_gibbs, seqs, outputs_info, non_seqs, n_steps,
                name=self.name+'_sample', strict=False)

        results = OrderedDict(vs=vs, hs=hs, pvs=pvs, phs=phs)

        return results, updates

    def reconstruct(self, x):
        '''Reconstruction error (cross entropy).

        Performs one step of Gibbs.

        Args:
            x: T.tensor, input
        Returns:
            pv: T.tensor, visible conditional probability density from
                hidden sampled from p(h | x)
        '''
        r = self.trng.uniform(
            size=(x.shape[0], self.dim_h),
            dtype=floatX)

        h, ph = self.step_sh_v(r, x, self.W, self.c)
        pv = self.step_pv_h(h, self.W, self.b)

        return pv

    def estimate_nll(self, X):
        '''Estimate the NLL using the estimate of log_Z.'''
        fe = self.free_energy(X)
        return fe.mean() + self.log_Z

    def step_gibbs_ais(self, r_h_a, r_h_b, r_v, v, beta,
                       W_a, b_a, c_a, W_b, b_b, c_b):
        '''Step Gibbs sample for AIS.

        Gibbs sampling for the transition operator that keeps p*_{k-1} invariant.

        Args:
            r_h_a: T.tensor, random tensor for sampling h_a.
            r_h_b: T.tensor, random tensor for sampling h_b.
            r_v: T.tensor, random tensor for sampling v.
            v: T.tensor, input to T_{k-1}(.|v_{k-1})
            beta: float, annealing factor.
            W_a, b_a, c_a: T.tensor, parameters of RBM a.
            W_b, b_b, c_b: T.tensor, parameters of RBM b.
        Returns:
            v: T.tensor, sample v_k \sim T_{k-1}(.|v_{k-1}).
        '''
        W_a = (1 - beta) * W_a
        b_a = (1 - beta) * b_a
        c_a = (1 - beta) * c_a

        W_b = beta * W_b
        b_b = beta * b_b
        c_b = beta * c_b

        h_a, _ = self.step_sh_v(r_h_a, v, W_a, c_a)
        h_b, _ = self.step_sh_v(r_h_b, v, W_b, c_b)

        pv_act = T.dot(h_a, W_a.T) + T.dot(h_b, W_b.T) + b_a + b_b
        pv = eval(self.v_act)(pv_act)
        v = (r_v <= pv).astype(floatX)

        return v

    def update_partition_function(self, K=10000, M=100):
        '''Updates the partition function.

        Args:
            K: int, number of AIS steps.
            M: int, number of AIS runs.
        Returns:
            results: OrderedDict, results from AIS.
            updates: OrderedUpdates, updates for partition function.
        '''
        log_za, d_logz, var_dlogz, log_ws, samples = self.ais(K, M)
        updates = theano.OrderedUpdates([
            (self.log_Z, log_za + d_logz),
            (self.std_log_Z, T.sqrt(var_dlogz))])
        results = OrderedDict(
            log_za=log_za,
            d_logz=d_logz,
            var_dlogz=var_dlogz,
            log_ws=log_ws
        )
        return results, updates

    def ais(self, K, M):
        '''Performs AIS to estimate the log of the partition function, Z.

        Args:
            K, int. Number of annealing steps.
            M: int. Number of annealing runs.
        Returns:
            log_za: T.tensor.
            d_logz: T.tensor.
            var_dlogz: T.tensor.
            log_ws: T.tensor, log weights.
            xs[-1]: T.tensor, samples.
        '''

        def free_energy(x, beta, *params):
            '''Calculates the free energy from the annealed distribution.'''
            fe_a = self.step_free_energy(x, 1. - beta, *(params[:3]))
            fe_b = self.step_free_energy(x, beta, *(params[3:]))
            return fe_a + fe_b

        def get_beta(k):
            return (k / float(K)).astype(floatX)

        def step_anneal(r_h_a, r_h_b, r_v, k, log_w, x, *params):
            '''Step annealing function for scan.'''
            beta_ = get_beta(k - 1)
            beta  = get_beta(k)
            log_w = log_w + free_energy(x, beta_, *params) - free_energy(x, beta, *params)
            x = self.step_gibbs_ais(r_h_a, r_h_b, r_v, x, beta, *params)
            return log_w, x

        # Random numbers for scan
        r_vs   = self.trng.uniform(size=(K, M, self.dim_v), dtype=floatX)
        r_hs_a = self.trng.uniform(size=(K, M, self.dim_h), dtype=floatX)
        r_hs_b = self.trng.uniform(size=(K, M, self.dim_h), dtype=floatX)

        # Parameters for RBM a and b
        W_a = T.zeros_like(self.W).astype(floatX)
        b_a = -T.log(1. / self.mean_image - 1.)
        c_a = T.zeros_like(self.c).astype(floatX)
        params = [W_a, b_a, c_a] + self.get_params()

        # x0 and log_w0
        p0     = T.tile(1. / (1. + T.exp(-b_a)), (M, 1))
        r      = self.trng.uniform(size=(M, self.dim_v), dtype=floatX)
        x0     = (r <= p0).astype(floatX)
        log_w0 = T.zeros((M,)).astype(floatX)

        seqs         = [r_hs_a, r_hs_b, r_vs, T.arange(1, K + 1)]
        outputs_info = [log_w0, x0]
        non_seqs     = params

        (log_ws, xs), updates = scan(
            step_anneal, seqs, outputs_info, non_seqs, K,
            name=self.name + '_ais', strict=False)

        log_w  = log_ws[-1]
        d_logz = T.log(T.sum(T.exp(log_w - log_w.max()))) + log_w.max() - T.log(M)
        log_za = self.dim_h * T.log(2.).astype(floatX) + T.log(1. + T.exp(b_a)).sum()

        var_dlogz = (M * T.exp(2. * (log_w - log_w.max())).sum() /
                     T.exp(log_w - log_w.max()).sum() ** 2 - 1.)

        return log_za, d_logz, var_dlogz, log_ws, xs[-1]

    def step_free_energy(self, x, beta, W, b, c):
        '''Step free energy function.'''
        vis_term = beta * T.dot(x, b)
        hid_act = beta * (T.dot(x, W) + c)
        fe = -vis_term - T.log(1. + T.exp(hid_act)).sum(axis=1)
        return fe

    def free_energy(self, x):
        '''Free energy function'''
        if x.ndim == 3:
            reduce_dims = (x.shape[0], x.shape[1])
            x = x.reshape((reduce_dims[0] * reduce_dims[1], x.shape[2]))
        else:
            reduce_dims = None
        fe = self.step_free_energy(x, 1., *self.get_params())

        if reduce_dims is not None:
            fe = fe.reshape(reduce_dims)

        return fe

    def energy(self, v, h):
        '''Energy of a visible, hidden configuration.'''

        if v.ndim == 3:
            reduce_dims = (v.shape[0], v.shape[1])
            v = v.reshape((reduce_dims[0] * reduce_dims[1], v.shape[2]))
            h = h.reshape((reduce_dims[0] * reduce_dims[1], h.shape[2]))
        else:
            reduce_dims = None

        joint_term = (h[:, None, :] * self.W[None, :, :] * v[:, :, None]).sum(axis=1)
        v_bias_term = (v * self.b[None, :]).sum(axis=1)
        h_bias_term = (h * self.c[None, :])
        energy = -joint_term - v_bias_term[:, None] - h_bias_term

        if reduce_dims is not None:
            energy = energy.reshape((reduce_dims[0], reduce_dims[1], energy.shape[1]))

        return energy

    def neg_log_prob(self, x, p):
        '''Convenience negative log prob function.'''
        p = T.clip(p, 1e-6, 1 - 1e-6)
        return (-x * T.log(p) - (1 - x) * T.log(1 - p)).sum(axis=x.ndim-1)

    def __call__(self, x, h_p=None, n_steps=1, n_chains=10):
        '''Call function.

        Returns results, including generic cost function, and samples from
        Gibbs chain.

        Args:
            x: T.tensor, input visible state.
            h_p: T.tensor (optional), for PCD.
            n_steps: int, number of Gibbs steps.
            n_chains: int (optional), for CD.
        Returns:
            results: OrderedDict, dictionary of results, all nums.
            samples: OrderedDict, dictionary of results, all arrays.
            updates: OrderedUpdates.
            []: (constants, TODO)
        '''
        ph0 = self.step_ph_v(x, self.W, self.c)
        if h_p is None:
            r = self.trng.uniform(size=(x.shape[0], self.dim_h))
            h_p = (r <= ph0).astype(floatX)
        outs, updates = self.sample(h0=h_p, n_steps=n_steps)

        v0 = x
        vk = outs['vs'][-1]

        positive_cost = self.free_energy(v0)
        negative_cost = self.free_energy(vk)
        cost          = positive_cost.mean() - negative_cost.mean()
        fe            = self.free_energy(v0)
        recon_error   = self.neg_log_prob(x, self.reconstruct(x)).mean()

        nll = self.estimate_nll(x)
        results = OrderedDict(
            cost=cost,
            positive_cost=positive_cost.mean(),
            negative_cost=negative_cost.mean(),
            free_energy=fe.mean(),
            nll=nll,
            log_z=self.log_Z,
            std_log_z=self.std_log_Z,
            recon_error=recon_error
        )

        samples = OrderedDict(
            vs=outs['vs'],
            hs=outs['hs'],
            pvs=outs['pvs'],
            phs=outs['phs'],
            positive_cost=positive_cost,
            negative_cost=negative_cost,
        )

        return results, samples, updates, []