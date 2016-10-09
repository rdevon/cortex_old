'''Module for RBM class.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Cell, norm_weight
from .distributions.binomial import Binomial
from ..utils import concatenate, floatX, scan


class RBM(Cell):
    '''
    RBM class.

    Currently supports only binary hidden units.

    Attributes:
        h_dist (Distribution): conditional distribution of hiddens.
        v_dist (Distribution): conditional distribution of visibles.
        dim_h (int): number of hidden units.
        dim_v (int): number of visible units.
        W (T.tensor): weights
        log_Z (T.tensor): current approximation of the log marginal.
        std_log_Z (T.tensor): current std of the approximate log marginal.
        mean_image (T.tensor): used for marginal approximation.

    '''
    _required = ['dim_in', 'dim_h']
    _components = {
        'h_dist': {
            'cell_type': '&h_dist_type',
            'dim': '&dim_h'
        },
        'v_dist': {
            'cell_type': '&v_dist_type',
            'dim': '&dim_in'
        }
    }
    _args = ['dim_in', 'dim_h', 'h_dist_type', 'v_dist_type']
    _dim_map = {'X': 'dim_in', 'input': 'dim_in', 'Y': 'dim_h',
                'output': 'dim_h'}
    _weights = ['W']
    
    def __init__(self, dim_in, dim_h, mean_image=None, name='rbm',
                 v_dist_type='binomial', h_dist_type='binomial',
                 n_persistent_chains=0, **kwargs):
        '''Init method for RBM class.

        Args:
            dim_v (int): number of visible layer units
            dim_h (int): number of hidden layer units
            mean_image (Optional[numpy.array]): used for marginal approximation.
                if None, then set to 0.5.

        '''
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.h_dist_type = h_dist_type
        self.v_dist_type = v_dist_type
        
        if mean_image is None:
            mean_image = np.zeros(self.dim_in,).astype(floatX) + 0.5
        self.mean_image = theano.shared(
            np.clip(mean_image, 1e-7, 1 - 1e-7), name='mean_image')
        
        self.log_Z = theano.shared(
            (self.dim_h * np.log(2.)).astype(floatX), name='log_Z')
        self.std_log_Z = theano.shared(
            np.array(0.).astype(floatX), name='std_log_Z')
        
        if n_persistent_chains > 0:
            self.H_p = theano.shared(
                np.zeros((n_persistent_chains, self.dim_h)).astype(floatX),
                name='H_p')
        else:
            self.H_p = None

        super(RBM, self).__init__(name=name, **kwargs)

    def init_params(self):
        W = norm_weight(self.dim_in, self.dim_h)
        self.params = OrderedDict(W=W)

    def pv_h(self, h):
        '''Function for probility of v given h

        Args:
            h (T.tensor): hidden state.

        Returns:
            T.tensor: conditional visible probability

        '''
        return self.step_pv_h(h, *self.get_params())

    def ph_v(self, x):
        '''Function for probability of h given v.

        Args:
            x (T.tensor): visible state.

        Returns:
            T.tensor: conditional hidden probability.

        '''
        return self.step_ph_v(x, *self.get_params())

    def step_pv_h(self, h, *params):
        '''Step function for cacluating probility of v given h.

        Args:
            h (T.tensor): hidden state.
            *params: theano shared variables.

        Returns:
            T.tensor: conditional visible probability.

        '''
        W = params[0]
        h_params = self.select_params('h_dist', *params)
        v_params = self.select_params('v_dist', *params)
        
        h = self.h_dist.scale_for_energy_model(h, *h_params)
        center = T.dot(h, W.T) + v_params[0]
        return self.v_dist.get_prob(*([center] + [v[None, :] for v in v_params[1:]]))

    def step_ph_v(self, x, *params):
        '''Step function for probability of h given v.

        Args:
            x (T.tensor): visible state.
            *params: theano shared variables.

        Returns:
            T.tensor: conditional hidden probability.

        '''
        W = params[0]
        h_params = self.select_params('h_dist', *params)
        v_params = self.select_params('v_dist', *params)
        
        x = self.v_dist.scale_for_energy_model(x, *v_params)
        center = T.dot(x, W)
        center += self.h_dist.get_center(*h_params)
        return self.h_dist.get_prob(center, *(h_params[1:]))

    def step_sv_h(self, r, h, *params):
        '''Step function for samples from v given h.

        Args:
            r (theano.randomstream): random variables.
            h (T.tensor): hidden state.
            *params: theano shared variables.

        Returns:
            T.tensor: samples.
            T.tensor: conditional visible probability.

        '''
        p = self.step_pv_h(h, *params)
        return self.v_dist._sample(r, p), p

    def step_sh_v(self, r, x, *params):
        '''Step function for sampling h given v.

        Args:
            r (theano.randomstream): random variables.
            x (T.tensor): visible state.
            *params: theano shared variables.

        Returns:
            T.tensor: samples.
            T.tensor: conditional hidden probability.

        '''
        p = self.step_ph_v(x, *params)
        return self.h_dist._sample(r, p), p

    def step_gibbs(self, r_h, r_v, h, *params):
        '''Step Gibbs sample.

        Args:
            r_h (theano.randomstream): random variables for hiddens.
            r_v (theano.randomstream): random variables for visibles.
            h (T.tensor): hidden state.
            *params: theano shared variables.

        Returns:
            T.tensor: hidden samples.
            T.tensor: visible samples.
            T.tensor: conditional hidden probability.
            T.tensor: conditional visible probability.

        '''
        v, pv = self.step_sv_h(r_v, h, *params)
        h, ph = self.step_sh_v(r_h, v, *params)
        return h, v, ph, pv

    def sample(self, h0, n_steps=1):
        '''Gibbs sampling function.

        Sampling starts from hidden state (arbitrary design choice).

        Args:
            h0 (T.tensor): Initial hidden layer state.
            n_steps (int): number of Gibbs steps.

        Returns:
            OrderedDict: all of the visible and hidden states as well as the \
                probability densities from Gibbs sampling.
            theano.OrderedUpdates: updates.

        '''

        r_vs = self.v_dist.generate_random_variables(
            shape=(n_steps, h0.shape[0]))
        r_hs = self.h_dist.generate_random_variables(
            shape=(n_steps, h0.shape[0]))
        seqs = [r_hs, r_vs]
        outputs_info = [h0, None, None, None]
        non_seqs = self.get_params()

        if n_steps == 1:
            inps = [s[0] for s in seqs] + outputs_info[:1] + non_seqs
            hs, vs, phs, pvs = self.step_gibbs(*inps)
            updates = theano.OrderedUpdates()
            vs = T.shape_padleft(vs)
            hs = T.shape_padleft(hs)
            phs = T.shape_padleft(phs)
            pvs = T.shape_padleft(pvs)
        else:
            (hs, vs, phs, pvs), updates = scan(
                self.step_gibbs, seqs, outputs_info, non_seqs, n_steps,
                name=self.name+'_sample', strict=False)

        return OrderedDict(vs=vs, hs=hs, pvs=pvs, phs=phs, r_vs=r_vs,
                           r_hs=r_hs, updates=updates)

    def reconstruct(self, x):
        '''Reconstruction error (cross entropy).

        Performs one step of Gibbs.

        Args:
            x (T.tensor): input

        Returns:
            pv (T.tensor): visible conditional probability density from
                hidden sampled from :math:`p(h | x)`

        '''
        r = self.h_dist.generate_random_variables(shape=(x.shape[0],))
        h, ph = self.step_sh_v(r, x, *self.get_params())
        pv = self.step_pv_h(h, *self.get_params())

        return pv

    def estimate_nll(self, X):
        '''Estimate the :math:`-\log p(x)` using the estimate of :math:`log_Z`.

        Args:
            X (T.tensor): data samples.

        Returns:
            T.tensor: NLL estimate.

        '''
        fe = self.free_energy(X)
        return fe.mean() + self.log_Z

    def step_gibbs_ais(self, r_h_a, r_h_b, r_v, v, beta,
                       W_a, b_a, c_a, W_b, b_b, c_b):
        '''Step Gibbs sample for AIS.

        Gibbs sampling for the transition operator that keeps
        :math:`p^{\star}_{k-1}` invariant.

        Note:
            Only works for Binomial / Binomial RBMs

        Args:
            r_h_a (T.tensor): random tensor for sampling h_a.
            r_h_b (T.tensor): random tensor for sampling h_b.
            r_v (T.tensor): random tensor for sampling v.
            v (T.tensor) input to :math:`T_{k-1}(.|v_{k-1})`.
            beta (float): annealing factor.
            W_a, b_a, c_a (T.tensor): parameters of RBM a.
            W_b, b_b, c_b (T.tensor): parameters of RBM b.

        Returns:
            T.tensor: sample :math:`v_k \sim T_{k-1}(.|v_{k-1})`.

        '''
        W_a = (1 - beta) * W_a
        b_a = (1 - beta) * b_a
        c_a = (1 - beta) * c_a
        
        W_b = beta * W_b
        b_b = beta * b_b
        c_b = beta * c_b

        h_a, _ = self.step_sh_v(r_h_a, v, W_a, b_a, c_a)
        h_b, _ = self.step_sh_v(r_h_b, v, W_b, b_b, c_b)

        pv_act = T.dot(h_a, W_a.T) + T.dot(h_b, W_b.T) + b_a + b_b
        pv = self.v_dist._act(pv_act)
        v = self.v_dist._sample(r_v, P=pv)

        return v

    def update_partition_function(self, K=10000, M=100):
        '''Updates the partition function.

        Note:
            Only works for Binomial / Binomial.

        Args:
            K (int): number of AIS steps.
            M (int): number of AIS runs.

        Returns:
            OrderedDict: results from AIS.
            OrderedUpdates: updates for partition function.

        '''

        if not (isinstance(self.v_dist, Binomial) and isinstance(self.h_dist, Binomial)):
            raise NotImplementedError('Only binomial / binomial RBM supported for AIS.')

        results = self.ais(K, M)
        updates = theano.OrderedUpdates([
            (self.log_Z, results['log_za'] + results['d_logz']),
            (self.std_log_Z, T.sqrt(results['var_dlogz']))])
        return results, updates

    def ais(self, K, M):
        '''Performs AIS to estimate the log of the partition function, Z.

        Note:
            Only works for Binomial / Binomial.

        Args:
            K (int): Number of annealing steps.
            M (int): Number of annealing runs.

        Returns:
            T.tensor: :math:`log Z_a`.
            T.tensor: :math:`d \log Z`.
            T.tensor: variance of :math:`d \log Z`.
            T.tensor: log weights.
            T.tensor: samples.

        '''

        if not (isinstance(self.v_dist, Binomial) and isinstance(self.h_dist, Binomial)):
            raise NotImplementedError('Only binomial / binomial RBM supported for AIS.')

        def free_energy(x, beta, *params):
            '''Calculates the free energy from the annealed distribution.

            Args:
                x (T.tensor): samples.
                beta (int): beta constant (for annealing).
                *params: shared variables.

            Returns:
                T.tensor: free energy.

            '''
            fe_a = self.step_free_energy(x, 1. - beta, *(params[:3]))
            fe_b = self.step_free_energy(x, beta, *(params[3:]))
            return fe_a + fe_b

        def get_beta(k):
            return (k / float(K)).astype(floatX)

        def step_anneal(r_h_a, r_h_b, r_v, k, log_w, x, *params):
            beta_ = get_beta(k - 1)
            beta  = get_beta(k)
            log_w = log_w + free_energy(x, beta_, *params) - free_energy(x, beta, *params)
            x = self.step_gibbs_ais(r_h_a, r_h_b, r_v, x, beta, *params)
            return log_w, x

        rval = OrderedDict()

        # Random numbers for scan
        r_vs = self.v_dist.generate_random_variables(shape=(K, M))
        r_hs_a = self.h_dist.generate_random_variables(shape=(K, M))
        r_hs_b = self.h_dist.generate_random_variables(shape=(K, M))
        rval.update(r_vs=r_vs, r_hs_a=r_hs_a, r_hs_b=r_hs_b)

        # Parameters for RBM a and b
        W_a = T.zeros_like(self.W).astype(floatX)
        b_a = -T.log(1. / self.mean_image - 1.)
        c_a = T.zeros_like(self.h_dist.z).astype(floatX)
        params = [W_a, b_a, c_a] + self.get_params()

        # x0 and log_w0
        p0     = T.tile(1. / (1. + T.exp(-b_a)), (M, 1))
        r      = self.trng.uniform(size=(M, self.v_dist.dim), dtype=floatX)
        x0     = (r <= p0).astype(floatX)
        log_w0 = T.zeros((M,)).astype(floatX)
        rval.update(p0=p0, r=r, x0=x0, log_w0=log_w0)

        seqs         = [r_hs_a, r_hs_b, r_vs, T.arange(1, K + 1)]
        outputs_info = [log_w0, x0]
        non_seqs     = params

        (log_ws, xs), updates = scan(
            step_anneal, seqs, outputs_info, non_seqs, K,
            name=self.name + '_ais', strict=False)
        rval.update(samples=xs[-1])

        log_w  = log_ws[-1]
        d_logz = T.log(T.sum(T.exp(log_w - log_w.max()))) + log_w.max() - T.log(M)
        log_za = self.h_dist.dim * T.log(2.).astype(floatX) + T.log(1. + T.exp(b_a)).sum()
        rval.update(log_w=log_w, d_logz=d_logz, log_za=log_za)

        var_dlogz = (M * T.exp(2. * (log_w - log_w.max())).sum() /
                     T.exp(log_w - log_w.max()).sum() ** 2 - 1.)
        rval['var_dlogz'] = var_dlogz

        return rval

    def step_free_energy(self, x, beta, *params):
        '''Step free energy function.

        Args:
            x (T.tensor): data sample.
            beta (float): beta value for annealing.
            *params: theano shared variables.

        Returns:
            T.tensor: free energy.

        '''
        W = params[0]
        h_params = self.select_params('h_dist', *params)
        v_params = self.select_params('v_dist', *params)

        vis_term = beta * self.v_dist.get_energy_bias(x, *v_params)
        x = self.v_dist.scale_for_energy_model(x, *v_params)
        hid_act = beta * (T.dot(x, W) + self.h_dist.get_center(*h_params))
        fe = -vis_term - T.log(1. + T.exp(hid_act)).sum(axis=1)
        return fe

    def step_free_energy_h(self, h, beta, *params):
        '''Step free energy function for hidden states.

        Args:
            h (T.tensor): hidden sample.
            beta (float): beta value for annealing.
            *params: theano shared variables.

        Returns:
            T.tensor: free energy.

        '''
        W = params[0]
        h_params = self.select_params('h_dist', *params)
        v_params = self.select_params('v_dist', *params)

        hid_term = beta * self.h_dist.get_energy_bias(h, *h_params)
        h = self.h_dist.scale_for_energy_model(h, *h_params)
        vis_act = beta * (T.dot(h, W.T) + self.v_dist.get_center(*v_params))
        fe = -hid_term - T.log(1. + T.exp(vis_act)).sum(axis=1)
        return fe

    def free_energy(self, x):
        '''Free energy function.

        Args:
            x (T.tensor): data sample.

        Returns:
            T.tensor: free energy.

        '''
        if x.ndim == 3:
            reduce_dims = (x.shape[0], x.shape[1])
            x = x.reshape((reduce_dims[0] * reduce_dims[1], x.shape[2]))
        else:
            reduce_dims = None
        fe = self.step_free_energy(x, 1., *self.get_params())
        if reduce_dims is not None: fe = fe.reshape(reduce_dims)

        return fe

    def free_energy_h(self, h):
        '''Free energy function for hidden states.

        Args:
            h (T.tensor): hidden sample.

        Returns:
            T.tensor: free energy.

        '''
        if h.ndim == 3:
            reduce_dims = (h.shape[0], h.shape[1])
            h = h.reshape((reduce_dims[0] * reduce_dims[1], h.shape[2]))
        else:
            reduce_dims = None
        fe = self.step_free_energy_h(h, 1., *self.get_params())
        if reduce_dims is not None: fe = fe.reshape(reduce_dims)

        return fe

    def energy(self, v, h):
        '''Energy of a visible, hidden configuration.

        Args:
            v (T.tensor): visible sample.
            h (T.tensor): hidden sample.

        Returns:
            T.tensor: energies.

        '''

        if v.ndim == 3:
            reduce_dims = (v.shape[0], v.shape[1])
            v = v.reshape((reduce_dims[0] * reduce_dims[1], v.shape[2]))
            h = h.reshape((reduce_dims[0] * reduce_dims[1], h.shape[2]))
        else:
            reduce_dims = None

        v_params = self.v_dist.get_params()
        h_params = self.h_dist.get_params()

        v_bias_term = self.v_dist.get_energy_bias(x, *v_params)
        h_bias_term = self.h_dist.get_energy_bias(h, *h_params)

        v = self.v_dist.scale_for_energy_model(v, *v_params)
        h = self.h_dist.scale_for_energy_model(h, *h_params)
        joint_term = (h[:, None, :] * self.W[None, :, :] * v[:, :, None]).sum(axis=1)
        energy = -joint_term - v_bias_term[:, None] - h_bias_term

        if reduce_dims is not None:
            energy = energy.reshape((reduce_dims[0], reduce_dims[1], energy.shape[1]))

        return energy

    def v_neg_log_prob(self, x, p):
        '''Convenience negative log prob function.

        Args:
            x (T.tensor): visible sample.
            p (T.tensor): probability.

        Returns:
            T.tensor: negative log probability.

        '''
        return self.v_dist.neg_log_prob(x, p)

    def h_neg_log_prob(self, h, p):
        '''Convenience negative log prob function.

        Args:
            h (T.tensor): hidden sample.
            p (T.tensor): probability.

        Returns:
            T.tensor: negative log probability.

        '''
        return self.h_dist.neg_log_prob(h, p)
    
    def _cost(self, X=None, V0=None, Vk=None):
        positive_cost = self.free_energy(V0)
        negative_cost = self.free_energy(Vk)
        cost          = positive_cost.mean() - negative_cost.mean()
        fe            = self.free_energy(V0)
        recon_error   = self.v_neg_log_prob(X, self.reconstruct(X)).mean()
        results = OrderedDict()

        try:
            nll = self.estimate_nll(X)
            results['nll'] = nll
        except NotImplementedError:
            pass
        
        results.update(**OrderedDict(
            cost=cost,
            positive_cost=positive_cost.mean(),
            negative_cost=negative_cost.mean(),
            free_energy=fe.mean(),
            log_z=self.log_Z,
            std_log_z=self.std_log_Z,#._as_TensorVariable(),
            recon_error=recon_error)
        )
        return results
    
    def init_args(self, X, persistent=False, n_steps=1):
        if persistent:
            H_p = self.H_p
        else:
            pH0 = self.step_ph_v(X, *self.get_params())
            H_p = self.h_dist.simple_sample(X.shape[0], P=pH0)
        return (X, H_p, n_steps)

    def _feed(self, X, H_p, n_steps, *params):
        pH0 = self.step_ph_v(X, *params)
        outs = self.sample(h0=H_p, n_steps=n_steps)

        V0 = X
        Vk = outs['vs'][-1]

        results = OrderedDict(
            pH0=pH0,
            V0=V0,
            H_p=H_p,
            Vk=Vk,
        )
        results.update(**outs)
        if self.H_p is not None:
            updates = theano.OrderedUpdates([(self.H_p, outs['hs'][-1])])
            results['updates'] = updates
        return results
    
    def update_partition(self, k=2000):
        try:
            results, updates = self.update_partition_function(K=k)
        except NotImplementedError:
            updates = theano.OrderedUpdates()
    
        return updates
    
_classes = {'RBM': RBM}
