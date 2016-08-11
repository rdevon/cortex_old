'''Demo for training VAE with VBM data.

Has additional semi-supervised classification.

'''

import cortex
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)

n_posterior_samples = 1
n_posterior_samples_test = 100
n_prior_samples = 100
dim_h = 100
batch_size = 10
distribution = 'gaussian'

cortex.prepare_data_split('MRI', name='data', split=[.8, .1, .1],
                          source='$data/VBM/VBM.yaml')

cortex.prepare_cell('DistributionMLP', name='approx_posterior', dim_hs=[500],
                    h_act='tanh')
cortex.prepare_cell(distribution, name='prior', dim=dim_h)
cortex.prepare_cell('DistributionMLP', name='conditional', dim_hs=[500],
                    h_act='tanh')

cortex.match_dims('prior.P', 'approx_posterior.P')
cortex.match_dims('conditional.P', 'data.input')

cortex.add_step('approx_posterior', 'data.input')
cortex.add_step('conditional', 'approx_posterior.samples')

cortex.prepare_samples('approx_posterior.P', n_posterior_samples)
cortex.prepare_samples('approx_posterior.P', n_posterior_samples_test,
                       name='test_samples')
cortex.prepare_samples('prior', n_prior_samples)

cortex.add_step('conditional', 'prior.samples', name='prior_gen')
cortex.add_step('conditional', 'approx_posterior.test_samples', name='test_gen')
cortex.add_step('prior.permute', scale=2., name='latent_viz')
cortex.add_step('conditional.viz', inputs='latent_viz.outputs', name='cond_viz')

cortex.build()

cortex.add_cost('conditional.negative_log_likelihood', X='data.input')
cortex.add_cost('kl_divergence', P='approx_posterior.P', Q='prior',
                P_samples='approx_posterior.samples',
                cells=['approx_posterior.distribution', 'prior'])
cortex.add_cost('l2_decay', 0.0002, 'approx_posterior.mlp.weights')
cortex.add_cost('l2_decay', 0.0002, 'conditional.mlp.weights')
cortex.add_stat('variational_inference', X='data.input',
                posterior='approx_posterior.P',
                prior='prior', conditional='test_gen.P',
                posterior_samples='approx_posterior.test_samples',
                cells=['conditional.distribution',
                       'approx_posterior.distribution', 'prior'])

train_session = cortex.create_session(batch_size=batch_size)
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    optimizer='rmsprop',
    epochs=1000,
    learning_rate=0.00001
)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

visualizer = cortex.setup_visualizer(valid_session)
visualizer.add('data.viz', 'cond_viz.output', out_file='$outs/vbm_vae_latents.png')

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='variational_inference.lower_bound',
    valid_sign=-1,
    batch_size=10
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

cortex.train(eval_every=1)