'''Demo for adaptive importance refinement.

'''

import cortex

n_posterior_samples = 10
n_posterior_samples_test = 1000
dim_h = 200

cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

cortex.prepare_cell('DistributionMLP', name='approx_posterior', dim_hs=[500],
                    h_act='softplus')
cortex.prepare_cell('binomial', name='prior', dim=dim_h)
cortex.prepare_cell('DistributionMLP', name='conditional', dim_hs=[500],
                    h_act='softplus')
cortex.prepare_cell('AIR', prior='prior', conditional='conditional',
                    posterior='approx_posterior', name='inference')

cortex.match_dims('prior.P', 'approx_posterior.P')
cortex.match_dims('conditional.P', 'mnist.input')
cortex.match_dims('approx_posterior.samples', 'conditional.input')

cortex.add_step('approx_posterior', 'mnist.input')
cortex.add_step('inference', 'mnist.input', 'approx_posterior.P',
                n_samples=20, n_steps=20, inference_rate=0.1)
cortex.prepare_samples('inference.Qk', n_posterior_samples)
cortex.prepare_samples('inference.Qk', n_posterior_samples_test,
                       name='test_samples')
cortex.add_step('conditional', 'inference.samples')
cortex.prepare_samples('prior', 100)

cortex.add_step('conditional', 'prior.samples', name='prior_gen')
cortex.add_step('conditional', 'inference.test_samples', name='test_gen')

cortex.build()
cortex.profile()

cortex.add_cost('conditional.negative_log_likelihood', X='mnist.input')
cortex.add_cost('kl_divergence', P='inference.Qk', Q='prior',
                P_samples='inference.samples',
                cells=['approx_posterior.distribution', 'prior'])
cortex.add_cost('approx_posterior.negative_log_likelihood', X='inference.samples')
cortex.add_cost('l2_decay', 0.0002, 'approx_posterior.mlp.weights')
cortex.add_cost('l2_decay', 0.0002, 'conditional.mlp.weights')
cortex.add_stat('variational_inference', X='mnist.input',
                posterior='inference.Qk',
                prior='prior', conditional='test_gen.P',
                posterior_samples='inference.test_samples',
                cells=['conditional.distribution',
                       'approx_posterior.distribution', 'prior'])

train_session = cortex.create_session()
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    optimizer='rmsprop',
    epochs=1000,
    learning_rate=0.001,
    batch_size=100
)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

visualizer = cortex.setup_visualizer(
    valid_session,
    batch_size=10)
visualizer.add('mnist.autoencoder_visualization',
               X_in='mnist.input',
               X_out='conditional.P',
               Y='mnist.labels',
               out_file='$outs/vae_recons_test.png')
visualizer.add('mnist.viz',
               X='prior_gen.P',
               out_file='$outs/vae_prior.png')

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='variational_inference.lower_bound',
    valid_sign=-1,
    batch_size=100
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

cortex.train(eval_every=1)