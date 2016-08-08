'''Generative adversarial networks

'''

from theano import tensor as T

import cortex

batch_size=100

cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

cortex.prepare_cell('binomial', name='noise', dim=100)
cortex.prepare_cell('DistributionMLP', name='discriminator',
                    distribution_type='binomial', dim_hs=[200],
                    h_act='softplus', dim=1)
cortex.add_step('discriminator', 'mnist.input', name='real')
cortex.prepare_cell('MLP', name='generator', out_act='sigmoid',
                    dim_hs=[200], h_act='softplus')
cortex.prepare_samples('noise', batch_size)

cortex.add_step('generator', 'noise.samples')
cortex.add_step('discriminator', 'generator.Y', name='fake')

cortex.add_step('discriminator._cost',
                P='fake.P', X=T.zeros(batch_size,), name='fake_cost')
cortex.add_step('discriminator._cost',
                P='real.P', X=T.ones(batch_size,), name='real_cost')

cortex.build()
cortex.profile()

cortex.add_cost(lambda x, y: x + y, 'fake_cost.output', 'real_cost.output',
                name='discriminator_cost')
cortex.add_cost(lambda x: -1 * x, 'fake_cost.output', name='generator_cost')
#cortex.add_cost('l2_decay', 0.0002, 'generator.weights', name='generator_l2')
#cortex.add_cost('l2_decay', 0.0002, 'discriminator.mlp.weights',
#                name='discriminator_l2')

train_session = cortex.create_session()
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    optimizer='rmsprop',
    epochs=1000,
    learning_rate=0.1,
    batch_size=100,
    models=['generator', 'discriminator'],
    costs=['generator_cost', 'discriminator_cost']
)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='generator_cost'
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

visualizer = cortex.setup_visualizer(valid_session)
visualizer.add('mnist.viz',
               X='generator.Y',
               out_file='/Users/devon/tmp/GAN_test.png')

cortex.train()