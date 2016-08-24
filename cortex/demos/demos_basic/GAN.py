'''Generative adversarial networks

'''

import theano
from theano import tensor as T

import cortex
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)

batch_size=100

cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

cortex.prepare_cell('gaussian', name='noise', dim=100)
cortex.prepare_cell('DistributionMLP', name='discriminator',
                    distribution_type='binomial', dim_hs=[500, 200],
                    h_act='softplus', dim=1, dropout=0.5)
cortex.add_step('discriminator', 'mnist.input', name='real')
cortex.prepare_cell('MLP', name='generator', out_act='sigmoid',
                    dim_hs=[200, 500], h_act='softplus',
                    batch_normalization=True)
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
cortex.add_cost('discriminator.negative_log_likelihood', X=T.ones(batch_size,),
                P='fake.P', name='generator_cost')
#cortex.add_cost('l2_decay', 0.0002, 'generator.weights', name='generator_l2')
#cortex.add_cost('l2_decay', 0.0002, 'discriminator.mlp.weights',
#                name='discriminator_l2')

train_session = cortex.create_session()
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    optimizer='rmsprop',
    epochs=1000,
    learning_rate=0.001,
    batch_size=100,
)

trainer.set_optimizer(models=['discriminator.mlp'], cost='discriminator_cost')
trainer.set_optimizer(models=['generator'],
                      cost='generator_cost.negative_log_likelihood',
                      freq=10)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='generator_cost.negative_log_likelihood'
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

visualizer = cortex.setup_visualizer(valid_session)
visualizer.add('mnist.viz',
               X='generator.Y',
               out_file='$outs/GAN_test.png')

cortex.train(monitor_grads=True)