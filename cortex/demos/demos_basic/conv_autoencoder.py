'''Basic autoencoder script demo.

'''

import cortex
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)


#cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
#cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

cortex.prepare_data('CIFAR', name='data', mode='train',
                    source='$data/basic/cifar-10-batches-py/')
cortex.prepare_data('CIFAR', name='data', mode='valid',
                    source='$data/basic/cifar-10-batches-py/')

cortex.prepare_cell('CNN2D', name='encoder',
                    input_shape=cortex._manager.datasets['data']['image_shape'],
                    filter_shapes=((8, 8), (5, 5), (4, 4)),
                    pool_sizes=((2, 2), (2, 2), (1, 1)),
                    n_filters=[20, 50, 100],
                    h_act='softplus', batch_normalization=True, dropout=0.1)
cortex.prepare_cell('RCNN2D', name='decoder', input_shape=(100, 1, 1),
                    filter_shapes=((4, 4), (5, 5), (5, 5), (3, 3)),
                    pool_sizes=((2, 2), (2, 2), (2, 2), (1, 1)),
                    border_modes=['full', 'full', 'full', 'half'],
                    n_filters=[50, 20, 10, 3],
                    h_act='softplus',
                    batch_normalization=True, dropout=0.1)

cortex.add_step('noise.gaussian', 'data.input', name='input_noise', noise=0.01)
cortex.add_step('encoder', 'input_noise.output')
cortex.add_step('decoder', 'encoder.output')
cortex.match_dims('decoder.output', 'data.input')

cortex.add_step('visualization.random_set',
                 X_in='input_noise.output',
                 X_out='decoder.output',
                 Y='data.labels',
                 n_samples=100)

cortex.build()
cortex.profile()

cortex.add_cost('squared_error', Y='data.input', Y_hat='decoder.output')

train_session = cortex.create_session(batch_size=100)
cortex.build_session(test=True)

trainer = cortex.setup_trainer(
    train_session,
    optimizer='rmsprop',
    epochs=1000,
    learning_rate=0.0001
)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='squared_error'
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

visualizer = cortex.setup_visualizer(valid_session)
visualizer.add('data.autoencoder_visualization',
               inputs='visualization.random_set.outputs',
               out_file='$outs/conv_autoencoder_test.png')

cortex.train(eval_every=1)