'''Basic autoencoder script demo.

'''

import cortex
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)


#cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
#cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

cortex.set_path('conv_demo')
cortex.prepare_data('CIFAR', name='data', mode='train',
                    source='$data/basic/cifar-10-batches-py/')
cortex.prepare_data('CIFAR', name='data', mode='valid',
                    source='$data/basic/cifar-10-batches-py/')

dropout = False
bn = False

cortex.prepare_cell('CNN2D', name='encoder',
                    input_shape=cortex._manager.datasets['data']['image_shape'],
                    filter_shapes=((5, 5), (5, 5)),
                    pool_sizes=((2, 2), (2, 2)),
                    n_filters=[128, 512],
                    dim_out=2048,
                    h_act='softplus', batch_normalization=bn, dropout=dropout)
cortex.prepare_cell('RCNN2D', name='decoder', input_shape=(512, 8, 8),
                    filter_shapes=((5, 5), (5, 5)),
                    pool_sizes=((2, 2), (2, 2)),
                    n_filters=[128, 3],
                    h_act='softplus',
                    dim_in=2048,
                    batch_normalization=bn, dropout=dropout)

cortex.add_step('noise.gaussian', 'data.input', name='input_noise', noise=0.01)
cortex.add_step('encoder', 'input_noise.output')
cortex.add_step('decoder', 'encoder.output')
cortex.match_dims('decoder.output', 'data.input')

cortex.add_step('visualization.random_set',
                 X_in='input_noise.output',
                 X_out='decoder.output',
                 Y='data.labels',
                 n_samples=10)

cortex.build()
cortex.profile()

cortex.add_cost('squared_error', Y='data.input', Y_hat='decoder.output')

train_session = cortex.create_session(batch_size=10)
cortex.build_session(test=False)

trainer = cortex.setup_trainer(
    train_session,
    optimizer='rmsprop',
    epochs=1000,
    learning_rate=0.0001
)

#valid_session = cortex.create_session(noise=False)
#cortex.build_session()

evaluator = cortex.setup_evaluator(
    train_session,
    valid_stat='squared_error',
    batch_size=10
)

monitor = cortex.setup_monitor(train_session, modes=['train', 'valid'])

visualizer = cortex.setup_visualizer(train_session, batch_size=10)
visualizer.add('data.autoencoder_visualization',
               inputs='visualization.random_set.outputs',
               name='reconstruction')

cortex.train(eval_every=1)