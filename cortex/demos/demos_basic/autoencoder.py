'''Basic autoencoder script demo.

'''

import cortex
from cortex.utils import logger as cortex_logger


cortex.set_path('demo_AE')
cortex_logger.set_stream_logger(2)


#cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
#cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

cortex.prepare_data('CIFAR', name='data', mode='train',
                    source='$data/basic/cifar-10-batches-py/')
cortex.prepare_data('CIFAR', name='data', mode='valid',
                    source='$data/basic/cifar-10-batches-py/')

cortex.prepare_cell('MLP', name='encoder', dim_hs=[500, 200, 100], dim_out=50,
                     batch_normalization=True,
                     dropout=0.1)
cortex.prepare_cell('MLP', name='decoder', dim_hs=[200, 500, 100],
                     dropout=0.1, out_act='identity',
                     batch_normalization=True)

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
cortex.add_cost('l2_decay', 0.0002, 'encoder.weights')
cortex.add_cost('l2_decay', 0.0002, 'decoder.weights')

train_session = cortex.create_session(batch_size=100)
cortex.build_session()

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
               out_file='$outs/autoencoder_test.png')

cortex.train()
