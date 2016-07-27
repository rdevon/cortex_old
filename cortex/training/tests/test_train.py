'''Module for testing trainer.

'''

import cortex


manager = cortex.manager

def test_autoencoder_training():
    manager.reset()

    manager.prepare_data(
        'dummy', name='data', batch_size=11, n_samples=103, data_shape=(131,))
    manager.prepare_data(
        'dummy', name='valid', batch_size=11, n_samples=31, data_shape=(131,))

    manager.prepare_cell(
        'MLP', name='mlp1', dim_hs=[43, 27], weight_noise=0.01, dropout=0.5)
    manager.prepare_cell(
        'MLP', name='mlp2', dim_hs=[27, 43], weight_noise=0.01, dropout=0.5)

    manager.add_step('mlp1', 'data.input')
    manager.add_step('mlp2', 'mlp1.output')
    manager.match_dims('mlp2.output', 'fibrous.input')
    manager.build()
    manager.add_cost('squared_error', Y_hat='mlp2.output', Y='data.input')

    train_session = manager.build_session()
    manager.create_session(noise=False)
    manager.build_session()

    manager.setup_trainer(
        train_session,
        optimizer='sgd',
        epochs=10,
        learning_rate=0.01
    )

    manager.setup_tester(
        valid_session,
        valid_key='-log p(x)'
    )

    manager.train()