'''Module for testing trainer.

'''

import cortex


manager = cortex.manager

def test_build():
    manager.reset()

    manager.prepare_data(
        'dummy', name='data', mode='train', n_samples=103, data_shape=(131,))
    manager.prepare_data(
        'dummy', name='data', mode='valid', n_samples=31, data_shape=(131,))

    manager.prepare_cell(
        'MLP', name='mlp1', dim_hs=[43, 27], weight_noise=0.01, dropout=0.5)
    manager.prepare_cell(
        'MLP', name='mlp2', dim_in=10, dim_hs=[27, 43], weight_noise=0.01,
        dropout=0.5)

    manager.add_step('mlp1', 'data.input')
    manager.add_step('mlp2', 'mlp1.output')
    manager.match_dims('mlp2.output', 'data.input')
    manager.build()
    manager.add_cost('squared_error', Y_hat='mlp2.output', Y='data.input')

def test_train_step():
    test_build()
    train_session = manager.create_session()
    manager.build_session()

    trainer = manager.setup_trainer(
        manager.get_session(),
        optimizer='sgd',
        epochs=10,
        learning_rate=0.01,
        batch_size=10
    )
    trainer.next_epoch()

def test_eval_step():
    test_train_step()
    valid_session = manager.create_session(noise=False)
    manager.build_session()
    trainer = manager.trainer

    evaluator = manager.setup_evaluator(
        valid_session,
        valid_stat='squared_error'
    )

    results_valid = evaluator(data_mode='valid')
    results_train = evaluator(data_mode='train')

    evaluator.validate(results_valid, trainer.epoch)

    monitor = manager.setup_monitor(modes=['train', 'valid'])
    monitor.add_section('cost', keys=['total_cost']+valid_session.costs.keys())
    monitor.add_section('stats', keys=valid_session.stats.keys())

    monitor.update('train', **results_train)
    monitor.update('valid', **results_valid)

    monitor.display()

def test_epochs():
    test_build()

    train_session = manager.create_session()
    manager.build_session()

    trainer = manager.setup_trainer(
        manager.get_session(),
        optimizer='sgd',
        epochs=100,
        learning_rate=0.01,
        batch_size=10
    )

    valid_session = manager.create_session(noise=False)
    manager.build_session()
    trainer = manager.trainer

    evaluator = manager.setup_evaluator(
        valid_session,
        valid_stat='squared_error'
    )

    monitor = manager.setup_monitor(modes=['train', 'valid'])
    monitor.add_section('cost', keys=['total_cost']+valid_session.costs.keys())
    monitor.add_section('stats', keys=valid_session.stats.keys())

    while True:
        br = False
        try:
            trainer.next_epoch(n_epochs=10)
        except StopIteration:
            br = True
        results_valid = evaluator(data_mode='valid')
        results_train = evaluator(data_mode='train')
        evaluator.validate(results_valid, trainer.epoch)
        monitor.update('train', **results_train)
        monitor.update('valid', **results_valid)
        monitor.display()
        if br:
            break