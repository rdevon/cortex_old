def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_argument_parser_trainer()
    args = parser.parse_args()
    exp_dict = set_experiment(args)
    module = ModuleContainer(
        path.abspath(exp_dict.pop('module')),
        preprocessing=exp_dict.pop('preprocessing', None),
        name=exp_dict.pop('name', None),
        out_path=exp_dict.pop('out_path', None))

    module.update(exp_dict)
    show_every = exp_dict.pop('show_every', 10)
    test_every = exp_dict.pop('test_every', 10)
    monitor_gradients = exp_dict.pop('monitor_gradients', False)
    model_to_load = exp_dict.pop('model_to_load', None)

    trainer = Trainer()
    trainer.run(module, show_every=show_every, test_every=test_every,
                model_to_load=model_to_load, monitor_gradients=monitor_gradients)