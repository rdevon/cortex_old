'''Module for ModuleContainer

'''


class ModuleContainer(object):
    __required_methods = ['_build', '_cost']
    __optional_methods = ['_setup', '_data', '_test', '_save', '_viz', '_check',
                          '_eval', '_finish', '_analysis', '_handle_deprecated']
    __required_arguments = ['_learning_args', '_dataset_args']
    _save_fields = ['name', 'preprocessing', 'module_path']

    def __init__(self, module_path, out_path, preprocessing=None, name=None):
        self.logger = logger
        self.preprocessing = preprocessing
        self.preprocessor = Preprocessor(self.preprocessing)

        if name is None:
            name = '.'.join(module_path.split('/')[-1].split('.')[:-1])
        self.name = name
        self.out_path = out_path
        self.module_path = module_path
        module = imp.load_source(self.name, self.module_path)

        for arg in self.__required_arguments + self.__required_methods:
            if not hasattr(module, arg):
                self._raise_no_attribute(arg)
        try:
            self.resolve_dataset = module.resolve_dataset
        except AttributeError:
            self._raise_no_attribute('resolve_dataset')

        self.models = OrderedDict()
        self.args = dict(extra=dict())
        self.dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.inputs = None
        self.set_methods(module)
        self.set_args(module)

    def _raise_no_attribute(self, method):
        required = self.__required_arguments + self.__required_methods
        raise AttributeError('No required `{method}` method or import '
            'found in module {module}. Please check that module {module} '
            'has {required}'.format(method=method, module=self.name,
                                    required=required))

    def check_protected(self, key):
        if key in self.__dict__.keys():
            raise KeyError('Module already has member or method {key}. If {key}'
                           'is a parameter name, please rename it.'.format(
                            key=key))

    def set_methods(self, module):
        for method in self.__required_methods + self.__optional_methods:
            if hasattr(module, method):
                m = method[1:]
                logger.debug('Setting method `%s` from module' % m)
                setattr(self, m, types.MethodType(getattr(module, method), self))

    def set_args(self, module):
        self.args.clear()
        logger.info('Settting arguments')
        try:
            module_arg_keys = module.extra_arg_keys
        except AttributeError:
            module_arg_keys = []

        arg_keys = self.__required_arguments[:] + ['_model_args']
        arg_keys = list(set(arg_keys + module_arg_keys))
        for arg_key in arg_keys:
            self.check_protected(arg_key)
            if not hasattr(module, arg_key) and arg_key != '_model_args':
                raise ImportError('Module %s must define %s'
                                  % (self.name, arg_key))
            args = getattr(module, arg_key)
            if arg_key in ['_learning_args', '_dataset_args', '_model_args']:
                arg_key = arg_key[1:]
            self.args[arg_key] = args
            if arg_key == 'model_args':
                for k, v in args.iteritems():
                    self.__dict__[k] = v
            else:
                self.__dict__[arg_key] = self.args[arg_key]
        logger.debug('Module default arguments are %s'
                     % pprint.pformat(self.args))
        self.learning_args = self.args['learning_args']

    def update(self, exp_dict):
        for key in exp_dict.keys():
            if not key.endswith('_args'):
                continue
            if key in self.args.keys():
                v = exp_dict.pop(key)
                self.args[key].update(**v)
                logger.info('Updating %s arguments with %s'
                             % (key, pprint.pformat(v)))
                if key == 'model_args':
                    self.__dict__.update(**v)


def flatten_component_layers(models, model_dict):
    component_list = []
    def add_component(component):
        if component is None:
            return
        if not isinstance(component, Layer):
            raise TypeError('Components must be a subtype of `Layer` or list '
                            'of `Layer` (%s)' % component)
        if component.name in model_dict.keys():
            raise ValueError('Duplicate key found: %s' % component.name)
        model_dict[component.name] = component
        component_list.append(component)

    for model in models:
        if hasattr(model, '_components'):
            components = [model.__dict__[c] for c in model._components]
            for component in components:
                if isinstance(component, list):
                    for c in component:
                        add_component(c)
                else:
                    add_component(component)
    if len(component_list) > 0:
        flatten_component_layers(component_list, model_dict)

def load_module(model_file, strict=True):
    '''Loads pretrained model.

    Args:
        model_file (str): path to file.
        strict (bool): fail on extra parameters.

    Returns:
        ModuleContainer: module container.
        dict: dictionary of models.
        dict: extra keyword arguments.

    '''

    logger.info('Loading model from %s' % model_file)
    params = np.load(model_file)
    d = dict()
    for k in params.keys():
        try:
            d[k] = params[k].item()
        except ValueError:
            d[k] = params[k]

    try:
        module_path = d.pop('module_path')
        name = d.pop('name')
        preprocessing = d.pop('preprocessing')
    except KeyError:
        raise TypeError('Model file does not contain the appropriate fields '
                        'to be loaded as a module.')
    out_path = '/'.join(module_path.split('/')[:-1])

    module = ModuleContainer(module_path, out_path, name=name,
                             preprocessing=preprocessing)

    pretrained_kwargs = dict()
    arg_kwargs = dict()
    for k, v in d.iteritems():
        if k.endswith('_args'):
            arg_kwargs[k] = v
        else:
            pretrained_kwargs[k] = v
    module.update(arg_kwargs)

    setup(module)
    set_data(module)
    make_inputs(module)
    build(module)
    module.handle_deprecated(pretrained_kwargs)

    logger.info('Pretrained model(s) has the following parameters: \n%s'
          % pprint.pformat(pretrained_kwargs.keys()))

    flatten_component_layers(module.models.values(), module.models)

    for model in module.models.values():
        if model is None:
            continue
        logger.info('---Loading params for %s' % model.name)
        for k, v in model.params.iteritems():
            try:
                param_key = _p(model.name, k)
                pretrained_v = pretrained_kwargs.pop(param_key)
                logger.info('Found %s for %s %s'
                            % (k, model.name, pretrained_v.shape))
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match: %s vs %s'
                    % (model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                try:
                    param_key = '{key}'.format(key=k)
                    pretrained_v = pretrained_kwargs[param_key]
                    logger.info('Found %s, but name is ambiguous' % k)
                    assert model.params[k].shape == pretrained_v.shape, (
                        'Sizes do not match: %s vs %s'
                        % (model.params[k].shape, pretrained_v.shape)
                    )
                    model.params[k] = pretrained_v
                except KeyError:
                    logger.info('{} not found'.format(k))

    if len(pretrained_kwargs) > 0 and strict:
        raise ValueError('ERROR: Leftover params: %s' %
                         pprint.pformat(pretrained_kwargs.keys()))
    elif len(pretrained_kwargs) > 0:
        logger.warn('Leftover params: %s' %
                      pprint.pformat(pretrained_kwargs.keys()))

    set_tparams(module)
    return module