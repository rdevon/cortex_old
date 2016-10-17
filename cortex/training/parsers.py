'''Generic parser

'''

import argparse


def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-a', '--autoname', action='store_true')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser

def make_argument_parser_trainer():
    '''Generic experiment parser for a trainer.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('module', default=None)
    parser.add_argument('experiment', nargs='?', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-g', '--monitor_gradients', action='store_true')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser

def make_argument_parser_test():
    '''Generic experiment parser for testing.

    Takes the experiment directory as the argument in command line.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', default=None)
    parser.add_argument('-m', '--mode', default='valid',
                        help='Dataset mode: valid, test, or train')
    parser.add_argument('-b', '--best', action='store_true',
                        help='Load best instead of last saved model.')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser