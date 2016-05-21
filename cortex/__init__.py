'''
Setup scripts for Cortex
'''

import readline, glob
from os import path
import sys

from datasets import fetch_basic_data
from utils.tools import get_paths


def complete(text, state):
    return (glob.glob(text + '*') + [None])[state]

def write_path_conf(data_path, out_path):
    d = path.expanduser('~')
    with open(path.join(d, '.cortexrc'), 'w') as f:
        f.write('[PATHS]\n')
        f.write('$data: %s\n' % path.abspath(data_path))
        f.write('$outs: %s\n' % path.abspath(out_path))

def query_yes_no(question, default='yes'):
    '''Ask a yes/no question via raw_input() and return their answer.

    (Pulled from the web)
    'question' is a string that is presented to the user.
    'default' is the presumed answer if the user just hits <Enter>.
        It must be 'yes' (the default), 'no' or None (meaning
        an answer is required of the user).

    The 'answer' return value is True for 'yes' or False for 'no'.

    '''
    valid = {'yes': True, 'y': True, 'ye': True, 'Y': True, 'Ye': True,
             'no': False, 'n': False, 'N': False, 'No': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError('invalid default answer: `%s`' % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write('Please respond with `yes` or `no` '
                             '(or `y` or `n`).\n')

def main():
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete)
    print ('Welcome to Cortex: a deep learning toolbox for '
            'neuroimaging')
    print ('Cortex requires that you enter some paths for '
            'default dataset and output directories. These '
            'can be changed at any time and are customizable '
            'via the ~/.cortexrc file.')

    try:
        path_dict = get_paths()
    except ValueError:
        path_dict = dict()

    if '$data' in path_dict:
        data_path = raw_input(
            'Default data path: [%s] ' % path_dict['$data']) or path_dict['$data']
    else:
        data_path = raw_input('Default data path: ')
    if not path.isdir(data_path):
        raise ValueError('path %s does not exist. Please create it.')

    if '$outs' in path_dict:
        out_path = raw_input(
            'Default output path: [%s] ' % path_dict['$outs']) or path_dict['$outs']
    else:
        out_path = raw_input('Default output path: ')
    if not path.isdir(data_path):
        raise ValueError('path %s does not exist. Please create it.')
    write_path_conf(data_path, out_path)

    print ('Cortex demos require additional data that is not necessary for '
           'general use of the Cortex as a package.'
           'This includes MNIST, Caltech Silhoettes, and some UCI dataset '
           'samples.')

    answer = query_yes_no('Download basic dataset? ')

    if answer:
        fetch_basic_data()