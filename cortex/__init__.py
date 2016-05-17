'''
Setup scripts for cortex
'''

import readline, glob
from os import path

def complete(text, state):
    return (glob.glob(text + '*') + [None])[state]

def write_path_conf(data_path, out_path):
    d = path.expanduser('~')
    with open(path.join(d, '.cortexrc'), 'w') as f:
        f.write('[PATHS]\n')
        f.write('$data: %s\n' % path.abspath(data_path))
        f.write('$outs: %s\n' % path.abspath(out_path))

def main():
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete)
    print ('Welcome to cortex: a deep learning toolbox for '
            'neuroimaging')
    print ('cortex requires that you enter some paths for '
            'default dataset and output directories. These '
            'can be changed at any time and are customizable '
            'via the ~/.cortexrc file.')
    data_path = raw_input('Default data path: ')
    out_path = raw_input('Default output path: ')
    write_path_conf(data_path, out_path)