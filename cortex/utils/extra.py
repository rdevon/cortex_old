'''Extra functions not used for learning.

'''

import glob
import os
from os import path
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    Timer
)
import sys
import urllib2
import zipfile


def complete_path(text, state):
    '''Completes a path for readline.

    '''
    return (glob.glob(text + '*') + [None])[state]


def download_data(url, out_path):
    '''Downloads the data from a url.

    Args:
        url (str): url of the data.
        out_path (str): Output directory or full file path.

    '''

    if path.isdir(out_path):
        file_name = path.join(out_path, url.split('/')[-1])
    else:
        d = path.abspath(os.path.join(out_path, os.pardir))
        if not path.isdir(d):
            raise IOError('Directory %s does not exist' % d)
        file_name = out_path

    u = urllib2.urlopen(url)
    with open(file_name, 'wb') as f:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])

        file_size_dl = 0
        block_sz = 8192

        widgets = ['Dowloading to %s (' % file_name, Timer(), '): ', Bar()]
        pbar = ProgressBar(widgets=widgets, maxval=file_size).start()

        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            pbar.update(file_size_dl)
    print

def unzip(source, out_path):
    '''Unzip function.

    Arguments:
        source (str): path to zip file
        out_path (str): path to out_file

    '''
    print 'Unzipping %s to %s' % (source, out_path)

    if not zipfile.is_zipfile(source):
        raise ValueError('%s is not a zipfile' % source)

    if not path.isdir(out_path):
        raise ValueError('%s is not a directory' % out_path)

    with zipfile.ZipFile(source) as zf:
        zf.extractall(out_path)

def write_path_conf(data_path, out_path):
    '''Writes basic configure file.

    Args:
        data_path (str): path to data.
        out_path (str): path to outputs.

    '''
    d = path.expanduser('~')
    with open(path.join(d, '.cortexrc'), 'w') as f:
        f.write('[PATHS]\n')
        f.write('$data: %s\n' % path.abspath(data_path))
        f.write('$outs: %s\n' % path.abspath(out_path))

def write_default_theanorc():
    '''Writes default .theanorc file.

    '''
    d = path.expanduser('~')
    with open(path.join(d, '.theanorc'), 'w') as f:
        f.write('[global]\n')
        f.write('floatX = float32')

def query_yes_no(question, default='yes'):
    '''Ask a yes/no question via raw_input() and return their answer.

    Args:
        question (str)
        default (Optional[str])

    Returns:
        str

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