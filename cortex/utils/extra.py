'''Extra functions not used for learning.

'''

import urllib2
import os
from os import path
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    Timer
)
import zipfile


def download_data(url, out_path):
    '''Downloads the data from a url.

    Arguments:
        url: str. url of the data.
        out_path: str. Output directory or full file path.

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
    '''Safe portable unzip function.

    Arguments:
        source: str. path to zip file
        out_path: str. path to out_file

    '''

    with zipfile.ZipFile(source) as zf:
        for member in zf.infolist():
            # Path traversal defense copied from
            # http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
            words = member.filename.split('/')
            d = out_path
            for word in words[:-1]:
                drive, word = path.splitdrive(word)
                head, word = path.split(word)
                if word in (os.curdir, os.pardir, ''):
                    continue
                d = os.path.join(d, word)
            zf.extract(member, d)