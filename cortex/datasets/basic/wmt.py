'''A NLP dataset.

'''

import numpy as np
import re

from .. import BasicDataset
from ...utils import floatX, intX, pi, _rng


class WMT(BasicDataset):
    sp_map = dict(
        amp='%',
        apos='\'',
        gt='>',
        lt='<',
        quot='"'
    )
    
    def __init__(self):
        pass
    
    def get_data(self, source):
        with open(source) as f:
            x = f.read()
            
        sp_chars = np.unique(re.findall(r'&[^&;]*;', x)).tolist()
        sp_chars = [s[1:-1] for s in sp_chars]
        
        for sp_char in sp_chars:
            c = sp_map.get(sp_char, None)
            if c is not None:
                pass
            if sp_char.startswith('#'):
                c = chr(int(sp_char[1:]))
            else:
                raise ValueError(
                    'Cannot handle special character {}'.format(sp_char))
            x = x.replace(sp_char, c)
        
        x_enc = [ord(c) for c in x]
        
        x_