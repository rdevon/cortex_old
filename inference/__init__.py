'''
Inference methods.
'''

from air import AIR
from gdir import MomentumGDIR
from rws import RWS


def inference_class(model, inference_method=None, **inference_args):
    if inference_method == 'momentum':
        return MomentumGDIR(model, **inference_args)
    elif inference_method == 'rws':
        return RWS(model, **inference_args)
    elif inference_method == 'air':
        return AIR(model, **inference_args)
    else:
        raise ValueError(inference_method)
