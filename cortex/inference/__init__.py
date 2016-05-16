'''
Inference methods.
'''

from .air import AIR, DeepAIR
from .gdir import MomentumGDIR
from .rws import RWS, DeepRWS


def resolve(model, inference_method=None, deep=False, **inference_args):
    if deep:
        if inference_method == 'momentum':
            raise NotImplementedError(inference_method)
            return DeepMomentumGDIR(model, **inference_args)
        elif inference_method == 'rws':
            return DeepRWS(model, **inference_args)
        elif inference_method == 'air':
            return DeepAIR(model, **inference_args)
        elif instance_method is None:
            return None
        else:
            raise ValueError(inference_method)
    else:
        if inference_method == 'momentum':
            return MomentumGDIR(model, **inference_args)
        elif inference_method == 'rws':
            return RWS(model, **inference_args)
        elif inference_method == 'air':
            return AIR(model, **inference_args)
        elif inference_method is None:
            return None
        else:
            raise ValueError(inference_method)
