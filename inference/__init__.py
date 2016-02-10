'''
Inference methods.
'''

from gdir import MomentumGDIR


def inference_class(model, inference_method=None, **inference_args):
    if inference_method == 'momentum':
        return MomentumGDIR(model, **inference_args)
    else:
        raise ValueError(inference_method)
