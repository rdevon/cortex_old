"""
Tests the mlp_mnist.py demo.
"""

from cortex.utils.trainer import ModuleContainer, Trainer
from cortex.utils.tools import resolve_path


def test_mlp_mnist(epochs=5):
    module = ModuleContainer('../mlp_mnist.py',
                             resolve_path('$outs'))
    module.learning_args['epochs'] = epochs

    trainer = Trainer()
    trainer.run(module)
