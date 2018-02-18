from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import torch.nn as nn
import numpy as np

class HParams(object):
    """
    Holds arbitrary hyperparameters. Converts dict key-values to objects
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        
    def add(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
        return self

    def __repr__(self):
        return '\nHyperparameters:\n' + '\n'.join([' {}={}'.format(k, v) for k,v in self.__dict__.items()])

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**json.load(f))


def hparams_basic():
    return HParams(
        batch_size=100,
        learning_rate=0.2,
        learning_rate_decay=None,
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.98,
    )