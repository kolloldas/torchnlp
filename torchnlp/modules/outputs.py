from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crf import CRF

class OutputLayer(nn.Module):
    """
    Abstract base class for output layer. 
    Handles projection to output labels
    """
    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """
    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)

        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))

class CRFOutputLayer(OutputLayer):
    """
    Implements a CRF based output layer
    """
    def __init__(self, hidden_size, output_size):
        super(CRFOutputLayer, self).__init__(hidden_size, output_size)
        self.crf = CRF(output_size)

    def forward(self, hidden):
        feats = self.output_projection(hidden)
        return self.crf(feats)

    def loss(self, hidden, labels):
        feats = self.output_projection(hidden)
        return self.crf.loss(feats, labels)