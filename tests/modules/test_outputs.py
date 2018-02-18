from torchnlp.modules.outputs import SoftmaxOutputLayer, CRFOutputLayer

import torch
from torch.autograd import Variable
import numpy as np

def test_softmax_output_layer():
    hidden = Variable(torch.randn(2, 3, 4))
    labels = Variable(torch.ones(2, 3).long())
    softmax = SoftmaxOutputLayer(4, 8)
    loss = softmax.loss(hidden, labels)
    assert loss.shape[0] == 1
    pred = softmax(hidden)
    assert pred.shape == (2, 3)

def test_crf_output_layer():
    hidden = Variable(torch.randn(2, 3, 4))
    labels = Variable(torch.ones(2, 3).long())
    crf = CRFOutputLayer(4, 8)
    loss = crf.loss(hidden, labels)
    assert loss.shape[0] == 1
    pred = crf(hidden)
    assert pred.shape == (2, 3)