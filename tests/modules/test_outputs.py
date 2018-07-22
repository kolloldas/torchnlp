from torchnlp.modules.outputs import SoftmaxOutputLayer, CRFOutputLayer

import torch
import numpy as np

def test_softmax_output_layer():
    hidden = torch.randn(2, 3, 4)
    labels = torch.ones(2, 3).long()
    softmax = SoftmaxOutputLayer(4, 8)
    loss = softmax.loss(hidden, labels)
    assert loss > 0
    pred = softmax(hidden)
    assert pred.shape == (2, 3)

def test_crf_output_layer():
    hidden = torch.randn(2, 3, 4)
    labels = torch.ones(2, 3).long()
    crf = CRFOutputLayer(4, 8)
    loss = crf.loss(hidden, labels)
    assert loss > 0
    pred = crf(hidden)
    assert pred.shape == (2, 3)