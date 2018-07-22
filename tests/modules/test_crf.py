from torchnlp.modules.crf import CRF

import torch
import torch.nn as nn

import numpy as np

def test_sequence_score():
    crf = CRF(3)
    crf.transitions = nn.Parameter(torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]))

    crf.start_transitions = nn.Parameter(torch.Tensor([
        2, 0, 0
    ]))

    crf.stop_transitions = nn.Parameter(torch.Tensor([
        1, 1, 1
    ]))

    feats = torch.Tensor([
        [
            [0, 0, 1],
            [1, 1, 1],
            [2, 3, 2]
        ],
        [
            [0, 0, 2],
            [0, 1, 0],
            [1, 1, 1]
        ]])
    tags = torch.LongTensor([
        [0, 1, 2],
        [0, 2, 1]
    ])

    scores = crf._sequence_score(feats, tags)
    assert scores.shape[0] == 2

    feat_score = torch.Tensor([0+1+2, 0+0+1])
    trans_score = torch.Tensor([2+6, 3+8])
    start_score = torch.Tensor([2, 2])
    stop_score = torch.Tensor([1, 1])

    assert list(scores.data) == list((feat_score + trans_score + start_score + stop_score).data)

def test_log_sum_exp():
    logits = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    crf = CRF(3)
    lse = crf._log_sum_exp(logits, -1)
    assert lse.shape[0] == 2
    assert (lse == logits.exp().sum(-1).log()).all()

def test_partition_function():
    crf = CRF(3)
    crf.transitions = nn.Parameter(torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]))

    crf.start_transitions = nn.Parameter(torch.Tensor([
        2, 0, 0
    ]))

    crf.stop_transitions = nn.Parameter(torch.Tensor([
        1, 1, 1
    ]))

    feats = torch.Tensor([
        [
            [0, 0, 1],
            [1, 1, 1],
            [2, 3, 2]
        ],
        [
            [0, 0, 2],
            [0, 1, 0],
            [1, 1, 1]
        ]])
    
    # Monkey patch _log_sum_exp
    crf._log_sum_exp = lambda logits, dim: logits.sum(dim)
    scores = crf._partition_function(feats)
    """
    2, 0, 1
    3+12+3, 3+15+3, 3+18+3 = 18, 21, 24
    63+12+6, 63+15+9, 63+18+6 = 81, 87, 87
    82+88+88 = 258

    2, 0, 2
    4+12+0, 4+15+3, 4+18+0 = 16, 22, 22
    60+12+3, 60+15+3, 60+18+3 = 75, 78, 81
    76+79+82 = 237
    """
    assert scores.shape[0] == 2
    print(scores.data)
    assert (scores.data == torch.Tensor([258, 237])).all()
    
def test_viterbi():
    crf = CRF(3)
    crf.transitions = nn.Parameter(torch.Tensor([
        [1, 2, 9],
        [4, 8, 6],
        [7, 5, 3]]))

    crf.start_transitions = nn.Parameter(torch.Tensor([
        2, 0, 0
    ]))

    crf.stop_transitions = nn.Parameter(torch.Tensor([
        1, 1, 1
    ]))

    feats = torch.Tensor([
        [
            [0, 0, 1],
            [1, 1, 1],
            [2, 3, 2]
        ],
        [
            [0, 0, 2],
            [0, 1, 0],
            [1, 1, 1]
        ]])

    tags = crf._viterbi(feats)
    """
    2, 0, 1
    8+1, 8+1, 11+1 | 2, 1, 0 = 9, 9, 12
    19+2, 17+3, 18+2 | 2, 1, 0 = 21, 20, 20
    22, 21, 21 | 0 <- 2 <- 0
    
    2, 0, 2
    9+0, 8+1, 11+0 | 2, 1, 0 = 9, 9, 11
    18+1, 17+1, 18+1 | 2, 1, 0 = 19, 18, 19
    20, 19, 20 | 0 <- 2 <- 0
    
    """
    assert tags.shape == (2, 3)
    print(tags.data)
    assert (tags.data == torch.LongTensor([[0, 2, 0],[0, 2, 0]])).all()