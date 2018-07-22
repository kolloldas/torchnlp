from torchnlp.modules.transformer import Encoder, Decoder
from torchnlp.modules.transformer.layers import EncoderLayer, DecoderLayer
from torchnlp.modules.transformer.sublayers import MultiHeadAttention, PositionwiseFeedForward

import torch
import numpy as np

def test_multi_head_attention():
    mask = (torch.from_numpy(np.triu(np.ones([10, 10])*float('-Inf'), 1))
        .type(torch.FloatTensor)
        .unsqueeze(0).unsqueeze(1))
    mha = MultiHeadAttention(32, 64, 32, 32, 4, bias_mask=mask)
    t = torch.randn(10, 5, 32)
    output = mha(t, t, t)
    assert output.shape == t.shape


def test_positionwise_feed_forward():
    pwff = PositionwiseFeedForward(32, 64, 16, layer_config='cl')
    assert len(pwff.layers) == 2
    t = torch.randn(10, 5, 32)
    output = pwff(t)
    assert output.shape == (10, 5, 16)

def test_encoder_layer():
    el = EncoderLayer(32, 64, 32, 64, 4)
    t = torch.randn(10, 5, 32)
    output = el(t)
    assert t.shape == output.shape

def test_decoder_layer():
    el = EncoderLayer(32, 64, 32, 64, 4)
    te = torch.randn(10, 5, 32)
    enc_output = el(te)

    dl = DecoderLayer(32, 64, 32, 64, 4, None)
    td = torch.randn(10, 5, 32)
    dec_output, _ = dl((td, enc_output))

    assert td.shape == dec_output.shape


def test_encoder():
    e = Encoder(16, 32, 2, 2, 0, 0, 16)
    te = torch.randn(10, 6, 16)
    enc_output = e(te)

    assert len(e.enc) == 2
    assert enc_output.shape == (10, 6, 32)

def test_decoder():
    e = Encoder(16, 32, 2, 2, 0, 0, 16)
    te = torch.randn(10, 6, 16)
    enc_output = e(te)

    d = Decoder(8, 32, 3, 4, 0, 0, 8)
    td = torch.randn(10, 5, 8)
    dec_output = d(td, enc_output)

    assert len(d.dec) == 3
    assert dec_output.shape == (10, 5, 32)