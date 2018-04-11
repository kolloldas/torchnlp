from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnlp.common.hparams import HParams

from .tagger import Tagger, hparams_tagging_base

class BiLSTMTagger(Tagger):
    """
    Sequence tagger using bidirectional LSTM. For character embeddings per word
    uses (unidirectional) LSTM
    """
    def __init__(self, hparams=None, **kwargs):
        """
        No additional parameters
        """
        super(BiLSTMTagger, self).__init__(hparams=hparams, **kwargs)

        embedding_size = hparams.embedding_size_word
        if hparams.embedding_size_char > 0:
            embedding_size += hparams.embedding_size_char_per_word
            self.lstm_char = nn.LSTM(hparams.embedding_size_char, 
                                     hparams.embedding_size_char_per_word,
                                     num_layers=1, 
                                     batch_first=True, 
                                     bidirectional=False)


        self.lstm = nn.LSTM(embedding_size, 
                            hparams.hidden_size, 
                            num_layers=hparams.num_hidden_layers,
                            batch_first=True,
                            bidirectional=True
                           )

        self.dropout = nn.Dropout(hparams.dropout)
        self.extra_layer = nn.Linear(2*hparams.hidden_size, hparams.hidden_size)

    def compute(self, inputs_word_emb, inputs_char_emb):
         
        if inputs_char_emb is not None:
            seq_len = inputs_word_emb.shape[1]

            # Compute per word character embeddings using unidirectional LSTM
            _, (h_n, _) = self.lstm_char(inputs_char_emb)
        
            inputs_char_emb = h_n.view(-1, seq_len, 
                                  self.hparams.embedding_size_char_per_word)

            # Combine embeddings
            inputs_word_emb = torch.cat([inputs_word_emb, inputs_char_emb], -1)
        
        hidden, _ = self.lstm(self.dropout(inputs_word_emb))
        hidden_extra = F.tanh(self.extra_layer(hidden))

        return hidden_extra
