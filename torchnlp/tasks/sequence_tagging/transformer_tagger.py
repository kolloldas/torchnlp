from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnlp.common.hparams import HParams
from torchnlp.modules import transformer

from .tagger import Tagger, hparams_tagging_base

class TransformerTagger(Tagger):
    """
    Sequence tagger using the Transformer network (https://arxiv.org/pdf/1706.03762.pdf)
    Specifically it uses the Encoder module. For character embeddings (per word) it uses
    the same Encoder module above which an additive (Bahdanau) self-attention layer is added
    """
    def __init__(self, hparams=None, **kwargs):
        """
        No additional parameters
        """
        super(TransformerTagger, self).__init__(hparams=hparams, **kwargs)

        embedding_size = hparams.embedding_size_word
        if hparams.embedding_size_char > 0:
            embedding_size += hparams.embedding_size_char_per_word
            self.transformer_char = transformer.Encoder(
                                        hparams.embedding_size_char,
                                        hparams.embedding_size_char_per_word,
                                        1,
                                        4,
                                        hparams.attention_key_channels,
                                        hparams.attention_value_channels,
                                        hparams.filter_size_char,
                                        hparams.max_length,
                                        hparams.input_dropout,
                                        hparams.dropout,
                                        hparams.attention_dropout,
                                        hparams.relu_dropout,
                                        use_mask=False
                                    ) 
            self.char_linear = nn.Linear(hparams.embedding_size_char_per_word, 1)

        self.transformer_enc = transformer.Encoder(
                                    embedding_size,
                                    hparams.hidden_size,
                                    hparams.num_hidden_layers,
                                    hparams.num_heads,
                                    hparams.attention_key_channels,
                                    hparams.attention_value_channels,
                                    hparams.filter_size,
                                    hparams.max_length,
                                    hparams.input_dropout,
                                    hparams.dropout,
                                    hparams.attention_dropout,
                                    hparams.relu_dropout,
                                    use_mask=False
                                )

    def compute(self, inputs_word_emb, inputs_char_emb):

        if inputs_char_emb is not None:
            seq_len = inputs_word_emb.shape[1]

            # Process character embeddings to get per word embeddings
            inputs_char_emb = self.transformer_char(inputs_char_emb)

            # Apply additive self-attention to combine outputs
            mask = self.char_linear(inputs_char_emb)
            mask = F.softmax(mask, dim=-1)
            inputs_emb_char = (torch.matmul(mask.permute(0, 2, 1), inputs_char_emb).contiguous()
                            .view(-1, seq_len, self.hparams.embedding_size_char_per_word))

            # Combine embeddings
            inputs_word_emb = torch.cat([inputs_word_emb, inputs_emb_char], -1)

        # Apply Transformer Encoder
        enc_out = self.transformer_enc(inputs_word_emb)

        return enc_out

