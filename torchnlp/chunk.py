"""
Chunking (Shallow parsing)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .common.hparams import HParams
from .data.conll import conll2000_dataset
from .tasks.sequence_tagging import TransformerTagger
from .tasks.sequence_tagging import BiLSTMTagger
from .tasks.sequence_tagging import hparams_tagging_base
from .tasks.sequence_tagging import train, evaluate, infer, interactive

from .common.prefs import PREFS
from .common.info import Info

import sys, os
import logging
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Info(__doc__).models(TransformerTagger, BiLSTMTagger).datasets(conll2000_dataset)

PREFS.defaults(
    data_root='./.data/conll2000',
    data_train='train.txt',
    data_test='test.txt',
    early_stopping='highest_5_F1'
)


# Default dataset is Conll2000
conll2000 = partial(conll2000_dataset, 
                                    root=PREFS.data_root,
                                    train_file=PREFS.data_train,
                                    test_file=PREFS.data_test)

# Hyperparameter configuration for Chunking tasks

def hparams_transformer_chunk():
    hparams = hparams_tagging_base()
    return hparams.update(
        embedding_size_char=16,
        embedding_size_char_per_word=100,
        num_hidden_layers=2,
        num_heads=4,
        attention_key_channels=0, # Take hidden size
        attention_value_channels=0, # Take hidden size
        filter_size = 128,
        filter_size_char = 64,
        input_dropout=0.2,
        attention_dropout=0.2,
        relu_dropout=0.2,
        learning_rate_decay='noam_step',
        learning_rate_warmup_steps=500,
        use_crf=True
    )

def hparams_lstm_chunk():
    hparams = hparams_tagging_base()

    return hparams.update(
        embedding_size_char=25,
        embedding_size_char_per_word=25,
        num_hidden_layers=2,
        hidden_size=100,
        learning_rate=0.05,
        learning_rate_decay='noam_step',
        learning_rate_warmup_steps=100,
        dropout=0.5,
        use_crf=True
    )

hparams_map = {
    TransformerTagger: hparams_transformer_chunk(),
    BiLSTMTagger: hparams_lstm_chunk()
}

# Add HParams mapping
train = partial(train, hparams_map=hparams_map)
