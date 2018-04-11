"""
Named Entity Recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .common.hparams import HParams
from .data.conll import conll2003_dataset
from .tasks.sequence_tagging import TransformerTagger
from .tasks.sequence_tagging import BiLSTMTagger
from .tasks.sequence_tagging import hparams_transformer_ner, hparams_lstm_ner, hparams_tagging_base
from .tasks.sequence_tagging import train, evaluate, infer, interactive

from .common.prefs import PREFS
from .common.info import Info

import sys, os
import logging
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Info(__doc__).models(TransformerTagger, BiLSTMTagger).datasets(conll2003_dataset)

PREFS.defaults(
    data_root='./.data/conll2003',
    data_train='eng.train.txt',
    data_validation='eng.testa.txt',
    data_test='eng.testb.txt',
    early_stopping='highest_5_F1'
)


# Default dataset is Conll2003
conll2003 = partial(conll2003_dataset, 'ner',  hparams_tagging_base().batch_size,  
                                    root=PREFS.data_root,
                                    train_file=PREFS.data_train,
                                    validation_file=PREFS.data_validation,
                                    test_file=PREFS.data_test)
hparams_transformer = hparams_transformer_ner()
hparams_lstm = hparams_lstm_ner()

hparams_map = {
    TransformerTagger: hparams_transformer,
    BiLSTMTagger: hparams_lstm
}

# Add HParams mapping
train = partial(train, hparams_map=hparams_map)
