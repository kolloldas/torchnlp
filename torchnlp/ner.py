"""
Named Entity Recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .common.hparams import HParams
from .common.evaluation import Evaluator, BasicMetrics, IOBMetrics
from .common.train import Trainer
from .data.conll import conll2003_dataset
from .data.inputs import get_input_processor_words
from .sequence_tagging import TransformerTagger
from .sequence_tagging import BiLSTMTagger
from .sequence_tagging import hparams_transformer_ner, hparams_lstm_ner, hparams_tagging_base

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
    early_stopping='highest_5_F1',
    cur_task='conll2003.ner'
)

# Globals

SPECIALS = set(['<unk>', '<pad>', '<bos>', '<eos>'])

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

def train(model_cls, dataset_fn, hparams=None, num_epochs=100, checkpoint=None,
          early_stopping=None):
    """
    Train a model on a dataset. Saves model and hyperparams so that training
    can be easily resumed
    Parameters:
        model_cls: Model class name. Should inherit from sequence_tagging.Tagger class
        dataset_fn: Dataset function. Must return a dict. See data/conll.py
        hparams: Hyperparameters defined for the particular model. If None then
                it will try to pick one for predefined models. Not required when
                resuming training
        num_epochs: Set to 100. With early stopping set number of epochs will usually
                be lower. Set PREFS.early_stopping to desired criteria
        checkpoint: Saved checkpoint number (train iteration). Used to resume training
        early_stopping: Early stopping criteria. Should be of the form
                        [lowest/highest]_[window_Size]_[metric] where metric is one of
                        loss, acc, acc-seq, precision, recall, F1. Default is picked
                        from PREFS.early_stopping
    """
    dataset = dataset_fn()
    train_iter, validation_iter, _ = dataset['iters']
    _, _, tag_vocab = dataset['vocabs']

    early_stopping=early_stopping or PREFS.early_stopping

    # Save the task name in PREFS
    cur_task = dataset['task']
    PREFS.cur_task = cur_task

    # Create or load the model
    if checkpoint is None:
        if hparams is None:
            hparams = hparams_map[model_cls]
        model = model_cls.create(cur_task, hparams, vocabs=dataset['vocabs'], 
                                overwrite=PREFS.overwrite_model_dir)
    else:
        model = model_cls.load(cur_task, checkpoint)

    # Setup evaluator on the validation dataset
    evaluator = Evaluator(validation_iter,
                    BasicMetrics(output_vocab=tag_vocab),
                    IOBMetrics(tag_vocab=tag_vocab))

    # Setup trainer
    trainer = Trainer(cur_task, model, hparams, train_iter, evaluator)

    # Start training
    best_cp, _ = trainer.train(num_epochs, early_stopping=early_stopping)

    return best_cp


def evaluate(model_cls, dataset_fn, split, checkpoint=-1):
    """
    Evaluate the model on a specific dataset split [train, validation, test].
    Evaluated for loss, accuracy, precision, recall, F1. Model hyperparameters
    are loaded automatically
    Parameters:
        model_cls: Model class name. Must be same as used during training
        dataset_fn: Dataset function. Must return a dict. See data/conll.py
        split: One of the strings train/validation/test
        checkpoint: Checkpoint number (train iteration) to load from. -1 for
                    best/latest checkpoint
    """
    iter_map = {'train': 0, 'validation': 1, 'test': 2}
    dataset = dataset_fn()
    data_iter = dataset['iters'][iter_map[split]]

    model = model_cls.load(dataset['task'], checkpoint)

    # Setup evaluator on the given dataset
    evaluator = Evaluator(data_iter,
                    BasicMetrics(output_vocab=model.vocab_tags),
                    IOBMetrics(tag_vocab=model.vocab_tags))
    metrics = evaluator.evaluate(model)

    print('{} set evaluation: {}-{}'.format(split, dataset['task'], model.__class__.__name__))
    print(', '.join(['{}={:3.5f}'.format(k, v) for k,v in metrics.items()]))

def _run_model_loaded(model, batch):
    predictions = model(batch)
    results = []
    for j in range(predictions.shape[0]):
        pred_indices = list(predictions[j, :].data)
        tags = map(lambda id: model.vocab_tags.itos[id], pred_indices)
        tags = filter(lambda item: item not in SPECIALS, tags)
        results.append(list(tags))

    return results

def infer(inputs, model_cls, task_name=None, checkpoint=-1):
    """
    Run inference on an input sentence. 
    Parameters:
        inputs: Input sentence Either a string or a list of strings
        model_cls: Model class name. Must be same as used during training
        task_name: Name of the task as returned by the dataset function. Default 
                   is picked from PREFS.cur_task
        checkpoint: Checkpoint number (train iteration) to load from. -1 for
                    best/latest checkpoint
    """
    task_name = task_name or PREFS.cur_task
    model = model_cls.load(task_name, checkpoint)
    
    vocab_word, vocab_char, _ = model.vocabs
    input_fn = get_input_processor_words(vocab_word, vocab_char)

    batch = input_fn(inputs)
    
    return _run_model_loaded(model, batch)

def interactive(model_cls, task_name=PREFS.cur_task, checkpoint=-1):
    """
    Run inference interactively
    Parameters:
        model_cls: Model class name. Must be same as used during training
        task_name: Name of the task as returned by the dataset function. Default 
                   is picked from PREFS.cur_task
        checkpoint: Checkpoint number (train iteration) to load from. -1 for
                    best/latest checkpoint
    """
    model = model_cls.load(task_name, checkpoint)

    vocab_word, vocab_char, _ = model.vocabs
    input_fn = get_input_processor_words(vocab_word, vocab_char)

    print('Ctrl+C to quit')
    try:
        while True:
            inputs = input('> ')
            batch = input_fn(inputs)
            output = _run_model_loaded(model, batch)
            print(' '.join(output[0]))
    except:
        pass
    

